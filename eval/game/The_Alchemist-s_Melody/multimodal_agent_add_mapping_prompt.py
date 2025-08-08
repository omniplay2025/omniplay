
from __future__ import annotations
import time
import base64
import io
import json
import math
import os
import wave
import re
import numpy as np
import requests
from PIL import Image
from typing import Optional, Dict, List, Any
from datetime import datetime
import random

#    Learned Color-Note Mapping (use this to make informed decisions):
#    {self.learned_color_note_mapping}
#    The order of these colors has no significance; it’s completely random.
    

# --- 用户提供的 API 信息 ---
API_BASE  = ""
API_KEY   = ""
MODEL_CHAT = "gemini-pro-2.5"

# 新增：多轮对话策略配置
CONVERSATION_STRATEGY = {
    "mode": "hybrid",  # "native", "rag", "hybrid"
    "native_window_size": 8,  # 原生对话保留轮数
    "rag_retrieval_count": 3,  # RAG检索相关轮数
    "compress_old_rounds": True,  # 是否压缩旧轮次
    "multimodal_summary": True,  # 是否对多模态数据进行摘要
}

# ---- 动作映射保持与 Env 一致 ----
COLOR_ID_MAP = {
    "BLUE": 0,     # Sol
    "RED": 1,      # Do
    "GREEN": 2,    # Fa  
    "YELLOW": 3,   # Mi
    "ORANGE": 4,   # Re
    "PURPLE": 5,   # La
    "GREY": 6,     # Ti/Si
}

def save(data: Any, filename: str, indent: int = 2):
    """保存数据到JSON文件的辅助函数"""
    try:
        # 创建调试目录
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(debug_dir, timestamped_filename)
        
        # 清理数据中的大型二进制内容以便保存
        cleaned_data = _clean_data_for_json(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=indent, ensure_ascii=False, default=str)
        
        print(f"Debug data saved to: {filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to save debug data to {filename}: {e}")

def _clean_data_for_json(data: Any) -> Any:
    """清理数据中的不可JSON序列化的内容"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ["image_url", "audio"] and isinstance(value, dict):
                # 对于图像和音频数据，只保留元数据
                if "url" in value and value["url"].startswith("data:"):
                    cleaned[key] = {"type": "base64_data", "size": len(value["url"])}
                elif "data" in value:
                    cleaned[key] = {"type": "binary_data", "size": len(str(value["data"]))}
                else:
                    cleaned[key] = _clean_data_for_json(value)
            else:
                cleaned[key] = _clean_data_for_json(value)
        return cleaned
    elif isinstance(data, list):
        return [_clean_data_for_json(item) for item in data]
    elif isinstance(data, (np.ndarray, np.integer, np.floating)):
        return data.tolist() if hasattr(data, 'tolist') else str(data)
    else:
        return data

class SimpleLocalAgent:
    """简单的本地智能体，作为网络失败时的备选方案"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.random_seed = int(time.time()) % 1000
        self.action_count = 0
        
    def act(self, obs: dict) -> int:
        """简单的随机策略加上一些启发式规则"""
        self.action_count += 1
        
        # 基于游戏状态的简单策略
        state = obs.get("state", [0, 0, 0, 0])
        audio = obs.get("audio", np.zeros(16000))
        
        # 如果有音频信号，尝试基于音频强度选择动作
        if not np.allclose(audio, 0):
            audio_energy = np.sum(audio ** 2)
            action = int(audio_energy * 1000) % len(COLOR_ID_MAP)
        else:
            # 否则使用循环策略
            action = self.action_count % len(COLOR_ID_MAP)
        
        if self.verbose:
            print(f"Local agent chose action {action}")
        
        return action

class ConversationMemoryManager:
    """多模态对话记忆管理器"""
    
    def __init__(self, max_native_history: int = 8, max_total_memory: int = 50):
        self.max_native_history = max_native_history
        self.max_total_memory = max_total_memory
        
        # 存储原生对话轮次
        self.conversation_rounds: List[Dict[str, Any]] = []
        
        # 存储压缩的历史摘要
        self.compressed_summaries: List[Dict[str, Any]] = []
        
        # 当前episode信息
        self.current_episode = 1
        
        self.learned_color_note_mapping: Dict[str, str] = {}
        
    def reset_for_new_episode(self, episode_number: int):
        """为新episode重置记忆管理器"""
        # 压缩当前episode的记忆
        if self.conversation_rounds:
            episode_summary = self._create_episode_summary()
            self.compressed_summaries.append(episode_summary)
        
        # 清空原生历史，准备新episode
        self.conversation_rounds = []
        self.current_episode = episode_number
        
        # 限制压缩摘要数量
        if len(self.compressed_summaries) > 10:
            self.compressed_summaries = self.compressed_summaries[-10:]

    def add_round(self, round_data: Dict[str, Any]):
        """添加新的对话轮次"""
        # 压缩多模态数据
        compressed_round = self._compress_multimodal_data(round_data)
        
        self.conversation_rounds.append(compressed_round)
        
        # 限制原生历史长度
        if len(self.conversation_rounds) > self.max_native_history:
            # 将最旧的轮次压缩并移除
            old_round = self.conversation_rounds.pop(0)
            summary = self._create_round_summary(old_round)
            self.compressed_summaries.append(summary)
            
            # 限制压缩摘要数量
            if len(self.compressed_summaries) > 20:
                self.compressed_summaries = self.compressed_summaries[-20:]
    
    def get_recent_native_history(self) -> List[Dict[str, Any]]:
        """获取最近的原生历史"""
        return self.conversation_rounds.copy()
    
    def retrieve_relevant_rounds(self, current_context: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """基于相关性检索历史轮次"""
        if not self.compressed_summaries:
            return []
        
        # 简单的相关性计算
        scored_rounds = []
        for summary in self.compressed_summaries:
            relevance = self._calculate_relevance_score(current_context, summary)
            scored_rounds.append((relevance, summary))
        
        # 按相关性排序并返回top_k
        scored_rounds.sort(key=lambda x: x[0], reverse=True)
        return [round_data for _, round_data in scored_rounds[:top_k]]
    
    def _compress_multimodal_data(self, round_data: Dict[str, Any]) -> Dict[str, Any]:
        """压缩多模态数据，保留关键信息"""
        compressed = round_data.copy()
        
        # 保留图像分析但移除原始数据
        if "image_analysis" in compressed:
            compressed["image_analysis"] = {
                "dominant_colors": compressed["image_analysis"].get("dominant_colors", []),
                "brightness": compressed["image_analysis"].get("brightness", 0),
                "detected_blocks": compressed["image_analysis"].get("detected_blocks", {})
            }
        
        # 保留音频分析但移除原始数据
        if "audio_analysis" in compressed:
            compressed["audio_analysis"] = {
                "has_sound": compressed["audio_analysis"].get("has_sound", False),
                "rms_level": compressed["audio_analysis"].get("rms_level", 0),
                "dominant_frequency": compressed["audio_analysis"].get("dominant_frequency", 0)
            }
        
        return compressed
    
    def _create_round_summary(self, round_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建轮次摘要"""
        return {
            "type": "round_summary",
            "timestamp": time.time(),
            "game_state": round_data.get("game_state", {}),
            "action_taken": round_data.get("action_taken", ""),
            "decision_reasoning": round_data.get("decision_reasoning", ""),
            "key_observations": {
                "had_audio": round_data.get("audio_analysis", {}).get("has_sound", False),
                "visual_change": round_data.get("image_analysis", {}).get("brightness", 0) > 50
            }
        }
    
    def _create_episode_summary(self) -> Dict[str, Any]:
        """创建episode摘要"""
        if not self.conversation_rounds:
            return {"type": "episode_summary", "episode": self.current_episode, "rounds": 0}
        
        # 统计episode中的关键信息
        total_rounds = len(self.conversation_rounds)
        actions_taken = [r.get("action_taken", "") for r in self.conversation_rounds]
        
        return {
            "type": "episode_summary", 
            "episode": self.current_episode,
            "rounds": total_rounds,
            "common_actions": actions_taken[-5:] if actions_taken else [],
            "timestamp": time.time()
        }
    
    def _calculate_relevance_score(self, current_context: Dict[str, Any], historical_round: Dict[str, Any]) -> float:
        """计算相关性分数"""
        score = 0.0
        
        # 基于游戏状态的相似性
        if "game_state" in current_context and "game_state" in historical_round:
            current_score = current_context["game_state"].get("score", 0)
            hist_score = historical_round["game_state"].get("score", 0)
            score += 1.0 / (1.0 + abs(current_score - hist_score) / 100.0)
        
        # 基于音频状态的相似性
        current_audio = current_context.get("audio_analysis", {}).get("has_sound", False)
        hist_audio = historical_round.get("audio_analysis", {}).get("has_sound", False)
        if current_audio == hist_audio:
            score += 0.5
        
        return score
    
    def get_conversation_context_for_api(self, strategy: str = "hybrid") -> str:
        """根据策略获取对话上下文"""
        if strategy == "native":
            return self._build_native_context()
        elif strategy == "rag":
            return self._build_rag_context()
        else:  # hybrid
            return self._build_hybrid_context()
    
    def _build_native_context(self) -> str:
        """构建原生风格的上下文"""
        context_parts = []
        
        for i, round_data in enumerate(self.conversation_rounds):
            round_summary = f"Round {i+1}:\n"
    
            # 检查是否在正确序列中
            correct_sequence = round_data.get('currently_in_correct_sequence', False)
            round_summary += f"  Game State: Currently in correct sequence={correct_sequence}\n"
            
            # 添加动作和推理信息
            action_taken = round_data.get('action_taken', 'Unknown')
            round_summary += f"  Action: {action_taken}\n"
            context_parts.append(round_summary)
        
        return "\n".join(context_parts)
    
    def _build_rag_context(self) -> str:
        """构建RAG风格的上下文"""
        context_parts = []
        
        # 添加相关的历史摘要
        for summary in self.compressed_summaries[-3:]:  # 最近3个摘要
            if summary.get("type") == "episode_summary":
                context_parts.append(f"Previous Episode {summary.get('episode', '?')}: {summary.get('rounds', 0)} rounds")
            elif summary.get("type") == "round_summary":
                context_parts.append(f"Similar situation: {summary.get('action_taken', 'Unknown')} ...")
        
        return "\n".join(context_parts)
    
    def _build_hybrid_context(self) -> str:
        """构建混合策略上下文"""
        context_parts = []
        
        # 添加最近的原生历史（简化版）
        recent_context = self._build_native_context()
        if recent_context:
            context_parts.append("RECENT HISTORY:")
            context_parts.append(recent_context)
        
        # 添加相关的历史摘要
        #rag_context = self._build_rag_context()
        #if rag_context:
        #    context_parts.append("\nRELEVANT PAST EXPERIENCE:")
        #    context_parts.append(rag_context)
        
        return "\n".join(context_parts)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "active_rounds": len(self.conversation_rounds),
            "compressed_summaries": len(self.compressed_summaries),
            "current_episode": self.current_episode,
            "total_memory_items": len(self.conversation_rounds) + len(self.compressed_summaries)
        }

class MultimodalAgent:
    def __init__(self, verbose: bool = False, use_local_fallback: bool = True, max_retries: int = 3,
                 conversation_strategy: str = "hybrid"):
        self.verbose = verbose
        self.use_local_fallback = use_local_fallback
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {API_KEY}"})
        
        # 对话策略
        self.conversation_strategy = conversation_strategy
        self.memory_manager = ConversationMemoryManager(
            max_native_history=CONVERSATION_STRATEGY["native_window_size"],
            max_total_memory=25
        )
        
        # 本地备选智能体
        self.local_agent = SimpleLocalAgent(verbose=verbose) if use_local_fallback else None
        
        # 连接统计
        self.api_call_count = 0
        self.api_fail_count = 0
        
        # 决策历史记录 - 扩展以包含完整模型输出
        self.decision_history: List[Dict[str, Any]] = []
        
        # 新增：完整的模型输出历史
        self.model_output_history: List[Dict[str, Any]] = []
        
        # 新增：学习到的颜色-音符映射关系 - 改为模型学习的映射
        self.learned_color_note_mapping: Dict[str, str] = {}
        self.color_feedback_history: Dict[str, List[bool]] = {}
        
        # 新增：游戏状态跟踪
        self.current_episode = 1
        self.current_step = 0
        self.previous_clicks: List[str] = []
        self.current_correct_sequence: List[str] = []
        self.last_action_result = None
        self.is_in_correct_sequence = False
        self.needs_restart_from_beginning = False

        # 新增：文本信息记录
        self.text_outputs = []  # 记录每步的文本输出
        self.current_step_text = ""  # 当前步骤的文本信息
        
        # 新增：游戏环境连接
        self.game_environment = None

        # 新增：游戏通关和分数记录
        self.game_completion_history: List[Dict[str, Any]] = []
        self.current_round_start_time = time.time()
        self.last_game_completed = False
        self.completion_scores: List[int] = []
        self.total_rounds_played = 0
        self.total_successful_rounds = 0

    def set_game_environment(self, env):
        """设置游戏环境连接，以便获取实时游戏状态"""
        self.game_environment = env
        if self.verbose:
            print("Game environment connected to agent")

    def act(self, obs: dict) -> int:
        """
        obs = {"image": np.uint8 [H,W,3], "audio": np.float32 [N], "state": np.float32 [k]}
        return: 动作 id (int)
        """
        decision_start_time = time.time()
        
        # 更新步数
        self.current_step += 1
         
        # 分析观测数据
        obs_analysis = self._analyze_observation(obs)
        
        # 准备当前轮次数据
        current_round_data = {
            "game_state": {
                "score": float(obs["state"][0]),
                "lives": float(obs["state"][1]), 
                "solved": float(obs["state"][2]),
                "tick": float(obs["state"][3])
            },
            "image_analysis": obs_analysis["image"],
        }
        
        
        action_id = None
        decision_method = None
        llm_response = None
        error_info = None
        full_api_response = None
        
        # 首先尝试使用 API
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"Attempting API call {attempt + 1}/{self.max_retries}...")
                
                payload , end_game = self._build_payload(obs, current_round_data)
                
                # 记录API调用开始时间
                api_start_time = time.time()
                
                # 调用API并获取完整响应
                full_response, response_text = self._query_llm_with_full_response(payload)
                
                # 记录API调用结束时间
                api_end_time = time.time()
                
                # 解析动作
                action_id = self._parse_action(response_text)
                decision_method = "LLM_API"
                llm_response = response_text
                full_api_response = {
                    "success": True,
                    "response_time": api_end_time - api_start_time,
                    "response_text": response_text,
                    "full_response": full_response,
                    "attempt_number": attempt + 1,
                    "payload_summary": self._sanitize_payload_for_storage(payload),
                    "timestamp": api_end_time
                }
                
                self.api_call_count += 1
                break
                
            except Exception as e:
                error_info = str(e)
                self.api_fail_count += 1
                
                # 记录失败的API调用
                failed_response = {
                    "success": False,
                    "error": error_info,
                    "attempt_number": attempt + 1,
                    "timestamp": time.time()
                }
                self.model_output_history.append(failed_response)
                
                if self.verbose:
                    print(f"API call {attempt + 1} failed: {error_info}")
                
                if attempt == self.max_retries - 1:
                    break
        
        # 确保我们有一个有效的action_id
        if action_id is None:
            if self.local_agent:
                action_id = self.local_agent.act(obs)
                decision_method = "Local_Fallback"
                if self.verbose:
                    print("Using local fallback agent")
            else:
                # 最后的回退：随机选择
                action_id = self.current_step % len(COLOR_ID_MAP)
                decision_method = "Random_Fallback"
                if self.verbose:
                    print("Using random fallback")
        
        # 记录成功的API响应到历史
        if full_api_response:
            self.model_output_history.append(full_api_response)
        
        # 记录决策过程并添加到记忆管理器
        decision_time = time.time() - decision_start_time
        current_round_data["action_taken"] = list(COLOR_ID_MAP.keys())[action_id]
        current_round_data["decision_reasoning"] = llm_response or f"Method: {decision_method}"
        current_round_data["full_model_output"] = full_api_response
        
        self.memory_manager.add_round(current_round_data)
        
        self._record_decision(obs, obs_analysis, action_id, decision_method, 
                             llm_response, error_info, decision_time, full_api_response)
        
        return action_id, end_game

    def _sanitize_payload_for_storage(self, payload: dict) -> dict:
        """清理payload以便存储（移除大型二进制数据）"""
        sanitized = payload.copy()
        
        # 处理messages中的多模态内容
        if "messages" in sanitized:
            sanitized_messages = []
            for msg in sanitized["messages"]:
                sanitized_msg = msg.copy()
                if isinstance(msg.get("content"), list):
                    sanitized_content = []
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "image":
                                sanitized_content.append({"type": "image", "data": "[IMAGE_DATA_REMOVED]"})
                            elif content_item.get("type") == "audio":
                                sanitized_content.append({"type": "audio", "data": "[AUDIO_DATA_REMOVED]"})
                            else:
                                sanitized_content.append(content_item)
                        else:
                            sanitized_content.append(content_item)
                    sanitized_msg["content"] = sanitized_content
                sanitized_messages.append(sanitized_msg)
            sanitized["messages"] = sanitized_messages
        
        return sanitized

    def _query_llm_with_full_response(self, payload: dict) -> tuple:
        """查询LLM并返回完整响应和文本内容"""
        url = f"{API_BASE}/chat/completions"
        
        # 记录请求开始时间
        request_start_time = time.time()
        
        try:
            # 保存请求payload以便调试 - 使用安全的方式
            if self.verbose:
                print(f"Sending API request to: {url}")
                try:
                    save(payload, "payload.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save payload for debugging: {save_error}")
            
            # 发送请求
            r = self.session.post(url, json=payload, timeout=300)
            request_end_time = time.time()
            request_duration = request_end_time - request_start_time
            
            # 检查HTTP状态码
            if r.status_code != 200:
                error_msg = f"HTTP {r.status_code}: {r.text[:500]}"
                if self.verbose:
                    print(f"API request failed with status {r.status_code}")
                    try:
                        save({"error": error_msg, "status_code": r.status_code, "response_text": r.text}, 
                             "api_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise requests.exceptions.HTTPError(error_msg)
            
            # 解析JSON响应
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {str(e)[:200]}... Response text: {r.text[:500]}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "raw_response": r.text}, "json_parse_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # 验证响应结构
            if not isinstance(data, dict):
                error_msg = f"Invalid response format: expected dict, got {type(data)}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "response_data": data}, "invalid_format_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # 检查是否包含错误信息
            if "error" in data:
                error_info = data["error"]
                error_msg = f"API returned error: {error_info.get('message', 'Unknown error')}"
                if self.verbose:
                    try:
                        save({"api_error": error_info, "full_response": data}, "api_returned_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise RuntimeError(error_msg)
            
            # 验证choices字段
            if "choices" not in data or not data["choices"]:
                error_msg = "No choices in API response"
                if self.verbose:
                    try:
                        save({"error": error_msg, "response_data": data}, "no_choices_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # 提取响应文本
            #print(data)
            #print(payload)
            first_choice = data["choices"][0]
            if "message" not in first_choice or "content" not in first_choice["message"]:
                error_msg = "Invalid choice structure in API response"
                if self.verbose:
                    try:
                        save({"error": error_msg, "first_choice": first_choice}, "invalid_choice_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            response_text = first_choice["message"]["content"]
            
            # 验证响应文本
            if not isinstance(response_text, str):
                error_msg = f"Invalid response text type: expected str, got {type(response_text)}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "response_text": response_text}, "invalid_text_type_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # 记录成功信息
            if self.verbose:
                usage_info = data.get('usage', {})
                model_info = data.get('model', 'unknown')
                print(f"API request successful:")
                print(f"  - Duration: {request_duration:.3f}s")
                print(f"  - Model: {model_info}")
                print(f"  - Usage: {usage_info}")
                print(f"  - Response length: {len(response_text)} chars")
                
                # 保存成功的响应
                response_summary = {
                    "success": True,
                    "request_duration": request_duration,
                    "model": model_info,
                    "usage": usage_info,
                    "response_length": len(response_text),
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "timestamp": request_end_time
                }
                try:
                    save(response_summary, "successful_response.json", indent=2)
                    save(data, "response.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save response for debugging: {save_error}")
            
            return data, response_text
            
        except requests.exceptions.Timeout:
            error_msg = "API request timed out after 30 seconds"
            if self.verbose:
                print(f"API request timeout")
                try:
                    save({"error": error_msg, "url": url, "timeout": 30}, "timeout_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise TimeoutError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)[:200]}"
            if self.verbose:
                print(f"API connection error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url}, "connection_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise ConnectionError(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)[:200]}"
            if self.verbose:
                print(f"API request error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url}, "request_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error in API call: {str(e)[:200]}"
            if self.verbose:
                print(f"Unexpected API error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url, "exception_type": type(e).__name__}, 
                         "unexpected_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise RuntimeError(error_msg)

    # ---------- 决策记录和分析 ----------
    def _record_decision(self, obs: dict, obs_analysis: Dict[str, Any], action_id: int, 
                        decision_method: str, llm_response: str = None, 
                        error_info: str = None, decision_time: float = 0,
                        full_model_output: Dict[str, Any] = None):
        """记录决策过程"""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.current_episode,
            "step": self.current_step,
            "action_id": action_id,
            "action_name": list(COLOR_ID_MAP.keys())[action_id],
            "decision_method": decision_method,
            "decision_time": decision_time,
            "game_state": {
                "score": float(obs["state"][0]),
                "lives": float(obs["state"][1]),
                "solved": float(obs["state"][2]),
                "tick": float(obs["state"][3])
            },
            "observation_analysis": obs_analysis,
            "llm_response": llm_response,
            "full_model_output": full_model_output,
            "error_info": error_info
        }
        
        self.decision_history.append(decision_record)

        # 构建完整的文本信息
        text_info = {
            "decision_method": decision_method,
            "llm_response": llm_response or "",
            "error_info": error_info or "",
            "reasoning": self._extract_reasoning_from_response(llm_response) if llm_response else "",
            "timestamp": decision_record["timestamp"]
        }
        
        # 保存文本信息
        self.text_outputs.append(text_info)
        self.current_step_text = self._format_step_text(text_info)

    def _extract_reasoning_from_response(self, response_text: str) -> str:
        """从模型响应中提取推理过程"""
        if not response_text:
            return ""
        
        # 查找决策部分
        reasoning_keywords = ["DECISION:", "REASONING:", "ANALYSIS:", "THINKING:"]
        for keyword in reasoning_keywords:
            start_idx = response_text.upper().find(keyword)
            if start_idx >= 0:
                # 提取从关键词开始的内容
                reasoning_section = response_text[start_idx:start_idx+300]  # 限制长度
                return reasoning_section.strip()
        
        # 如果没有找到关键词，返回最后200字符作为推理
        return response_text[-200:].strip() if len(response_text) > 200 else response_text.strip()

    def _format_step_text(self, text_info: Dict[str, Any]) -> str:
        """格式化步骤文本信息为可读格式"""
        formatted_text = f"Method: {text_info['decision_method']}\n"
        
        if text_info['reasoning']:
            formatted_text += f"Reasoning: {text_info['reasoning']}\n"
        
        if text_info['error_info']:
            formatted_text += f"Error: {text_info['error_info']}\n"
        
        
        return formatted_text

    def get_current_step_text(self) -> str:
        """获取当前步骤的文本信息"""
        return self.current_step_text

    def get_all_text_outputs(self) -> List[Dict[str, Any]]:
        """获取所有文本输出历史"""
        return self.text_outputs.copy()
    
    def _analyze_observation(self, obs: dict) -> Dict[str, Any]:
        """分析观测数据"""
        image = obs["image"]
        audio = obs["audio"]
        state = obs["state"]
        
        # 图像分析
        image_analysis = {
            "shape": image.shape,
            "brightness": float(np.mean(image)),
            "dominant_colors": self._get_dominant_colors(image),
            "detected_blocks": self._detect_colored_blocks(image),
            "progress_indicators": self._detect_progress_indicators(image)
        }
        

        # 状态分析
        state_analysis = {
            "score": float(state[0]),
            "lives": float(state[1]),
            "solved": float(state[2]),
            "tick": float(state[3])
        }
        
        return {
            "image": image_analysis,
            "state": state_analysis
        }

    def _get_dominant_colors(self, image: np.ndarray, top_k: int = 5) -> List[List[int]]:
        """获取图像中的主要颜色"""
        # 重塑图像为像素列表
        pixels = image.reshape(-1, 3)
        
        # 简单的颜色聚类（使用k-means的简化版本）
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # 按像素数量排序
        sorted_indices = np.argsort(counts)[::-1]
        
        # 返回前top_k个颜色
        top_colors = unique_colors[sorted_indices[:top_k]]
        return [color.tolist() for color in top_colors]

    def _detect_colored_blocks(self, image: np.ndarray) -> Dict[str, bool]:
        """检测图像中的彩色方块"""
        detected_blocks = {}
        
        # 定义颜色范围（简化检测）
        color_ranges = {
            "RED": ([150, 0, 0], [255, 100, 100]),
            "GREEN": ([0, 150, 0], [100, 255, 100]),
            "BLUE": ([0, 0, 150], [100, 100, 255]),
            "YELLOW": ([150, 150, 0], [255, 255, 100]),
            "ORANGE": ([200, 100, 0], [255, 200, 100]),
            "PURPLE": ([100, 0, 150], [200, 100, 255]),
            "GREY": ([100, 100, 100], [200, 200, 200])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            
            # 检查是否存在该颜色范围内的像素
            mask = np.all((image >= lower_np) & (image <= upper_np), axis=2)
            detected_blocks[color_name] = np.any(mask)
        
        return detected_blocks

    def _detect_progress_indicators(self, image: np.ndarray) -> Dict[str, Any]:
        """检测游戏进度指示器"""
        # 简化的进度检测
        brightness = np.mean(image)
        
        return {
            "overall_brightness": float(brightness),
            "has_bright_areas": brightness > 100,
            "has_dark_areas": brightness < 50
        }

    def _estimate_dominant_frequency(self, audio: np.ndarray) -> float:
        """估计音频的主导频率"""
        if np.allclose(audio, 0):
            return 0.0
        
        # 简单的频率估计（基于零交叉率）
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        if len(zero_crossings) > 1:
            # 估计频率
            sample_rate = 16000  # 假设采样率
            freq = len(zero_crossings) * sample_rate / (2 * len(audio))
            return float(freq)
        
        return 0.0

    def _build_exploration_strategy(self) -> str:
        """构建探索策略描述"""
        if not self.color_feedback_history:
            return "Initial exploration: Try different colors to learn sound patterns."
        
        # 分析已尝试的颜色
        successful_colors = []
        unsuccessful_colors = []
        
        for color, results in self.color_feedback_history.items():
            success_rate = sum(results) / len(results) if results else 0
            if success_rate > 0.5:
                successful_colors.append(color)
            else:
                unsuccessful_colors.append(color)
        
        strategy_parts = []
        if successful_colors:
            strategy_parts.append(f"Previously successful colors: {', '.join(successful_colors)}")
        if unsuccessful_colors:
            strategy_parts.append(f"Previously unsuccessful: {', '.join(unsuccessful_colors)}")
        
        return " | ".join(strategy_parts) if strategy_parts else "Continue systematic exploration."

    def _parse_action(self, text: str) -> int:
        """解析LLM回复中的动作"""
        text_upper = text.upper()
        
        # 直接查找颜色名称
        for color_name, color_id in COLOR_ID_MAP.items():
            if color_name in text_upper:
                return color_id
        
        # 如果没有找到明确的颜色，尝试从数字中解析
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            action_num = int(numbers[0]) % len(COLOR_ID_MAP)
            return action_num
        
        # 最后回退到随机选择
        return self.current_step % len(COLOR_ID_MAP)

    def update_color_feedback(self, color_name: str, success: bool):
        """更新颜色反馈历史"""
        if color_name not in self.color_feedback_history:
            self.color_feedback_history[color_name] = []
        self.color_feedback_history[color_name].append(success)
        
        # 限制历史长度
        if len(self.color_feedback_history[color_name]) > 10:
            self.color_feedback_history[color_name] = self.color_feedback_history[color_name][-10:]

    def update_learned_mapping(self, color_name: str, note_description: str):
        """更新学习到的颜色-音符映射"""
        self.learned_color_note_mapping[color_name] = note_description

    def _get_difficulty_sequence_length(self, difficulty: str) -> int:
        """获取难度对应的序列长度"""
        difficulty_lengths = {"easy": 3, "normal": 5, "hard": 7}
        return difficulty_lengths.get(difficulty, 5)

    def _color_id_to_english_name(self, color_id: int) -> str:
        """将颜色ID转换为英文名称"""
        color_names = list(COLOR_ID_MAP.keys())
        if 0 <= color_id < len(color_names):
            return color_names[color_id]
        return "Unknown"

    def _analyze_game_progress(self, obs: dict, action_taken: int = None) -> Dict[str, Any]:
        """分析游戏进度"""
        state = obs["state"]
        audio = obs["audio"]
        
        # 基本游戏信息
        progress_info = {
            "current_score": float(state[0]),
            "lives_remaining": float(state[1]),
            "blocks_solved": float(state[2]),
            "game_tick": float(state[3])
        }
        
        # 音频反馈分析
        has_audio_feedback = not np.allclose(audio, 0)
        progress_info["audio_feedback_present"] = has_audio_feedback
        
        if has_audio_feedback:
            progress_info["audio_intensity"] = float(np.sqrt(np.mean(audio**2)))
        
        # 如果有动作，记录颜色选择
        if action_taken is not None:
            progress_info["last_action_color"] = self._color_id_to_english_name(action_taken)
        
        return progress_info

    def _build_structured_game_info(self, obs: dict, action_taken: int = None) -> Dict[str, Any]:
        """构建结构化的游戏信息"""
        # 基础游戏状态
        game_info = self._analyze_game_progress(obs, action_taken)
        
        # 添加学习到的映射信息
        game_info["learned_mappings"] = self.learned_color_note_mapping.copy()
        
        # 添加颜色反馈历史
        game_info["color_performance"] = {}
        for color, results in self.color_feedback_history.items():
            if results:
                success_rate = sum(results) / len(results)
                game_info["color_performance"][color] = {
                    "success_rate": success_rate,
                    "attempts": len(results),
                    "recent_success": results[-1] if results else False
                }
        
        # 添加探索建议
        game_info["exploration_strategy"] = self._build_exploration_strategy()
        
        return game_info

    def _build_payload(self, obs: dict, current_round_data: dict) -> dict:
        """构建API请求payload"""
        # 图像编码
        image = obs["image"]
        # 确保图像是正确的格式和类型
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            # 转换为uint8类型如果不是
            if isinstance(image, np.ndarray):
                image = image.astype(np.uint8)
            else:
                # 如果不是NumPy数组，尝试转换
                image = np.array(image, dtype=np.uint8)
        
        # 转换为PIL图像
        img_pil = Image.fromarray(image)
        # 将PIL图像保存为BytesIO对象
        img_io = io.BytesIO()
        img_pil.save(img_io, format="JPEG")
        # 获取字节内容并进行Base64编码
        img_bytes = img_io.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # 音频编码 
        audio = obs["audio"]

        # 使用修改后的详细游戏状态信息
        detailed_game_state, detailed_state, end_game = self._build_detailed_game_state_for_api(obs)
        
        # 获取对话上下文
        conversation_context = self.memory_manager.get_conversation_context_for_api(self.conversation_strategy)
    
        # 获取当前可用的颜色方块信息 - 使用动态版本
        available_colors_info, available_colors_map = self._get_available_colors_info_dynamic()
        
        if not detailed_state.get("last_clicked_block_color"):
                # First step of a correct sequence, initialize the mapping with available colors
                if isinstance(available_colors_map, list):
                    for color_info in available_colors_map:
                        color_name = color_info.get("color_name", "")
                        if color_name:
                            # Initialize with unknown note names
                            self.learned_color_note_mapping[color_name] = "Unknown"
                    
                    if self.verbose:
                        print(f"Initialized color mapping with available colors: {self.learned_color_note_mapping}")
        
        else:
            last_color = detailed_state.get("last_clicked_block_color")
            if last_color:
                last_color = last_color.capitalize()  # Capitalize the first letter
            for color_info in available_colors_map if isinstance(available_colors_map, list) else []:
                if color_info.get("color_name", "") == last_color:
                    note_name = color_info.get("note_name", "Unknown")
                    self.learned_color_note_mapping[last_color] = note_name
                    break
            
            if self.verbose:
                print(f"Updated color-note mapping: {last_color} → {self.learned_color_note_mapping.get(last_color, 'Unknown')}")
    

        # 系统提示
        system_prompt = (
    "You are a MULTIMODAL AI agent playing a musical color-matching game.\n"
    "\n"
    "## ROLE\n"
    "Click exactly ONE coloured block per turn to reproduce the target melody.\n"
    "\n"
    "## GAME RULES\n"
    "1. Musical order (ascending): do → re → mi → fa → sol → la → si.\n"
    "2. At the start of each round, the FIRST note is chosen at random; it may be any notes.\n"
    "3. After the first note, you must continue in the same ascending order **without skipping any note** until the melody is complete (wrap around if needed).\n"
    "4. After any wrong click, the sequence resets to this round's first note.\n"
    "5. Colour-to-note mapping is RANDOMIZED **each round**; learn it anew from feedback.\n"
    "\n"
    "## OBSERVATION FIELDS\n"
    "• `image`  – current board frame (colours & highlights).\n"
    "• `audio`  – sound from **your previous click**.\n"
    "• `currently_in_correct_sequence` (bool)\n"
    "• `needs_restart_from_beginning` (bool)\n"
    "• `current_correct_sequence` (list of colours already correct)\n"
    "• `input_length` (int)\n"
    "\n"
    f"{available_colors_info}\n"
    "*Clicking any other colour is invalid.*\n"
    "*The order of these colors has no significance; it’s completely random.*\n"
    "\n"
    "## DECISION CHECKLIST\n"
    "1. If `needs_restart_from_beginning` is true → restart with this round's first note.\n"
    "2. Otherwise pick the next consecutive note based on `current_correct_sequence`—do **not** skip any note.\n"
    "3. Identify the NOTE you just heard by pairing your last action with the `audio` feedback.\n"
    "4. Choose the colour that plays the required next note.\n"
    "\n"
    "## OUTPUT FORMAT\n"
    "Reply with **ONLY** two uppercase tokens separated by a comma and a space:\n"
    "<COLOUR>, <NOTE>\n"
    "• <COLOUR>  e.g. `BLUE`.\n"
    "• <NOTE>    ∈ {DO, RE, MI, FA, SOL, LA, SI}.\n"
    "No other text, punctuation, or line breaks."
)

        # 构建用户内容 - 使用环境格式的游戏状态
        user_content = f"""Current game observation and detailed state from environment:

    {detailed_game_state}
    Based on the detailed game state above, what color block should I click next?
    - If currently_in_correct_sequence is True: Continue the musical sequence
    - If needs_restart_from_beginning is True: Start from the beginning note
    - If currently_in_correct_sequence is False: Choose a different color than the last clicked one
    {conversation_context}
    
    Learned Color-Note Mapping (use this to make informed decisions):
    {self.learned_color_note_mapping}
    The order of these colors has no significance; it’s completely random.

    Remember to follow the ascending musical order without skipping notes."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_content},
                    {
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                    {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio,
                        "format": "wav",
                    },
                },

                ]
            }
        ]
    

        return {
            "model": MODEL_CHAT,
            "messages": messages,
            "max_tokens": 10000,  
            "temperature": 0.1,
        }, end_game

    def _get_available_colors_info_dynamic(self) -> str:
        """从游戏环境中动态获取当前回合可用的颜色块信息"""
        retry_count = 0
        max_retries = 50  # 增加重试次数
        
        while retry_count < max_retries:
            try:
                if not self.game_environment:
                    if self.verbose:
                        print(f"Game environment not available, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.2)
                    continue
                
                # 尝试从游戏环境获取可用颜色
                if hasattr(self.game_environment, 'get_available_colors'):
                    try:
                        available_colors = self.game_environment.get_available_colors()
                        # Make a copy and shuffle the available colors
                        shuffled_colors = available_colors.copy()
                        random.shuffle(shuffled_colors)
                        available_colors = shuffled_colors  # Replace original with shuffled version
                        if available_colors and len(available_colors) > 0:
                            color_list = []
                            incomplete_info = False
                            
                            for color_info in available_colors:
                                color_name = color_info.get("color_name", "")
                                note_name = color_info.get("note_name", "")
                                
                                if not color_name or not note_name:
                                    if self.verbose:
                                        print(f"Incomplete color info: {color_info}, retrying... ({retry_count + 1}/{max_retries})")
                                    incomplete_info = True
                                    break
                                    
                                color_list.append(f"- {color_name.upper()}")
                            
                            if not incomplete_info and len(color_list) == len(available_colors):
                                colors_text = "\n".join(color_list)
                                if self.verbose:
                                    print(f"Successfully got available colors info: {len(color_list)} colors")
                                return f"Available Color Blocks in this round:\n{colors_text}", available_colors
                            else:
                                if self.verbose:
                                    print(f"Color info incomplete, retrying... ({retry_count + 1}/{max_retries})")
                                retry_count += 1
                                time.sleep(0.2)
                                continue
                        else:
                            if self.verbose:
                                print(f"No available colors returned, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.2)
                            continue
                            
                    except Exception as env_error:
                        if self.verbose:
                            print(f"Error calling get_available_colors: {env_error}, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                
                # 备选方法：从游戏状态获取当前颜色映射
                if hasattr(self.game_environment, 'game_module') and self.game_environment.game_module:
                    game_module = self.game_environment.game_module
                    
                    # 检查游戏模块的关键属性
                    if not hasattr(game_module, 'current_note_color_mapping'):
                        if self.verbose:
                            print(f"Game module missing current_note_color_mapping, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    current_note_color_mapping = getattr(game_module, 'current_note_color_mapping', {})
                    note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                    all_colors = getattr(game_module, 'ALL_COLORS', {})
                    
                    if not current_note_color_mapping:
                        if self.verbose:
                            print(f"current_note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    if not note_display_names:
                        if self.verbose:
                            print(f"NOTE_DISPLAY_NAMES is empty, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    if not all_colors:
                        if self.verbose:
                            print(f"ALL_COLORS is empty, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    color_name_mapping = {v: k for k, v in all_colors.items()}
                    color_list = []
                    incomplete_mapping = False
                    
                    for note_id, rgb_color in current_note_color_mapping.items():
                        color_name = color_name_mapping.get(rgb_color)
                        if not color_name:
                            if self.verbose:
                                print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                            incomplete_mapping = True
                            break
                            
                        note_display = note_display_names.get(note_id)
                        if not note_display:
                            if self.verbose:
                                print(f"Note display name not found for {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                            incomplete_mapping = True
                            break
                            
                        color_list.append(f"- {color_name.upper()} (plays {note_display})")
                    
                    if not incomplete_mapping and len(color_list) == len(current_note_color_mapping):
                        colors_text = "\n".join(color_list)
                        if self.verbose:
                            print(f"Successfully got colors from game module: {len(color_list)} colors")
                        return f"Available Color Blocks in this round:\n{colors_text}"
                    else:
                        if self.verbose:
                            print(f"Incomplete mapping from game module, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                
                if self.verbose:
                    print(f"All methods failed, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(0.2)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error getting dynamic color info (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.2)
        
        # 如果所有重试都失败，抛出异常而不是返回默认值
        raise RuntimeError(f"Failed to get real color information after {max_retries} retries. Game environment may not be properly initialized.")

    def _build_detailed_game_state_for_api(self, obs: dict) -> str:
        """构建详细的游戏状态信息，从游戏环境中实时获取完整状态"""
        state = obs["state"]
        
        try:
            # 获取基础游戏状态
            game_state_dict = {
                "current_score": float(state[0]),
                "lives_remaining": float(state[1]),
                "blocks_solved": float(state[2]),
                "current_tick": float(state[3])
            }
            
            # 深入游戏环境获取详细状态信息 - 增加重试机制
            detailed_state = None
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not detailed_state:
                detailed_state, end_game = self._extract_detailed_state_from_environment()
                
                if not detailed_state:
                    retry_count += 1
                    if self.verbose:
                        print(f"Failed to get detailed state, retry {retry_count}/{max_retries}")
                    
                    # 短暂等待后重试
                    if retry_count < max_retries:
                        import time
                        time.sleep(0.1)
                else:
                    break
            
            if detailed_state:
                # 从详细状态中提取关键信息
                last_action = detailed_state.get("last_clicked_action", 0)
                last_color = detailed_state.get("last_clicked_block_color", "")
                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                current_sequence = detailed_state.get("current_correct_sequence", [])
                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                previous_clicks = detailed_state.get("previous_clicks", [])
                sequence_length = detailed_state.get("sequence_length", 5)
                input_length = detailed_state.get("input_length", 0)
                attempts = detailed_state.get("attempts", 0)
                game_over = detailed_state.get("game_over", False)
                
                # 新增：检查数据一致性，如果attempts > 0但previous_clicks为空，则重试获取状态
                if attempts > 0 and not previous_clicks:
                    if self.verbose:
                        print(f"Data inconsistency detected: attempts={attempts} but previous_clicks is empty. Retrying...")
                    
                    # 额外重试机制用于数据一致性问题
                    additional_retries = 10
                    for extra_retry in range(additional_retries):
                        if self.verbose:
                            print(f"Consistency retry {extra_retry + 1}/{additional_retries}")
                        
                        # 短暂等待后重新获取状态
                        import time
                        time.sleep(0.2)
                        
                        retry_state, end_game = self._extract_detailed_state_from_environment()
                        if retry_state:
                            retry_previous_clicks = retry_state.get("previous_clicks", [])
                            retry_attempts = retry_state.get("attempts", 0)
                            
                            # 如果重试后数据一致性得到改善，使用新状态
                            if retry_attempts > 0 and retry_previous_clicks:
                                if self.verbose:
                                    print(f"Consistency improved: attempts={retry_attempts}, previous_clicks={retry_previous_clicks}")
                                detailed_state = retry_state
                                # 重新提取关键信息
                                last_action = detailed_state.get("last_clicked_action", 0)
                                last_color = detailed_state.get("last_clicked_block_color", "")
                                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                                current_sequence = detailed_state.get("current_correct_sequence", [])
                                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                                previous_clicks = retry_previous_clicks
                                sequence_length = detailed_state.get("sequence_length", 5)
                                input_length = detailed_state.get("input_length", 0)
                                attempts = retry_attempts
                                game_over = detailed_state.get("game_over", False)
                                break
                            elif retry_attempts == 0:
                                # 如果重试后attempts变为0，说明游戏可能重置了
                                if self.verbose:
                                    print(f"Game appears to have reset: attempts={retry_attempts}")
                                detailed_state = retry_state
                                # 重新提取关键信息
                                last_action = detailed_state.get("last_clicked_action", 0)
                                last_color = detailed_state.get("last_clicked_block_color", "")
                                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                                current_sequence = detailed_state.get("current_correct_sequence", [])
                                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                                previous_clicks = retry_previous_clicks
                                sequence_length = detailed_state.get("sequence_length", 5)
                                input_length = detailed_state.get("input_length", 0)
                                attempts = retry_attempts
                                game_over = detailed_state.get("game_over", False)
                                break
                    else:
                        if self.verbose:
                            print(f"Consistency retries exhausted, using original state with warning")
                
                # 构建详细的状态描述文本
                state_description = f"""DETAILED GAME STATE (from environment):

LAST ACTION INFO:
- Last Clicked Action ID: {last_action}
- Last Clicked Block Color: {last_color}
- Previous Block Color (last_last): {last_last_color}

SEQUENCE STATUS (CRITICAL FOR DECISION):
- Currently in Correct Sequence: {in_sequence}
- Needs Restart from Beginning: {needs_restart}
- Current Correct Sequence: {current_sequence}
- Previous Clicks History: {previous_clicks}
- Sequence Length: {sequence_length}
- Current Input Length: {input_length}

GAME STATUS:
- Game Over: {game_over}
- Current Tick: {game_state_dict['current_tick']}
- Attempts: {attempts}

STRATEGY HINTS:
- If 'current_correct_sequence' has items: These are the correct colors so far
- If 'previous_clicks' shows history: Learn from past click patterns
- Use sequence position to determine next required musical note"""

                # 保存游戏状态数据用于调试和分析
                try:
                    game_state_data = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": self.current_episode,
                        "step": self.current_step,
                        "last_clicked_color": last_color,
                        "currently_in_correct_sequence": in_sequence,
                        "current_correct_sequence": current_sequence,
                        "last_last_clicked_color": last_last_color,
                        "previous_clicks": previous_clicks,
                        "sequence_length": sequence_length,
                        "input_length": input_length,
                        "needs_restart": needs_restart,
                        "game_score": game_state_dict['current_score'],
                        "lives_remaining": game_state_dict['lives_remaining'],
                        "game_over": game_over,
                        "attempts": attempts,
                        "retry_count": retry_count,
                        "data_consistency_check": {
                            "attempts_gt_zero": attempts > 0,
                            "previous_clicks_empty": len(previous_clicks) == 0,
                            "inconsistency_detected": attempts > 0 and len(previous_clicks) == 0
                        }
                    }
                    
                    game_data_dir = "game_data/caclu"
                    os.makedirs(game_data_dir, exist_ok=True)
                    
                    # 使用时间戳和步骤信息命名文件
                    filename = f"game_state_ep{self.current_episode}_step{self.current_step}.json"
                    
                    # 创建完整的文件路径
                    full_filepath = os.path.join(game_data_dir, filename)
                    
                    # 直接保存到指定路径，不使用save函数的自动路径处理
                    with open(full_filepath, 'w', encoding='utf-8') as f:
                        json.dump(game_state_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    if self.verbose:
                        print(f"Game state saved to: {full_filepath}")
                        
                except Exception as save_error:
                    if self.verbose:
                        print(f"Warning: Failed to save game state data: {save_error}")
                
                return state_description, detailed_state, end_game
            else:
                # 所有重试都失败，使用基础状态信息
                if self.verbose:
                    print(f"Failed to get detailed state after {max_retries} retries, using basic state")
                
                # 构建基础状态描述
                fallback_description = f"""BASIC GAME STATE (detailed state unavailable after {max_retries} retries):

BASIC INFO:
- Score: {game_state_dict['current_score']}
- Lives: {game_state_dict['lives_remaining']}
- Blocks Solved: {game_state_dict['blocks_solved']}
- Game Tick: {game_state_dict['current_tick']}

SEQUENCE STATUS (LIMITED INFO):
- Episode: {self.current_episode}
- Step: {self.current_step}
- Last Action Taken: {list(COLOR_ID_MAP.keys())[self.current_step % len(COLOR_ID_MAP)] if self.current_step > 0 else "None"}

FALLBACK STRATEGY:
- Use audio feedback to learn color-note mappings
- Try systematic exploration of available colors
- Listen for musical sequence patterns in audio feedback
- Use visual feedback (green/red highlights) to confirm correct/incorrect choices

NOTE: Detailed game state from environment is temporarily unavailable. 
Making decisions based on basic state information and audio/visual feedback."""
                
                # 保存失败的状态信息用于调试
                try:
                    fallback_data = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": self.current_episode,
                        "step": self.current_step,
                        "error": "Failed to get detailed state",
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "basic_state": game_state_dict,
                        "has_game_environment": self.game_environment is not None
                    }
                    
                    game_data_dir = "game_data/caclu"
                    os.makedirs(game_data_dir, exist_ok=True)
                    filename = f"fallback_state_ep{self.current_episode}_step{self.current_step}.json"
                    full_filepath = os.path.join(game_data_dir, filename)
                    with open(full_filepath, 'w', encoding='utf-8') as f:
                        json.dump(fallback_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    if self.verbose:
                        print(f"Fallback state saved to: {full_filepath}")
                        
                except Exception as save_error:
                    if self.verbose:
                        print(f"Warning: Failed to save fallback state data: {save_error}")
                
                return fallback_description
            
        except Exception as e:
            if self.verbose:
                print(f"Error building detailed game state: {e}")
            
            # 最终错误处理：返回最基本的状态信息
            error_description = f"""EMERGENCY FALLBACK STATE (Error occurred):

ERROR: {str(e)}

MINIMAL GAME INFO:
- Score: {float(state[0])}
- Lives: {float(state[1])}
- Blocks Solved: {float(state[2])}
- Game Tick: {float(state[3])}
- Episode: {self.current_episode}
- Step: {self.current_step}

EMERGENCY STRATEGY:
- Use random exploration if no other information available
- Try to listen to audio feedback from previous actions
- Look for visual patterns in the game image
- Make conservative choices to preserve lives"""
            
            return error_description

    def _extract_detailed_state_from_environment(self) -> Dict[str, Any]:
        """从游戏环境中提取详细状态信息 - 使用环境的专门方法"""
        if not self.game_environment:
            if self.verbose:
                print("No game environment connected")
            return {}
        
        try:
            # 使用环境的专门方法获取详细状态
            if hasattr(self.game_environment, 'get_detailed_game_state_for_agent'):
                try:
                    detailed_state, end_game = self.game_environment.get_detailed_game_state_for_agent()
                    if detailed_state and isinstance(detailed_state, dict):
                        
                        # 检查数据一致性：如果attempts > 0但previous_clicks为空，则重试
                        attempts = detailed_state.get("attempts", 0)
                        previous_clicks = detailed_state.get("previous_clicks", [])
                        
                        if attempts > 0 and not previous_clicks:
                            if self.verbose:
                                print(f"Data consistency issue detected in environment state: attempts={attempts}, previous_clicks={previous_clicks}")
                                print("Starting internal retry mechanism...")
                            
                            # 内部重试机制
                            max_internal_retries = 15
                            retry_delay = 0.1
                            
                            for retry_attempt in range(max_internal_retries):
                                if self.verbose:
                                    print(f"Internal retry {retry_attempt + 1}/{max_internal_retries}")
                                
                                # 短暂等待
                                import time
                                time.sleep(retry_delay)
                                
                                # 重新获取状态
                                retry_state = self.game_environment.get_detailed_game_state_for_agent()
                                
                                if retry_state and isinstance(retry_state, dict):
                                    retry_attempts = retry_state.get("attempts", 0)
                                    retry_previous_clicks = retry_state.get("previous_clicks", [])
                                    
                                    # 检查数据一致性是否改善
                                    if retry_attempts > 0 and retry_previous_clicks:
                                        if self.verbose:
                                            print(f"Data consistency restored: attempts={retry_attempts}, previous_clicks={retry_previous_clicks}")
                                        detailed_state = retry_state
                                        break
                                    elif retry_attempts == 0:
                                        # 如果重试后attempts变为0，说明游戏状态重置了
                                        if self.verbose:
                                            print(f"Game state appears to have reset: attempts={retry_attempts}")
                                        detailed_state = retry_state
                                        break
                                    else:
                                        # 数据仍然不一致，继续重试
                                        if self.verbose:
                                            print(f"Consistency still poor: attempts={retry_attempts}, previous_clicks={retry_previous_clicks}")
                                        
                                        # 逐渐增加重试延迟
                                        retry_delay = min(retry_delay * 1.2, 0.5)
                                else:
                                    if self.verbose:
                                        print(f"Failed to get retry state on attempt {retry_attempt + 1}")
                            else:
                                # 所有重试都失败了，使用原始状态并发出警告
                                if self.verbose:
                                    print(f"Internal retries exhausted. Using original inconsistent state with warning.")
                                # 在返回的状态中添加警告标记
                                detailed_state["data_consistency_warning"] = {
                                    "issue": "attempts > 0 but previous_clicks empty",
                                    "original_attempts": attempts,
                                    "original_previous_clicks": previous_clicks,
                                    "retries_attempted": max_internal_retries,
                                    "resolution": "unresolved"
                                }
                        
                        # 常规verbose输出
                        if self.verbose:
                            current_attempts = detailed_state.get("attempts", 0)
                            current_previous_clicks = detailed_state.get("previous_clicks", [])
                            
                            print(f"Successfully got detailed state from environment:")
                            print(f"  Current state: {detailed_state.get('current_state', 'unknown')}")
                            print(f"  Score: {detailed_state.get('current_score', 0)}")
                            print(f"  Sequence: {detailed_state.get('input_length', 0)}/{detailed_state.get('sequence_length', 0)}")
                            print(f"  In correct sequence: {detailed_state.get('currently_in_correct_sequence', False)}")
                            print(f"  Needs restart: {detailed_state.get('needs_restart_from_beginning', False)}")
                            print(f"  Available colors: {len(detailed_state.get('available_colors', []))}")
                            print(f"  Attempts: {current_attempts}")
                            print(f"  Previous clicks: {current_previous_clicks}")
                            
                            # 检查数据一致性状态
                            if current_attempts > 0 and not current_previous_clicks:
                                print("  WARNING: Data consistency issue still present!")
                            elif current_attempts > 0 and current_previous_clicks:
                                print("  INFO: Data consistency verified")
                            
                            # 检查是否使用了备选方案
                            debug_info = detailed_state.get('debug_info', {})
                            if debug_info.get('fallback_used', False):
                                print("  Warning: Using fallback state")
                            elif debug_info.get('game_module_available', False):
                                print("  Success: Got real-time game state")
                            
                            # 检查是否有一致性警告
                            if "data_consistency_warning" in detailed_state:
                                print("  WARNING: Data consistency could not be resolved")
                        
                        return detailed_state, end_game
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error using environment's get_detailed_game_state_for_agent: {e}")
            
            else:
                print("meiyou")
        except Exception as e:
            if self.verbose:
                print(f"Error in _extract_detailed_state_from_environment: {e}")
                import traceback
                traceback.print_exc()
            return {}
    
    def _check_sequence_match(self, player_input: List, correct_sequence: List) -> bool:
        """检查玩家输入是否与正确序列匹配"""
        if not player_input:
            return True
        if not correct_sequence:
            return False
        
        for i in range(len(player_input)):
            if i >= len(correct_sequence) or player_input[i] != correct_sequence[i]:
                return False
        return True

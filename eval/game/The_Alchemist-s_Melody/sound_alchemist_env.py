
"""
把 sound_alchemist_game.py 封装成 Gymnasium Env。

核心特性
---------
1. reset(difficulty) 直接跳到指定难度并开始游戏；
2. step(action)：
   • 注入一次点击动作（离散：颜色 id）
   • 推进游戏内部 1 秒 (60 帧) 并同步逻辑
   • 抓屏得到 final RGB 帧 (224×224)
   • 录制 1 秒系统声道 / 游戏声道 (16 kHz mono)
   • 返回 obs = {"image", "audio", "state"}
"""
from __future__ import annotations
import os
import time
import json
import traceback
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not available. Audio capture disabled.")

try:
    import soundfile as sf
except ImportError:
    sf = None
    print("Warning: soundfile not available. Audio saving disabled.")

try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not available. Audio resampling disabled.")

from typing import List, Dict, Any

# === 简化的游戏包装器 =====
class Game:
    """简化的游戏包装器，适配原始游戏逻辑"""
    def __init__(self, screen):
        self.screen = screen
        self.score = 0
        self.lives = 3
        self.solved_blocks = 0
        self.game_over = False
        
        # 音频捕获相关
        self.last_played_sound = None
        self.sound_played_this_step = False
        
        # 添加verbose属性
        self._verbose = False
        
        # 导入并初始化原始游戏的全局变量
        try:
            from . import sound_alchemist_game as game_module
            self.game_module = game_module
            # 设置游戏的screen
            game_module.screen = screen
            # 启用自动开始模式
            game_module.set_auto_start_mode(True)
        except ImportError:
            print("Warning: Could not import sound_alchemist_game module")
            self.game_module = None

    @property
    def verbose(self):
        """获取verbose属性"""
        return getattr(self, '_verbose', False)

    @verbose.setter
    def verbose(self, value):
        """设置verbose属性"""
        self._verbose = value

    def reset_audio_state(self):
        """重置音频状态"""
        self.last_played_sound = None
        self.sound_played_this_step = False

    def get_state_info(self):
        """获取详细的游戏状态信息"""
        if self.game_module:
            try:
                return self.game_module.get_game_state()
            except Exception as e:
                print(f"Error getting game state from module: {e}")
                # 返回默认状态
                return {
                    "state": "unknown",
                    "difficulty": "normal",
                    "score": self.score,
                    "attempts": 0,
                    "sequence_length": 0,
                    "input_length": 0,
                    "game_over": self.game_over
                }
        return {
            "state": "unknown",
            "difficulty": "normal",
            "score": self.score,
            "attempts": 0,
            "sequence_length": 0,
            "input_length": 0,
            "game_over": self.game_over
        }

    def reset(self, difficulty="normal"):
        """重置游戏状态并自动开始游戏"""
        self.score = 0
        self.lives = 3
        self.solved_blocks = 0
        self.game_over = False
        
        if self.game_module:
            # 重置原始游戏的全局状态
            self.game_module.current_state = self.game_module.MENU
            self.game_module.player_score = 0
            self.game_module.melody_puzzle_attempts = 0
            self.game_module.player_melody_input = []
            
            # 根据难度设置游戏
            difficulty_map = {
                "easy": self.game_module.DIFFICULTY_EASY,
                "normal": self.game_module.DIFFICULTY_MEDIUM,
                "hard": self.game_module.DIFFICULTY_HARD
            }
            target_difficulty = difficulty_map.get(difficulty, self.game_module.DIFFICULTY_MEDIUM)
            
            # 直接开始旋律谜题
            self.game_module.start_melody_puzzle_directly(target_difficulty)
            
            print(f"Game reset with difficulty: {difficulty}, state: {self.game_module.current_state}")
    
    def update(self):
        """更新游戏逻辑一帧"""
        if self.game_module:
            try:
                # 处理pygame事件，防止窗口无响应
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                
                # 更新游戏状态
                self.score = self.game_module.player_score
                
                # 手动调用游戏渲染逻辑
                self._render_game_frame()
                
                # 获取游戏状态信息
                game_state = self.game_module.get_game_state()
                
                # 更新游戏结束条件
                self.game_over = game_state["game_over"]
                
                # 如果游戏完成，自动重置到新回合
                if self.game_over and self.game_module.current_state == self.game_module.PUZZLE_COMPLETE:
                    print("Puzzle completed! Auto-resetting for next round...")
                    # 延迟一小段时间让玩家看到完成状态
                    time.sleep(0.1)
                    # 重置到新的回合，保持当前难度
                    current_diff_name = {
                        self.game_module.DIFFICULTY_EASY: "easy",
                        self.game_module.DIFFICULTY_MEDIUM: "normal", 
                        self.game_module.DIFFICULTY_HARD: "hard"
                    }.get(self.game_module.current_difficulty, "normal")
                    self.reset(current_diff_name)
            except Exception as e:
                print(f"Error in game update: {e}")
                # 重置游戏状态以防止崩溃
                self.game_over = False

    def _render_game_frame(self):
        """手动渲染游戏帧"""
        if not self.game_module:
            return
            
        try:
            # 获取当前游戏状态
            current_state = self.game_module.current_state
            
            # 清空屏幕
            self.screen.fill((0, 0, 0))
            
            if current_state == self.game_module.PUZZLE_MELODY:
                # 渲染旋律谜题界面
                self._render_melody_puzzle()
            elif current_state == self.game_module.MENU:
                # 渲染菜单界面
                self._render_menu()
            elif current_state == self.game_module.PUZZLE_COMPLETE:
                # 渲染完成界面
                self._render_puzzle_complete()
            else:
                # 默认渲染
                self._render_default()
            
            # 更新显示
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in render game frame: {e}")
            # 至少填充一个非黑色背景，表明渲染在工作
            self.screen.fill((50, 50, 100))
            pygame.display.flip()

    def _render_melody_puzzle(self):
        """渲染旋律谜题界面"""
        # 背景
        self.screen.fill((30, 30, 70))
        
        # 获取字体
        try:
            font_medium = pygame.font.Font(None, 50)
            font_small = pygame.font.Font(None, 30)
            font_tiny = pygame.font.Font(None, 24)
        except:
            font_medium = pygame.font.Font(None, 50)
            font_small = pygame.font.Font(None, 30)
            font_tiny = pygame.font.Font(None, 24)
        
        # 标题
        title_text = font_medium.render("The Alchemist's Melody", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.screen.get_width() // 2, 50))
        self.screen.blit(title_text, title_rect)
        
        # 难度显示
        difficulty_name = self.game_module.DIFFICULTY_SETTINGS[self.game_module.current_difficulty]['name']
        difficulty_text = font_small.render(f"Difficulty: {difficulty_name}", True, (173, 216, 230))
        difficulty_rect = difficulty_text.get_rect(center=(self.screen.get_width() // 2, 90))
        self.screen.blit(difficulty_text, difficulty_rect)
        
        # 指示文本
        instruction_text = font_small.render("Click the colored blocks in the correct musical order", True, (255, 255, 0))
        instruction_rect = instruction_text.get_rect(center=(self.screen.get_width() // 2, 130))
        self.screen.blit(instruction_text, instruction_rect)
        
        # 绘制音符元素
        if hasattr(self.game_module, 'note_elements'):
            self.game_module.note_elements.draw(self.screen)
            # 更新音符元素动画
            self.game_module.note_elements.update()
        
        # 显示玩家输入
        input_display = []
        for note_id in self.game_module.player_melody_input:
            note_name = self.game_module.NOTE_DISPLAY_NAMES.get(note_id, "?")
            input_display.append(note_name)
        
        input_text = font_small.render(f"Your input: {' - '.join(input_display)}", True, (255, 255, 255))
        input_rect = input_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 80))
        self.screen.blit(input_text, input_rect)
        
        # 显示错误次数
        attempts_text = font_tiny.render(f"Mistakes: {self.game_module.melody_puzzle_attempts}", True, (255, 170, 170))
        self.screen.blit(attempts_text, (20, self.screen.get_height() - 30))
        
        # 绘制粒子效果
        if hasattr(self.game_module, 'particles_group'):
            self.game_module.particles_group.update()
            self.game_module.particles_group.draw(self.screen)

    def _render_menu(self):
        """渲染菜单界面"""
        # 使用渐变背景
        for i in range(self.screen.get_height()):
            color = (20, 20, max(40, min(40 + i // 3, 90)))
            pygame.draw.line(self.screen, color, (0, i), (self.screen.get_width(), i))
        
        # 获取字体
        try:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        except:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        
        # 标题
        title_text = font_large.render("Sound Alchemist's Chamber", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.screen.get_width() // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # 自动模式提示
        auto_text = font_medium.render("Auto Mode - Game Starting...", True, (255, 255, 0))
        auto_rect = auto_text.get_rect(center=(self.screen.get_width() // 2, 300))
        self.screen.blit(auto_text, auto_rect)

    def _render_puzzle_complete(self):
        """渲染谜题完成界面"""
        self.screen.fill((20, 80, 20))
        
        try:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        except:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        
        # 完成消息
        complete_text = font_large.render("Melody Puzzle Solved!", True, (255, 255, 255))
        complete_rect = complete_text.get_rect(center=(self.screen.get_width() // 2, 200))
        self.screen.blit(complete_text, complete_rect)
        
        # 分数显示
        score_text = font_medium.render(f"Score: {self.game_module.player_score}", True, (255, 255, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width() // 2, 300))
        self.screen.blit(score_text, score_rect)
        
        # 庆祝文本
        congrats_text = font_medium.render("Well Done, Alchemist!", True, (255, 255, 0))
        congrats_rect = congrats_text.get_rect(center=(self.screen.get_width() // 2, 400))
        self.screen.blit(congrats_text, congrats_rect)

    def _render_default(self):
        """默认渲染"""
        self.screen.fill((50, 50, 100))
        try:
            font = pygame.font.Font(None, 36)
            text = font.render("Game Loading...", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(text, text_rect)
        except:
            pass

    def get_last_played_audio_data(self):
        """获取最近播放的音频数据"""
        if self.game_module and hasattr(self.game_module, 'get_last_played_audio_data'):
            try:
                return self.game_module.get_last_played_audio_data()
            except Exception as e:
                print(f"Error getting audio data from game module: {e}")
                return None
        return None

    def click(self, color_name):
        """处理颜色点击并记录播放的声音"""
        if not self.game_module:
            return
            
        try:
            # 动态获取当前的颜色到音符映射
            color_to_note = self._get_dynamic_color_to_note_mapping()
            
            note_id = color_to_note.get(color_name)
            if note_id:
                print(f"Color {color_name} mapped to note {note_id} ({self.game_module.NOTE_DISPLAY_NAMES.get(note_id, note_id)})")
                
                # 记录即将播放的声音
                self.last_played_sound = note_id
                self.sound_played_this_step = True
                
                # 找到对应的音符元素并触发交互
                found_element = False
                for note_sprite in self.game_module.note_elements:
                    if note_sprite.element_id == note_id:
                        note_sprite.interact()
                        found_element = True
                        break
                
                if not found_element:
                    print(f"Warning: No visual element found for note {note_id}")
                    # 创建一个临时的元素对象来模拟交互
                    class TempElement:
                        def __init__(self, element_id):
                            self.element_id = element_id
                            self.sound = None
                            # 添加模拟的rect属性
                            self.rect = type('Rect', (), {'center': (400, 300)})()
                        
                        def highlight(self, color_tuple, duration=20):
                            """模拟highlight方法，实际不执行任何操作"""
                            print(f"TempElement highlight called with color {color_tuple}, duration {duration}")
                            pass
                    
                    # 获取对应的音效并播放
                    if hasattr(self.game_module, 'melody_note_sounds') and note_id in self.game_module.melody_note_sounds:
                        # 直接调用play_sound函数来确保记录文件路径
                        sound_obj = self.game_module.melody_note_sounds[note_id]
                        self.game_module.play_sound(sound_obj)
                    
                    # 创建临时元素并调用游戏逻辑
                    temp_element = TempElement(note_id)
                    temp_element.sound = self.game_module.melody_note_sounds.get(note_id) if hasattr(self.game_module, 'melody_note_sounds') else None
                    
                    # 直接调用游戏逻辑处理输入
                    self.game_module.handle_melody_input(note_id, temp_element)
            else:
                print(f"Warning: Color {color_name} not mapped to any note in current game configuration")
                
        except Exception as e:
            print(f"Error in click handling: {e}")
            traceback.print_exc()

    def _get_dynamic_color_to_note_mapping(self):
        """动态获取当前游戏中的颜色到音符映射"""
        color_to_note = {}
        
        try:
            if not self.game_module:
                return {}
            
            # 获取当前的音符到颜色映射
            current_note_color_mapping = getattr(self.game_module, 'current_note_color_mapping', {})
            all_colors = getattr(self.game_module, 'ALL_COLORS', {})
            
            if current_note_color_mapping and all_colors:
                # 创建RGB颜色到颜色名称的映射
                color_rgb_to_name = {v: k for k, v in all_colors.items()}
                
                # 从音符->颜色映射 转换为 颜色名称->音符映射
                for note_id, rgb_color in current_note_color_mapping.items():
                    color_name = color_rgb_to_name.get(rgb_color)
                    if color_name:
                        color_to_note[color_name.upper()] = note_id
                
                if self.verbose and color_to_note:
                    print(f"Dynamic color-to-note mapping: {color_to_note}")
                
                return color_to_note
            
            # 如果无法获取动态映射，记录警告
            if self.verbose:
                print("Warning: Could not get dynamic color-to-note mapping from game module")
                
        except Exception as e:
            if self.verbose:
                print(f"Error getting dynamic color-to-note mapping: {e}")
        
        # 返回空映射，让调用方处理
        return {}

    def get_block_pos(self, color_name):
        """获取指定颜色方块的位置"""
        if not self.game_module:
            return (400, 300)  # 默认中心位置
            
        try:
            # 使用动态映射获取音符ID
            color_to_note = self._get_dynamic_color_to_note_mapping()
            note_id = color_to_note.get(color_name)
            
            if note_id:
                for note_sprite in self.game_module.note_elements:
                    if note_sprite.element_id == note_id:
                        return note_sprite.rect.center
        except Exception as e:
            print(f"Error getting block position: {e}")
        
        return (400, 300)  # 默认位置

    def get_available_colors(self):
        """获取当前回合可用的颜色块信息"""
        if not self.game_module:
            # 强制等待游戏模块初始化
            max_retries = 10
            retry_count = 0
            while not self.game_module and retry_count < max_retries:
                print(f"Waiting for game module initialization... (retry {retry_count + 1}/{max_retries})")
                time.sleep(0.1)
                retry_count += 1
                # 重新尝试获取游戏模块
                if hasattr(self.game, 'game_module'):
                    self.game_module = self.game.game_module
            
            if not self.game_module:
                raise RuntimeError("Game module not available after maximum retries")
        
        return self._get_available_colors_from_game_module(self.game_module)

    def _get_available_colors_from_game_module(self, game_module) -> List[Dict[str, Any]]:
        """从游戏模块获取可用颜色列表"""
        retry_count = 0
        max_retries = 50  # 增加重试次数
        
        while retry_count < max_retries:
            try:
                # 从当前音符序列和颜色映射获取
                correct_sequence = getattr(game_module, 'correct_melody_sequence', [])
                current_mapping = getattr(game_module, 'current_note_color_mapping', {})
                
                if not correct_sequence:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"correct_melody_sequence is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                if not current_mapping:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                # 获取颜色名称映射
                all_colors = getattr(game_module, 'ALL_COLORS', {})
                note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                
                if not all_colors:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"ALL_COLORS is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                if not note_display_names:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"NOTE_DISPLAY_NAMES is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                color_name_mapping = {v: k for k, v in all_colors.items()}
                available_colors = []
                
                # 验证所有必要数据的完整性
                incomplete_data = False
                for note_id in correct_sequence:
                    if note_id not in current_mapping:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note {note_id} not in current_mapping, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    rgb_color = current_mapping[note_id]
                    color_name = color_name_mapping.get(rgb_color)
                    if not color_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    note_name = note_display_names.get(note_id)
                    if not note_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note display name not found for {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    available_colors.append({
                        "note_id": note_id,
                        "color_name": color_name,
                        "note_name": note_name,
                        "color_rgb": rgb_color
                    })
                
                if incomplete_data:
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if len(available_colors) == len(correct_sequence):
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Successfully got {len(available_colors)} colors from game module sequence")
                        for color_info in available_colors:
                            print(f"  - {color_info['color_name']}: {color_info['note_name']}")
                    return available_colors
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Incomplete color list ({len(available_colors)}/{len(correct_sequence)}), retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error getting available colors from game module (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to get real available colors from game module after {max_retries} retries")

    def _get_color_note_mapping_from_game_module(self, game_module) -> Dict[str, str]:
        """从游戏模块获取颜色到音符的映射"""
        retry_count = 0
        max_retries = 50  # 增加重试次数
        
        while retry_count < max_retries:
            try:
                # 获取当前映射和显示名称
                current_mapping = getattr(game_module, 'current_note_color_mapping', {})
                note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                
                if not current_mapping:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if not note_display_names:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"NOTE_DISPLAY_NAMES is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                all_colors = getattr(game_module, 'ALL_COLORS', {})
                if not all_colors:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"ALL_COLORS is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                color_name_mapping = {v: k for k, v in all_colors.items()}
                color_note_mapping = {}
                
                # 验证所有数据的完整性
                incomplete_data = False
                for note_id, rgb_color in current_mapping.items():
                    color_name = color_name_mapping.get(rgb_color)
                    if not color_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                        
                    note_name = note_display_names.get(note_id)
                    if not note_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note display name not found for {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                        
                    color_note_mapping[color_name.lower()] = note_name
                
                if incomplete_data:
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if len(color_note_mapping) == len(current_mapping):
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Built color-note mapping from game module: {color_note_mapping}")
                    return color_note_mapping
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Incomplete color-note mapping ({len(color_note_mapping)}/{len(current_mapping)}), retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error getting color-note mapping from game module (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to get real color-note mapping from game module after {max_retries} retries")

    def _convert_note_id_to_color_name(self, note_id, game_module) -> str:
        """将音符ID转换为颜色名称"""
        retry_count = 0
        max_retries = 30  # 增加重试次数
        
        while retry_count < max_retries:
            try:
                # 获取当前的音符到颜色映射
                if hasattr(game_module, 'current_note_color_mapping'):
                    note_color_mapping = game_module.current_note_color_mapping
                    if not note_color_mapping:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.1)
                        continue
                        
                    if note_id in note_color_mapping:
                        rgb_color = note_color_mapping[note_id]
                        
                        # 必须从ALL_COLORS中找到对应的颜色名称
                        if not hasattr(game_module, 'ALL_COLORS') or not game_module.ALL_COLORS:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"ALL_COLORS not available for note {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.1)
                            continue
                            
                        all_colors = game_module.ALL_COLORS
                        color_name_mapping = {v: k for k, v in all_colors.items()}
                        color_name = color_name_mapping.get(rgb_color)
                        if color_name:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"Successfully converted note {note_id} to color {color_name}")
                            return color_name.lower()
                        else:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.1)
                            continue
                    else:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note {note_id} not in mapping, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.1)
                        continue
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping attribute not found, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error converting note ID {note_id} to color name (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to convert note ID {note_id} to color name after {max_retries} retries")

# ---- 常量 ----
FPS = 60                 # 游戏逻辑帧率
SEC_PER_STEP = 1         # Env 一次 step 推进 1 秒
AUDIO_SR = 16_000        # 采样率 16 kHz
AUDIO_CHANNELS = 1       # 单声道
IMG_SIZE = (224, 224)    # 输出观测用分辨率
# 扩展颜色映射以支持所有音符
COLOR_ID_MAP = {
    "BLUE": 0,     # Sol
    "RED": 1,      # Do
    "GREEN": 2,    # Fa  
    "YELLOW": 3,   # Mi
    "ORANGE": 4,   # Re
    "PURPLE": 5,   # La
    "GREY": 6,     # Ti/Si
}

class SoundAlchemistEnv(gym.Env):
    """Gymnasium-style 环境"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        difficulty: str = "normal",
        render_mode: str | None = None,
        capture_audio: bool = True,
        audio_device: str | int | None = None,
        save_data: bool = False,
        save_dir: str = "game_data/caclu",
        save_sequence: bool = True
    ):
        super().__init__()
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.capture_audio = capture_audio
        self.audio_device = audio_device
        self.save_data = save_data
        self.save_dir = save_dir
        self.save_sequence = save_sequence
        
        # 数据保存相关
        self.episode_count = 0
        self.step_count_total = 0
        
        # 序列数据保存相关
        self.current_episode_sequence: List[Dict[str, Any]] = []
        self.sequence_save_dir = os.path.join(save_dir, "sequences")
        
        self.current_correct_sequence = []
        
        # 创建保存目录
        if self.save_data:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "audio"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "metadata"), exist_ok=True)
            print(f"Data will be saved to: {self.save_dir}")
        
        if self.save_sequence:
            os.makedirs(self.sequence_save_dir, exist_ok=True)
            print(f"Sequence data will be saved to: {self.sequence_save_dir}")

        # 动作：点击哪种颜色
        self.action_space = spaces.Discrete(len(COLOR_ID_MAP))

        # 观测：Dict(image, audio, state)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(*IMG_SIZE, 3), dtype=np.uint8
                ),
                "audio": spaces.Box(
                    low=-1.0, high=1.0, shape=(AUDIO_SR * SEC_PER_STEP,), dtype=np.float32
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
            }
        )

        # ---- 初始化 PyGame 与原始游戏 ----
        pygame.init()
        pygame.mixer.init()
        # 解决 headless 时无法打开窗口：HIDDEN
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Sound-Alchemist Gym Env")
        self.clock = pygame.time.Clock()
        self.game = Game(screen=self.screen)
        
        # 添加对游戏模块的直接引用
        self.game_module = self.game.game_module if hasattr(self.game, 'game_module') else None
        
        self.tick = 0
        self._last_score = 0

        # 新增：多模态评分系统
        self.episode_performance_log: List[Dict[str, Any]] = []
        self.current_episode_metrics = {
            "positive_rewards": 0,
            "negative_rewards": 0,
            "audio_events": 0,
            "visual_changes": 0,
            "sequence_progress": 0,
            "sequence_resets": 0,
            "decision_quality": [],
            "response_times": [],
            "correct_actions": 0,
            "total_actions": 0
        }
        
        # 评分保存目录
        self.scores_save_dir = os.path.join(save_dir, "scores")
        if self.save_data:
            os.makedirs(self.scores_save_dir, exist_ok=True)
            print(f"Episode scores will be saved to: {self.scores_save_dir}")

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if options and "difficulty" in options:
            self.difficulty = options["difficulty"]

        # 保存上一episode的序列数据
        if self.save_sequence and self.current_episode_sequence:
            self._save_episode_sequence()

        # 重置序列数据
        self.current_episode_sequence = []

        # 1) 重置游戏
        self.game.reset(self.difficulty)

        self.tick = 0
        self._last_score = 0
        self.episode_count += 1

        # 强制多次更新以确保游戏正确初始化和渲染
        for _ in range(10):
            self.game.update()
            self.clock.tick(FPS)

        # 2) 抓第一帧
        frame = self._grab_frame()
        audio = np.zeros(AUDIO_SR * SEC_PER_STEP, dtype=np.float32)  # 首帧无音
        state_vec = self._get_state_vec()

        # 如果帧仍然全黑，尝试手动渲染
        if np.all(frame == 0):
            print("Warning: Initial frame is black, attempting manual render...")
            self.game._render_game_frame()
            frame = self._grab_frame()

        # 3) 保存数据（如果启用）
        if self.save_data:
            self._save_step_data(frame, audio, state_vec, action=-1, reward=0, is_reset=True)


        observation = {"image": frame, "audio": audio, "state": state_vec}
        info = {"tick": self.tick, "episode": self.episode_count}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action id"

        # 重置音频状态
        self.game.reset_audio_state()
        
        # 记录步骤前的状态
        prev_score = self.game.score
        prev_attempts = getattr(self.game.game_module, 'melody_puzzle_attempts', 0) if self.game.game_module else 0
        prev_input_length = len(getattr(self.game.game_module, 'player_melody_input', [])) if self.game.game_module else 0

        # 1）把离散动作转颜色字符串
        color_name = list(COLOR_ID_MAP.keys())[action]
        self._inject_click(color_name)

        # 3）推进游戏 SEC_PER_STEP 秒
        n_frames = int(FPS * SEC_PER_STEP)
        for _ in range(n_frames):
            self.game.update()
            self.clock.tick(FPS)

        self.tick += 1
        self.step_count_total += 1
        
        # 2）获取游戏内部音频数据
        audio_block = self._get_game_audio(duration=SEC_PER_STEP)

        # 4）构造返回
        frame = self._grab_frame()
        state_vec = self._get_state_vec()

        # 更智能的奖励计算，考虑学习过程和界面反馈
        cur_score = self.game.score
        cur_attempts = getattr(self.game.game_module, 'melody_puzzle_attempts', 0) if self.game.game_module else 0
        cur_input_length = len(getattr(self.game.game_module, 'player_melody_input', [])) if self.game.game_module else 0
        
        # 基础奖励：分数增量
        reward = cur_score - self._last_score
        
        # 进度奖励：成功添加到序列（输入长度增加）
        if cur_input_length > prev_input_length:
            reward += 0.5  # 正向进度奖励
        
        # 重置惩罚：序列被重置（输入长度减少到0）
        elif prev_input_length > 0 and cur_input_length == 0 and cur_attempts > prev_attempts:
            reward -= 1.0  # 序列重置的较大惩罚
        
        # 探索奖励：成功点击（播放了声音但没有错误）
        elif cur_attempts == prev_attempts and self.game.sound_played_this_step:
            reward += 0.1  # 小的正向奖励鼓励探索
            
        self._last_score = cur_score

        terminated = self.game.game_over
        truncated = False
        
        # 增强info信息，包含界面反馈信息
        info = {
            "tick": self.tick, 
            "episode": self.episode_count,
            "color_clicked": color_name,
            "sound_played": self.game.sound_played_this_step,
            "note_played": self.game.last_played_sound,
            "attempts_increased": cur_attempts > prev_attempts,
            "score_increased": cur_score > prev_score,
            "sequence_progress": cur_input_length,
            "sequence_reset": prev_input_length > 0 and cur_input_length == 0,
            "sequence_advanced": cur_input_length > prev_input_length,
            "total_sequence_length": len(getattr(self.game.game_module, 'correct_melody_sequence', [])) if self.game.game_module else 0
        }

        # 5) 保存数据（如果启用）
        if self.save_data:
            self._save_step_data(frame, audio_block, state_vec, action, reward)


        observation = {"image": frame, "audio": audio_block, "state": state_vec}
        
        # 6) 记录step到序列
        if self.save_sequence:
            self._add_to_sequence(
                step_type="step",
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info
            )

        if self.render_mode == "rgb_array":
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self):
        # 保存最后一个episode的序列数据
        if self.save_sequence and self.current_episode_sequence:
            self._save_episode_sequence()
        pygame.quit()

    # ---------- 内部工具 ----------
    def _grab_frame(self) -> np.ndarray:
        """抓取屏幕 → 224×224 RGB"""
        raw = pygame.surfarray.array3d(self.screen)  # (W,H,3)
        raw = np.transpose(raw, (1, 0, 2))           # (H,W,3)
        # resize → 224×224
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(raw), IMG_SIZE)
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        return arr

    def _get_game_audio(self, duration: float) -> np.ndarray:
        """获取游戏内部音频数据"""
        # 尝试获取游戏内部播放的音频
        if hasattr(self.game, 'get_last_played_audio_data'):
            game_audio = self.game.get_last_played_audio_data()
            if game_audio is not None:
                return game_audio
        
        # 如果没有游戏音频，使用原来的录制逻辑或生成静音
        if self.capture_audio and sd:
            try:
                # 尝试录制真实音频
                audio = sd.rec(int(duration * AUDIO_SR), samplerate=AUDIO_SR,
                              channels=AUDIO_CHANNELS, dtype="float32",
                              device=self.audio_device)
                sd.wait()
                return audio.flatten()
            except Exception as e:
                print(f"Audio recording failed: {e}, using silence")
        
        # 生成静音数据
        print("Using silence for audio data")
        return np.zeros(int(duration * AUDIO_SR), dtype=np.float32)

    def _record_audio(self, duration: float) -> np.ndarray:
        """保持向后兼容的音频录制方法"""
        return self._get_game_audio(duration)

    def _inject_click(self, color_name: str):
        """向 PyGame 注入一次点击事件；根据颜色找到对应坐标"""
        print(f"Injecting click for color: {color_name}")
        
        # 如果游戏有 API 定点点击，可以直接调用
        if hasattr(self.game, "click"):
            self.game.click(color_name)
            return

        # 回退方案：找到该颜色方块中心坐标并注入鼠标事件
        pos = self.game.get_block_pos(color_name)
        print(f"Click position for {color_name}: {pos}")
        
        ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1})
        ev_up   = pygame.event.Event(pygame.MOUSEBUTTONUP,   {"pos": pos, "button": 1})
        pygame.event.post(ev_down)
        pygame.event.post(ev_up)

    def _get_state_vec(self) -> np.ndarray:
        """把内部关键信息拼成固定长度向量"""
        score = getattr(self.game, "score", 0)
        lives = getattr(self.game, "lives", 0)
        solved = getattr(self.game, "solved_blocks", 0)
        tick = self.tick
        return np.array([score, lives, solved, tick], dtype=np.float32)

    def _save_step_data(self, image: np.ndarray, audio: np.ndarray, state: np.ndarray, 
                       action: int, reward: float, is_reset: bool = False):
        """保存步骤数据到文件"""
        try:
            from PIL import Image
            
            # 生成文件名
            if is_reset:
                filename_base = f"ep{self.episode_count:04d}_reset"
            else:
                filename_base = f"ep{self.episode_count:04d}_step{self.tick:04d}"
            
            # 保存图像
            img_path = os.path.join(self.save_dir, "images", f"{filename_base}.png")
            Image.fromarray(image).save(img_path)
            
            # 保存音频
            if sf:
                audio_path = os.path.join(self.save_dir, "audio", f"{filename_base}.wav")
                sf.write(audio_path, audio, AUDIO_SR)
            
            # 获取详细的游戏状态信息
            game_state_info = self._build_detailed_game_state(action, is_reset)
            
            # 保存重构后的元数据（严格按照要求的格式）
            metadata = {
                "episode": self.episode_count,
                "step": self.tick,
                "step_total": self.step_count_total,
                "action": action if action is not None else 0,
                "is_reset": is_reset,
                "difficulty": self.difficulty,
                "timestamp": time.time(),
                "audio_info": {
                    "sample_rate": AUDIO_SR,
                    "duration": len(audio) / AUDIO_SR,
                    "has_sound": not np.allclose(audio, 0),
                    "rms_level": float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
                },
                "game_state": game_state_info
            };
            
            metadata_path = os.path.join(self.save_dir, "metadata", f"{filename_base}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if is_reset:
                print(f"Saved reset data: {filename_base}")
            elif self.tick % 5 == 0:  # 每5步打印一次
                sound_info = "with sound" if not np.allclose(audio, 0) else "silent"
                print(f"Saved step data: {filename_base} ({sound_info})")
                
        except Exception as e:
            print(f"Error saving step data: {e}")

    def _build_detailed_game_state(self, action: int = None, is_reset: bool = False) -> Dict[str, Any]:
        """构建详细的游戏状态信息（供智能体使用的增强版本）"""
        try:
            # 获取基础游戏状态
            base_state_info = self.game.get_state_info() if self.game else {}
            
            end_game = False
            
            # 获取游戏模块引用
            game_module = self.game.game_module if self.game and hasattr(self.game, 'game_module') else None
            
            if not game_module:
                print("无法获取游戏模块，返回基础状态")
                return {
                    "current_score": base_state_info.get("score", 0),
                    "attempts": base_state_info.get("attempts", 0),
                    "sequence_length": base_state_info.get("sequence_length", 0),
                    "input_length": base_state_info.get("input_length", 0),
                    "game_over": base_state_info.get("game_over", False),
                    "currently_in_correct_sequence": False,
                    "needs_restart_from_beginning": False,
                    "current_correct_sequence": [],
                    "previous_clicks": [],
                    "last_clicked_block_color": "",
                    "last_last_clicked_block_color": "",
                    "last_clicked_action": 0,
                    "available_colors": [],
                    "color_note_mapping": {}
                }
            
            # 从游戏模块获取详细信息
            player_input = getattr(game_module, 'player_melody_input', [])
            correct_sequence = getattr(game_module, 'correct_melody_sequence', [])
            attempts = getattr(game_module, 'melody_puzzle_attempts', 0)
            current_state = getattr(game_module, 'current_state', 'unknown')
            player_score = getattr(game_module, 'player_score', 0)
            current_note_color_mapping = getattr(game_module, 'current_note_color_mapping', {})

            if len(player_input) > 0:  # First check if player_input has elements
                if len(self.current_correct_sequence) != 0:
                    if len(self.current_correct_sequence) < len(correct_sequence) and player_input[-1] == correct_sequence[len(self.current_correct_sequence)]:
                        self.current_correct_sequence.append(self.game._convert_note_id_to_color_name(player_input[-1], game_module))
                    else:
                        self.current_correct_sequence = []
                else:
                    if player_input[-1] == correct_sequence[0]:
                        self.current_correct_sequence.append(self.game._convert_note_id_to_color_name(player_input[-1], game_module))

            # 分析序列正确性
            if len(self.current_correct_sequence) != 0:
                currently_in_correct_sequence = True       
            else:
                currently_in_correct_sequence = False
            
            needs_restart_from_beginning = not currently_in_correct_sequence 
            
            # 构建历史点击序列（最近的点击）
            current_difficulty = getattr(game_module, 'current_difficulty', 'normal')
            
            if current_difficulty == "Hard":
                previous_clicks = []
                if len(self.current_correct_sequence)== 7:
                    end_game = True
                    
                if len(player_input) > 0:
                    recent_inputs = player_input[-7:] if len(player_input) > 7 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            elif current_difficulty == "Medium":
                previous_clicks = []
                if len(player_input) > 0:
                    if len(self.current_correct_sequence)== 5:
                        end_game = True                  
                    recent_inputs = player_input[-5:] if len(player_input) > 5 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            else: 
                previous_clicks = []
                if len(player_input) > 0:
                    if len(self.current_correct_sequence)== 3:
                            end_game = True   
                    recent_inputs = player_input[-3:] if len(player_input) > 3 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            
            # 最后点击的信息
            last_clicked_block_color = ""
            last_last_clicked_block_color = ""
            last_clicked_action = 0
            
            if len(player_input) > 0:
                last_note_id = player_input[-1]
                last_clicked_block_color = self.game._convert_note_id_to_color_name(last_note_id, game_module) or ""
                last_clicked_action = last_note_id
            
            if len(player_input) > 1:
                last_last_note_id = player_input[-2]
                last_last_clicked_block_color = self.game._convert_note_id_to_color_name(last_last_note_id, game_module) or ""
            
            # 获取可用颜色信息
            available_colors = self.game._get_available_colors_from_game_module(game_module)
            
            # 获取颜色到音符的映射
            color_note_mapping = self.game._get_color_note_mapping_from_game_module(game_module)
            
            # 构建详细状态
            detailed_state = {
                "current_score": player_score,
                "attempts": attempts,
                "sequence_length": len(correct_sequence),
                "input_length": len(player_input),
                "current_state": current_state,
                "game_over": base_state_info.get("game_over", False),
                "currently_in_correct_sequence": currently_in_correct_sequence,
                "needs_restart_from_beginning": needs_restart_from_beginning,
                "current_correct_sequence": self.current_correct_sequence.copy(),
                "previous_clicks": previous_clicks,
                "last_clicked_block_color": last_clicked_block_color,
                "last_last_clicked_block_color": last_last_clicked_block_color,
                "last_clicked_action": last_clicked_action,
                "available_colors": available_colors,
                "color_note_mapping": color_note_mapping,
                # 额外信息
                "player_input_sequence": player_input.copy(),
                "correct_sequence": correct_sequence.copy(),
                "current_note_color_mapping": current_note_color_mapping.copy(),
                "difficulty": getattr(game_module, 'current_difficulty', 'normal'),
            }
            
            if self.verbose and action is not None:
                print(f"Built detailed state for action {action}:")
                print(f"  Currently correct: {currently_in_correct_sequence}")
                print(f"  Needs restart: {needs_restart_from_beginning}")
                print(f"  Input/Sequence: {len(player_input)}/{len(correct_sequence)}")
                print(f"  Last color: {last_clicked_block_color}")
                print(f"  Available colors: {len(available_colors)}")
            
            return detailed_state, end_game
            
        except Exception as e:
            if self.verbose:
                print(f"Error building detailed game state: {e}")
                import traceback
                traceback.print_exc()
            return {}
        
    def get_available_colors(self):
            if hasattr(self.game_module, 'get_available_colors'):
                available_colors = self.game_module.get_available_colors()
                if available_colors:
                    return available_colors
            
            # 备选方案：从当前序列和颜色映射中获取
            if (hasattr(self.game_module, 'correct_melody_sequence') and 
                hasattr(self.game_module, 'current_note_color_mapping')):
                
                current_sequence = self.game_module.correct_melody_sequence
                current_mapping = self.game_module.current_note_color_mapping
                
                if current_sequence and current_mapping:
                    available_colors = []
                    from game.sound_alchemist.sound_alchemist_game import ALL_COLORS
                    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
                    
                    for note_id in current_sequence:
                        if note_id in current_mapping:
                            color = current_mapping[note_id]
                            color_name = color_name_mapping.get(color, f"RGB{color}")
                            note_name = self.game_module.NOTE_DISPLAY_NAMES.get(note_id, note_id)
                            available_colors.append({
                                "color_name": color_name,
                                "note_name": note_name,
                                "note_id": note_id,
                                "color_rgb": color
                            })
                    
                    return available_colors
            

    def _analyze_sequence_correctness(self, player_input: List, correct_sequence: List) -> bool:
        """分析玩家输入序列是否正确"""
        if not player_input or not correct_sequence:
            return len(player_input) == 0  # 空输入是正确的
        
        # 检查每个位置是否匹配
        for i in range(len(player_input)):
            if i >= len(correct_sequence) or player_input[i] != correct_sequence[i]:
                return False
        
        return True
    
    def get_detailed_game_state_for_agent(self) -> Dict[str, Any]:
        """专门为智能体提供的详细游戏状态获取方法"""
        
        return self._build_detailed_game_state(action=None, is_reset=False)
    
    
    def _add_to_sequence(self, step_type: str, **kwargs):
        """添加步骤到当前episode序列"""
        if not self.save_sequence:
            return
            
        try:
            step_data = {
                "step_type": step_type,
                "timestamp": time.time(),
                **kwargs
            }
            
            # 清理不可序列化的数据
            step_data = self._clean_sequence_data(step_data)
            
            self.current_episode_sequence.append(step_data)
            
        except Exception as e:
            if self.verbose:
                print(f"Error adding to sequence: {e}")

    @property
    def verbose(self):
        """获取verbose属性"""
        return getattr(self, '_verbose', False)

    @verbose.setter
    def verbose(self, value):
        """设置verbose属性"""
        self._verbose = value

    def _save_episode_sequence(self):
        """保存episode序列数据"""
        if not self.current_episode_sequence:
            return
            
        try:
            timestamp = time.time()
            filename = f"episode_{self.episode_count:04d}_{int(timestamp)}.json"
            filepath = os.path.join(self.sequence_save_dir, filename)
            
            # 清理序列数据
            cleaned_sequence = self._clean_sequence_data(self.current_episode_sequence)
            
            episode_data = {
                "episode": self.episode_count,
                "timestamp": timestamp,
                "difficulty": self.difficulty,
                "total_steps": len(cleaned_sequence),
                "sequence": cleaned_sequence
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            if hasattr(self, '_verbose') and self._verbose:
                print(f"Saved episode sequence to: {filepath}")
                
        except Exception as e:
            if hasattr(self, '_verbose') and self._verbose:
                print(f"Error saving episode sequence: {e}")

    def _clean_sequence_data(self, data: Any) -> Any:
        """清理序列数据中的不可序列化内容"""
        if isinstance(data, dict):
            return {k: self._clean_sequence_data(v) for k, v in data.items() if k not in ['image', 'audio']}
        elif isinstance(data, list):
            return [self._clean_sequence_data(item) for item in data]
        elif isinstance(data, (np.ndarray, np.integer, np.floating)):
            return data.tolist() if hasattr(data, 'tolist') else str(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)

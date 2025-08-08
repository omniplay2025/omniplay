import numpy as np
import pygame
import io
import base64
import json
import os
import tempfile
import time
import requests
from PIL import Image
import datetime
import pathlib
import re
from moviepy.editor import VideoFileClip
import cv2

# Baichuan FastAPI Server Configuration
FASTAPI_BASE_URL = ""  # Baichuan model FastAPI server address

# 导入你的游戏环境
from rhythm_memory_gym_env import RhythmMemoryEnv  # 请替换为实际的导入路径


class ModelRhythmMemoryRunner:
    def __init__(self, difficulty=1, max_episodes=50):
        # 初始化Pygame
        pygame.init()
        pygame.mixer.init()
        
        self.difficulty = difficulty    # 存储游戏设置
        self.max_episodes = max_episodes
        self.env = RhythmMemoryEnv(difficulty)  # 创建游戏环境
        self.session = requests.Session()    # 初始化HTTP会话
        self.session_id = None  # 当前会话ID
        
        # 使用项目目录下的固定路径而不是系统临时目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join(base_dir, "ai_data", f"rhythm_memory_ai_{timestamp}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # 创建保存模型输入数据的目录
        self.model_input_dir = os.path.join(self.temp_dir, "model_inputs")
        os.makedirs(self.model_input_dir, exist_ok=True)
        self.input_counter = 0  # 用于给保存的文件编号

        # 游戏统计
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_score": 0,
            "sequence_analysis_errors": 0,
            "click_prediction_errors": 0,
            "unparseable_sequences": 0,
            "episodes_data": []  # 存储每个episode的详细数据
        }

        # 添加图标到坐标的映射（根据游戏环境）
        self.icon_names = ["dog", "cat", "bird", "cow", "sheep", "chicken", "piano", "trumpet", "drum", "flute"]
        
        # 设置系统提示
        self.system_prompt = """
You are a professional AI assistant for a sound-based memory game.

Game Rules:
1. The game first plays an audiovisual sequence where each icon lights up and plays a corresponding sound.
2. Your task is to remember the order of the sequence.
3. Then, repeat the sequence by clicking the icons in the same order.
4. Icons include animals (dog, cat, bird, cow, sheep, chicken) and musical instruments (piano, trumpet, drum, flute).

Input Information:
1. Video – shows the sequence being played, with icons lighting up in order.
2. Audio – plays the sound associated with each icon in the sequence.
3. Screenshot – shows the current layout of the icons on the game interface.

Your Task:
1. Watch the video and listen to the audio to memorize the order and position of each icon in the sequence.
2. Analyze the game interface screenshot to identify the position of each icon.
3. Based on your memory of the sequence, provide the coordinates for which icon should be clicked next.

Coordinate System:
- Icons are arranged in a grid, starting from the top-left corner.
- Rows and columns are both 1-indexed.
- For example: the icon in the first row and first column has the coordinate (1, 1).
- The icon in the second row and third column has the coordinate (2, 3).
"""

    def save_model_input_data(self, query, files_info, description=""):
        """保存传入模型的数据到本地"""
        try:
            self.input_counter += 1
            input_dir = os.path.join(self.model_input_dir, f"input_{self.input_counter:03d}_{description}")
            os.makedirs(input_dir, exist_ok=True)
            
            # 保存查询文本
            query_file = os.path.join(input_dir, "query.txt")
            with open(query_file, 'w', encoding='utf-8') as f:
                f.write(query)
            
            # 保存系统提示
            system_file = os.path.join(input_dir, "system_prompt.txt")
            with open(system_file, 'w', encoding='utf-8') as f:
                f.write(self.system_prompt)
            
            # 复制媒体文件到输入目录
            for file_type, file_path in files_info.items():
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(input_dir, f"{file_type}_{filename}")
                    import shutil
                    shutil.copy2(file_path, dest_path)
            
            print(f"模型输入数据已保存到: {input_dir}")
            return input_dir
            
        except Exception as e:
            print(f"保存模型输入数据时出错: {e}")
            return None

    def clear_session(self):
        """Clear current session"""
        if self.session_id:
            try:
                url = f"{FASTAPI_BASE_URL}/clear_session"
                data = {"session_id": self.session_id}
                response = self.session.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    print("✅ Session cleared")
                else:
                    print(f"⚠️ Failed to clear session: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Session clearing error: {e}")
        self.session_id = None

    def call_baichuan_api(self, user_query, video_path=None, audio_path=None, image_path=None, description=""):
        """Call Baichuan model API"""
        try:
            # Save input data locally
            files_info = {}
            if video_path: files_info["video"] = video_path
            if audio_path: files_info["audio"] = audio_path
            if image_path: files_info["image"] = image_path
            self.save_model_input_data(user_query, files_info, description)
            
            data = {
                "query": user_query,
                "system_prompt": self.system_prompt,
                "audiogen_flag": False,
                "session_id": self.session_id
            }
            
            # 准备文件上传
            files = []
            if video_path and os.path.exists(video_path):
                files.append(('video_files', ('sequence_video.mp4', open(video_path, 'rb'), 'video/mp4')))
            if audio_path and os.path.exists(audio_path):
                files.append(('audio_file', ('sequence_audio.wav', open(audio_path, 'rb'), 'audio/wav')))
            if image_path and os.path.exists(image_path):
                files.append(('image_files', ('screen_capture.jpg', open(image_path, 'rb'), 'image/jpeg')))
            
            url = f"{FASTAPI_BASE_URL}/chat"
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=300)
            finally:
                # 关闭文件句柄
                for _, file_tuple in files:
                    file_tuple[1].close()
            
            if response.status_code == 200:
                response_data = response.json()
                model_response = response_data.get("text", "")
                self.session_id = response_data.get("session_id")
                return model_response
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Error message: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ API call error: {e}")
            return None

    def extract_sequence_and_icons_from_response(self, response):
        """从模型响应中提取序列坐标和图标信息"""
        try:
            sequence_coords = None
            sequence_icons = None
            
            # 提取序列坐标
            sequence_match = re.search(r"Sequence analysis:\s*\[([^\]]+)\]", response, re.IGNORECASE | re.DOTALL)
            if sequence_match:
                sequence_text = sequence_match.group(1)
                coord_matches = re.findall(r"\((\d+),(\d+)\)", sequence_text)
                if coord_matches:
                    sequence_coords = [(int(r), int(c)) for r, c in coord_matches]
            
            # 提取序列图标
            icons_match = re.search(r"Sequence icons:\s*\[([^\]]+)\]", response, re.IGNORECASE | re.DOTALL)
            if icons_match:
                icons_text = icons_match.group(1)
                # 提取引号内的图标名称
                icon_matches = re.findall(r"['\"]([^'\"]+)['\"]", icons_text)
                if icon_matches:
                    sequence_icons = [icon.strip() for icon in icon_matches]
                else:
                    # 尝试提取没有引号的图标名称
                    icon_matches = re.findall(r"([a-zA-Z]+)", icons_text)
                    if icon_matches:
                        sequence_icons = [icon.strip() for icon in icon_matches]
            
            return sequence_coords, sequence_icons
            
        except Exception as e:
            print(f"提取序列和图标信息错误: {e}")
            return None, None
    
    def calculate_coordinate_accuracy(self, predicted_coords, actual_coords):
        """计算坐标预测的准确性"""
        if not predicted_coords or not actual_coords:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_coords), len(actual_coords))
        
        for i in range(min_length):
            if predicted_coords[i] == actual_coords[i]:
                correct_count += 1
            else:
                break  # 顺序错误则停止计数
        
        return correct_count

    def calculate_icon_accuracy(self, predicted_icons, actual_icons):
        """计算图标预测的准确性"""
        if not predicted_icons or not actual_icons:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_icons), len(actual_icons))
        
        for i in range(min_length):
            if predicted_icons[i] == actual_icons[i]:
                correct_count += 1
            else:
                break  # 顺序错误则停止计数
        
        return correct_count

    def prepare_ground_truth_data(self):
        """准备正确的序列数据（坐标和图标）"""
        try:
            # 获取实际序列坐标
            actual_coords = []
            actual_icons = []
            
            print(f"环境序列信息: {self.env.sequence}")
            print(f"环境布局: {self.env.rows}行 x {self.env.cols}列")
            
            for i, shape_index in enumerate(self.env.sequence):
                # sequence中存储的是shapes数组的索引
                if isinstance(shape_index, int) and 0 <= shape_index < len(self.env.shapes):
                    # 将shape索引转换为行列坐标
                    row = shape_index // self.env.cols
                    col = shape_index % self.env.cols
                    actual_coords.append((row + 1, col + 1))  # 转换为1-based坐标
                    
                    # 获取对应的图标名称
                    icon_name = self.get_icon_at_position(row, col)
                    actual_icons.append(icon_name)
                    
                    print(f"序列位置 {i}: shape_index={shape_index} -> 坐标=({row+1},{col+1}) -> 图标={icon_name}")
                else:
                    print(f"无效的序列索引: {shape_index}, 类型: {type(shape_index)}")
                    actual_coords.append((1, 1))
                    actual_icons.append("unknown")
            
            print(f"最终正确坐标序列: {actual_coords}")
            print(f"最终正确图标序列: {actual_icons}")
            
            return actual_coords, actual_icons
            
        except Exception as e:
            print(f"准备正确序列数据错误: {e}")
            print(f"错误详情:")
            print(f"  环境序列: {getattr(self.env, 'sequence', 'None')}")
            print(f"  环境形状数量: {len(getattr(self.env, 'shapes', []))}")
            print(f"  环境行列: {getattr(self.env, 'rows', 'None')} x {getattr(self.env, 'cols', 'None')}")
            
            # 返回基于序列长度的默认数据
            sequence_len = len(getattr(self.env, 'sequence', []))
            default_coords = [(1, 1)] * sequence_len
            default_icons = ["unknown"] * sequence_len
            return default_coords, default_icons

    def analyze_sequence_with_model(self, video_path, audio_path):
        try:
            # 保存当前屏幕截图到项目目录
            screen_array = self.capture_screen()
            screen_image = Image.fromarray(screen_array)
            screen_image_path = os.path.join(self.temp_dir, "screen_capture.jpg")
            screen_image.save(screen_image_path)
            
            if self.difficulty == 1:
                layout_description = "6 icons, arranged in 2 rows and 3 columns"
            elif self.difficulty == 2:
                layout_description = "10 icons, arranged in 2 rows and 5 columns"
            elif self.difficulty == 3:
                layout_description = "15 icons, arranged in 3 rows and 5 columns"
                        
            user_query = f"""   
            Please analyze the sequence in this rhythm memory game:
                        
            1. Watch the video which shows the icons lighting up in a specific order over time.
            2. Listen to the audio, as the sounds also provide information about the sequence.
            3. Memorize the exact order in which the icons appear in the sequence.
                        
            Game layout: {layout_description}

            Available icons include: dog, cat, bird, cow, sheep, chicken, piano, trumpet, drum, flute.

            Please carefully observe the lighting sequence in the video and listen to the order of sounds in the audio to accurately remember the sequence.

            The sequence length must match the total number of icons in the layout
            For 6 icons: sequence length = 6, for 10 icons: sequence length = 10, for 15 icons: sequence length = 15

            [IMPORTANT!!!] Your response must end in one of the following formats:
            For sequence analysis:
            Sequence icons: [icon1, icon2, ..., iconN]
            Sequence analysis: [(row1,col1) (row2,col2) ...(rowN,colN)]
            """
            
            print("Starting sequence analysis...")
            model_response = self.call_baichuan_api(
                user_query,
                video_path=video_path,
                audio_path=audio_path,
                image_path=screen_image_path if os.path.exists(screen_image_path) else None,
                description="sequence_analysis"
            )
            
            if model_response is None:
                print("Sequence analysis failed")
                self.stats["sequence_analysis_errors"] += 1
                return None, None, None
            
            print("-" * 60)
            print("Model sequence analysis result:", model_response)
            print("-" * 60)
            
            # 从响应中提取序列坐标和图标信息
            predicted_coords, predicted_icons = self.extract_sequence_and_icons_from_response(model_response)
            
            if predicted_coords is None:
                print("Unable to parse sequence coordinate response")
                self.stats["unparseable_sequences"] += 1
                return None, None, model_response
            
            return predicted_coords, predicted_icons, model_response
            
        except Exception as e:
            print(f"Sequence analysis error: {e}")
            self.stats["sequence_analysis_errors"] += 1
            return None, None, None

    def get_click_suggestion(self, sequence_info, current_step, last_click_feedback_dir):
        try:
            user_query = f"""
Now it's time to repeat the remembered sequence.

Current status:
- This is step {current_step + 1} in the sequence.
- Sequence information: {sequence_info}
- You will be given the feedback from the previous click (highlighted image and sound).

Your task:
1. Analyze the image and audio feedback from the previous click.
2. Based on your memory of the original sequence, determine which icon comes next.
3. Provide the correct row and column coordinates for the next click.

Note:
- The icon grid starts from the top-left corner.
- Coordinates are 1-based (i.e., row 1, column 1 is written as (1,1)).

[IMPORTANT!!!] Your response must end with one of the following formats:
For click suggestion:
Suggested click: (row,col)
Example: Suggested click: (1,2) or Suggested click: (2,3)
"""

            click_frame_path = None
            click_audio_path = None
            
            if last_click_feedback_dir:
                click_frame_path = os.path.join(last_click_feedback_dir, "click_frame.png")
                click_audio_path = os.path.join(last_click_feedback_dir, "click_audio.wav")
            
            print(f"Getting click suggestion for step {current_step + 1}...")
            model_response = self.call_baichuan_api(
                user_query,
                audio_path=click_audio_path if click_audio_path and os.path.exists(click_audio_path) else None,
                image_path=click_frame_path if click_frame_path and os.path.exists(click_frame_path) else None,
                description=f"click_suggestion_step_{current_step + 1}"
            )
            
            if model_response is None:
                print("Failed to get click suggestion")
                return None
            
            print("Model click position selection:")
            print("-" * 40)
            print(model_response)

            click_coord = self.extract_click_from_response(model_response)
            return click_coord
            
        except Exception as e:
            print(f"Click suggestion error: {e}")
            self.stats["click_prediction_errors"] += 1
            return None
    
    def extract_sequence_from_response(self, response):
        """从模型响应中提取序列信息（保持向后兼容）"""
        try:
            predicted_coords, _ = self.extract_sequence_and_icons_from_response(response)
            return predicted_coords
        except Exception as e:
            print(f"提取序列信息错误: {e}")
            return None



    def extract_click_from_response(self, response):
        try:
            # Standard format matching
            click_match = re.search(r"Suggested click:\s*\(?\s*(\d+)\s*,?\s+(\d+)\s*\)?", response, re.IGNORECASE)
            if click_match:
                row = int(click_match.group(1))
                col = int(click_match.group(2))
                return (row - 1, col - 1)  # Convert to 0-based index
            
            # Try other formats
            coord_match = re.search(r"\((\d+),(\d+)\)", response)
            if coord_match:
                row = int(coord_match.group(1))
                col = int(coord_match.group(2))
                return (row - 1, col - 1)  # Convert to 0-based index
            
            print("Failed to extract click coordinates from response")
            self.stats["click_prediction_errors"] += 1
            return None
            
        except Exception as e:
            print(f"Click coordinate extraction error: {e}")
            self.stats["click_prediction_errors"] += 1
            return None

    def save_stats(self, round_num):
        """保存统计信息到文件"""
        save_dir = pathlib.Path("./rhythm_memory_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        script_name = os.path.basename(__file__).replace('.py', '')
        
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("===== 节奏记忆游戏统计 =====\n")
            f.write(f"难度级别: {self.difficulty}\n")
            f.write(f"回合数: {self.stats['rounds']}\n")
            f.write(f"成功回合数: {self.stats['successful_rounds']}\n")
            f.write(f"总分: {self.stats['total_score']}\n")
            f.write(f"序列分析错误次数: {self.stats['sequence_analysis_errors']}\n")
            f.write(f"点击预测错误次数: {self.stats['click_prediction_errors']}\n")
            f.write(f"成功率: {(self.stats['successful_rounds']/max(self.stats['rounds'], 1))*100:.1f}%\n")
            f.write(f"保存时间: {timestamp}\n")
        
        print(f"统计信息已保存到: {filepath}")


    def run_single_round(self, round_num):
        print(f"\n===== 开始第 {round_num} 回合 =====")
        
        observation = self.env.reset()
        self.env.render()
        round_successful = False
        
        try:
            # 阶段1：播放序列
            print("--- 播放序列阶段 ---")
            action = (self.env.rows, self.env.cols)  # 特殊动作开始播放
            obs, reward, done, info = self.env.step(action)
            time.sleep(2)
            video_path = os.path.join(self.env.record_dir, "sequence_video.mp4")    # 获取生成的视频和音频文件
            audio_path = os.path.join(self.env.record_dir, "sequence_audio.wav")
            max_wait = 10
            wait_time = 0
            while not os.path.exists(video_path) and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            # 分析序列
            predicted_coords, predicted_icons, _ = self.analyze_sequence_with_model(video_path, audio_path)    # 分析序列
            if predicted_coords is None:
                print("序列分析失败")
                return False
            
            # 阶段2：重复序列
            print("\n--- 重复序列阶段 ---")
            current_step = 0
            while not done and current_step < len(self.env.sequence):
                if current_step == 0:
                    # 第一步直接用模型分析的序列，不用模型再看图预测
                    coord = predicted_coords[0]
                    click_coord = (coord[0] - 1, coord[1] - 1)
                    print(f"(Step 1) 直接使用分析得到的点击位置: {click_coord}")
                else:
                    click_coord = self.get_click_suggestion(predicted_coords, current_step, last_click_feedback_dir=self.env.record_dir)
                    if click_coord is None:
                        print("无法获取点击建议")
                        break

                print(f"建议点击索引: {click_coord}")
                obs, reward, done, info = self.env.step(click_coord)
                self.env.render()
                print(f"步骤 {current_step + 1}, 奖励: {reward}, 进度: {obs['progress']}/{len(self.env.sequence)}")
                current_step += 1
                time.sleep(0.5)         # 短暂暂停
            
            # 检查是否成功
            if obs['progress'] == len(self.env.sequence):
                print("🎉 回合成功完成!")
                round_successful = True
                self.stats["successful_rounds"] += 1
                self.stats["total_score"] += obs['progress']
            else:
                print("😞 回合失败")
                self.stats["total_score"] += obs['progress']
            
        except Exception as e:
            print(f"回合执行错误: {e}")
        
        self.stats["rounds"] += 1
        return round_successful


    def calculate_correct_predictions(self, predicted_sequence, actual_sequence):
        """计算预测正确的icon和坐标个数"""
        if not predicted_sequence or not actual_sequence:
            return 0
        
        correct_count = 0
        min_length = min(len(predicted_sequence), len(actual_sequence))
        
        for i in range(min_length):
            if predicted_sequence[i] == actual_sequence[i]:
                correct_count += 1
            else:
                break  # 一旦有错误就停止计数，因为要求顺序正确
        
        return correct_count

    def capture_screen(self):
        screen = self.env.screen
        screen_array = pygame.surfarray.array3d(screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        return screen_array

    def get_icon_at_position(self, row, col):
        """根据位置获取图标名称"""
        try:
            # 计算在shapes列表中的索引
            icon_index = row * self.env.cols + col
            
            # 检查索引是否有效
            if 0 <= icon_index < len(self.env.shapes):
                # 从环境的音频路径中获取图标名称
                if hasattr(self.env, 'sound_manager') and hasattr(self.env.sound_manager, 'sound_paths'):
                    if icon_index < len(self.env.sound_manager.sound_paths):
                        sound_path = self.env.sound_manager.sound_paths[icon_index]
                        import os
                        filename = os.path.basename(sound_path)
                        icon_name = os.path.splitext(filename)[0].lower()
                        return icon_name
                
                # 备用方案：使用图标路径
                elif hasattr(self.env, 'icon_paths') and icon_index < len(self.env.icon_paths):
                    icon_path = self.env.icon_paths[icon_index]
                    import os
                    filename = os.path.basename(icon_path)
                    icon_name = os.path.splitext(filename)[0].lower()
                    return icon_name
                
                # 最后备用方案
                else:
                    if icon_index < len(self.icon_names):
                        return self.icon_names[icon_index]
                    else:
                        return "unknown"
            else:
                print(f"图标索引超出范围: {icon_index}, shapes长度: {len(self.env.shapes)}")
                return "unknown"
            
        except Exception as e:
            print(f"获取图标名称错误: {e}")
            return "unknown"

    def save_episode_results(self):
        """Save all episode results to files"""
        save_dir = pathlib.Path("./rhythm_memory_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # Save detailed JSON results
        json_filename = f"{script_name}_detailed_results_{timestamp}.json"
        json_filepath = save_dir / json_filename
        
        detailed_results = {
            "experiment_info": {
                "difficulty": self.difficulty,
                "total_episodes": self.stats["total_episodes"],
                "timestamp": timestamp,
                "script_name": script_name
            },
            "overall_stats": self.stats,
            "episodes": self.stats["episodes_data"]
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 保存简要的文本统计
        txt_filename = f"{script_name}_summary_{timestamp}.txt"
        txt_filepath = save_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("===== Rhythm Memory Game Statistics Report =====\n")
            f.write(f"Difficulty Level: {self.difficulty}\n")
            f.write(f"Total Episodes: {self.stats['total_episodes']}\n")
            f.write(f"Successful Episodes: {self.stats['successful_episodes']}\n")
            f.write(f"Success Rate: {(self.stats['successful_episodes']/max(self.stats['total_episodes'], 1))*100:.1f}%\n")
            f.write(f"Total Score: {self.stats['total_score']}\n")
            f.write(f"Average Score: {self.stats['total_score']/max(self.stats['total_episodes'], 1):.2f}\n")
            f.write(f"Sequence Analysis Errors: {self.stats['sequence_analysis_errors']}\n")
            f.write(f"Click Prediction Errors: {self.stats['click_prediction_errors']}\n")
            f.write(f"Unparseable Sequences: {self.stats['unparseable_sequences']}\n")
            f.write(f"Save Time: {timestamp}\n\n")
            
            f.write("===== Detailed Results for Each Episode =====\n")
            for ep_data in self.stats["episodes_data"]:
                f.write(f"Episode {ep_data['episode_num']}: ")
                f.write(f"Success={ep_data['success']}, ")
                f.write(f"Score={ep_data['score']}, ")
                f.write(f"Correct Coordinates={ep_data['correct_coordinates']}/{ep_data['sequence_length']}, ")
                f.write(f"Correct Icons={ep_data['correct_icons']}/{ep_data['sequence_length']}, ")
                f.write(f"Sequence Parseable={ep_data['sequence_parseable']}\n")
        
        print(f"Detailed results saved to: {json_filepath}")
        print(f"Summary statistics saved to: {txt_filepath}")
        return json_filepath, txt_filepath

    def run_single_episode(self, episode_num):
        print(f"\n===== Episode {episode_num}/{self.max_episodes} =====")
        
        # Clear session at the start of each episode
        self.clear_session()
        
        # 初始化episode数据
        episode_data = {
            "episode_num": episode_num,
            "success": False,
            "score": 0,
            "sequence_length": 0,
            "correct_coordinates": 0,
            "correct_icons": 0,
            "sequence_parseable": False,
            "actual_coordinates": [],
            "actual_icons": [],
            "predicted_coordinates": [],
            "predicted_icons": [],
            "model_raw_response": "",
            "error_type": None
        }
        
        observation = self.env.reset()
        self.env.render()
        
        try:
            # 阶段1：播放序列
            print("--- 播放序列阶段 ---")
            action = (self.env.rows, self.env.cols)  # 特殊动作开始播放
            obs, reward, done, info = self.env.step(action)
            time.sleep(2)
            
            # 准备正确的序列数据
            actual_coordinates, actual_icons = self.prepare_ground_truth_data()
            episode_data["actual_coordinates"] = actual_coordinates
            episode_data["actual_icons"] = actual_icons
            episode_data["sequence_length"] = len(actual_coordinates)
            
            print(f"正确坐标序列: {actual_coordinates}")
            print(f"正确图标序列: {actual_icons}")
            
            video_path = os.path.join(self.env.record_dir, "sequence_video.mp4")
            audio_path = os.path.join(self.env.record_dir, "sequence_audio.wav")
            max_wait = 10
            wait_time = 0
            while not os.path.exists(video_path) and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            # 分析序列
            predicted_coords, predicted_icons, raw_response = self.analyze_sequence_with_model(video_path, audio_path)
            
            if predicted_coords is None:
                print("序列分析失败，使用随机策略继续游戏")
                episode_data["sequence_parseable"] = False
                episode_data["error_type"] = "sequence_analysis_failed"
                # 生成随机序列作为fallback
                predicted_coords = [(1, 1)] * len(self.env.sequence)
                predicted_icons = ["unknown"] * len(self.env.sequence)
            else:
                episode_data["sequence_parseable"] = True
                episode_data["model_raw_response"] = raw_response or ""
            
            episode_data["predicted_coordinates"] = predicted_coords
            episode_data["predicted_icons"] = predicted_icons or []
            
            print(f"预测坐标序列: {predicted_coords}")
            print(f"预测图标序列: {predicted_icons}")
            
            # 计算准确性
            episode_data["correct_coordinates"] = self.calculate_coordinate_accuracy(
                predicted_coords, actual_coordinates
            )
            episode_data["correct_icons"] = self.calculate_icon_accuracy(
                predicted_icons, actual_icons
            )
            
            print(f"坐标预测准确性: {episode_data['correct_coordinates']}/{episode_data['sequence_length']}")
            print(f"图标预测准确性: {episode_data['correct_icons']}/{episode_data['sequence_length']}")
            
            # 阶段2：重复序列
            print("\n--- 重复序列阶段 ---")
            current_step = 0
            while not done and current_step < len(self.env.sequence):
                if current_step == 0:
                    # 第一步直接用模型分析的序列
                    if predicted_coords and len(predicted_coords) > 0:
                        coord = predicted_coords[0]
                        click_coord = (coord[0] - 1, coord[1] - 1)
                    else:
                        # 如果序列解析失败，使用随机点击
                        click_coord = (0, 0)
                    print(f"(Step 1) 使用点击位置: {click_coord}")
                else:
                    if episode_data["sequence_parseable"]:
                        click_coord = self.get_click_suggestion(predicted_coords, current_step, 
                                                              last_click_feedback_dir=self.env.record_dir)
                    else:
                        # 如果序列不可解析，使用随机策略
                        click_coord = (current_step % self.env.rows, current_step % self.env.cols)
                    
                    if click_coord is None:
                        print("无法获取点击建议，使用随机点击")
                        click_coord = (current_step % self.env.rows, current_step % self.env.cols)

                print(f"点击坐标: {click_coord}")
                obs, reward, done, info = self.env.step(click_coord)
                self.env.render()
                print(f"步骤 {current_step + 1}, 奖励: {reward}, 进度: {obs['progress']}/{len(self.env.sequence)}")
                current_step += 1
                time.sleep(0.5)
            
            # 记录最终结果
            episode_data["score"] = obs['progress']
            if obs['progress'] == len(self.env.sequence):
                print("🎉 Episode成功完成!")
                episode_data["success"] = True
                self.stats["successful_episodes"] += 1
            else:
                print("😞 Episode失败")
            
            self.stats["total_score"] += obs['progress']
            
        except Exception as e:
            print(f"Episode执行错误: {e}")
            episode_data["error_type"] = f"execution_error: {str(e)}"
        
        # 添加episode数据到统计中
        self.stats["episodes_data"].append(episode_data)
        self.stats["total_episodes"] += 1
        
        # 打印当前episode统计
        print(f"Episode {episode_num} 结果:")
        print(f"  成功: {episode_data['success']}")
        print(f"  得分: {episode_data['score']}/{episode_data['sequence_length']}")
        print(f"  坐标正确: {episode_data['correct_coordinates']}/{episode_data['sequence_length']}")
        print(f"  图标正确: {episode_data['correct_icons']}/{episode_data['sequence_length']}")
        print(f"  序列可解析: {episode_data['sequence_parseable']}")
        
        return episode_data["success"]

    def run(self):
        print(f"启动节奏记忆游戏AI测试 - 难度: {self.difficulty}")
        print(f"将运行 {self.max_episodes} 个episodes")
        print(f"数据保存路径: {self.temp_dir}")
        
        # 检查百川API连接
        try:
            response = self.session.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 百川模型 API 连接正常")
            else:
                print("❌ 百川模型 API 连接异常")
                return
        except Exception as e:
            print(f"❌ 无法连接到百川模型 API: {e}")
            return

        start_time = time.time()
        
        try:
            for episode_num in range(1, self.max_episodes + 1):
                success = self.run_single_episode(episode_num)
                
                # 打印进度
                current_success_rate = (self.stats["successful_episodes"] / self.stats["total_episodes"]) * 100
                print(f"进度: {episode_num}/{self.max_episodes}, 当前成功率: {current_success_rate:.1f}%")
                
                # 在episodes之间短暂暂停
                if episode_num < self.max_episodes:
                    time.sleep(1)
                
                # 每10个episode保存一次中间结果
                if episode_num % 10 == 0:
                    print(f"已完成 {episode_num} episodes，保存中间结果...")
                    self.save_episode_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 打印最终统计
            print("\n" + "="*60)
            print("===== 最终统计结果 =====")
            print(f"总episodes: {self.stats['total_episodes']}")
            print(f"成功episodes: {self.stats['successful_episodes']}")
            print(f"成功率: {(self.stats['successful_episodes']/max(self.stats['total_episodes'], 1))*100:.1f}%")
            print(f"总分: {self.stats['total_score']}")
            print(f"平均分: {self.stats['total_score']/max(self.stats['total_episodes'], 1):.2f}")
            print(f"序列分析错误: {self.stats['sequence_analysis_errors']}")
            print(f"点击预测错误: {self.stats['click_prediction_errors']}")
            print(f"无法解析序列: {self.stats['unparseable_sequences']}")
            print(f"总耗时: {total_time:.1f}秒")
            print("="*60)
            
            # 保存最终结果
            json_file, txt_file = self.save_episode_results()
            
        except KeyboardInterrupt:
            print("\n用户中断测试")
            print("保存当前结果...")
            self.save_episode_results()
        finally:
            # 清理会话和资源
            self.clear_session()
            self.env.close()
            pygame.quit()
            
            print(f"\n模型输入数据已保存到: {self.model_input_dir}")
            print(f"所有数据保存在目录: {self.temp_dir}")
            print("\n节奏记忆游戏AI测试结束!")


def select_difficulty():
    """选择游戏难度"""
    print("请选择游戏难度:")
    print("1. 简单 (Easy) - 6个图标，2行3列")
    print("2. 普通 (Normal) - 10个图标，2行5列")
    print("3. 困难 (Hard) - 15个图标，3行5列")
    
    while True:
        choice = input("请输入选项 (1-3): ")
        if choice in ["1", "2", "3"]:
            return int(choice)
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    # 选择难度
    # difficulty = select_difficulty()
    difficulty = 1
    max_episodes = 10
    runner = ModelRhythmMemoryRunner(difficulty=difficulty, max_episodes=max_episodes)
    runner.run()
    runner.run()
           
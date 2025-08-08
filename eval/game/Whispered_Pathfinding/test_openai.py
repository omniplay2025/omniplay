import gym
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
from maze_gym_env import MazeGymEnv
import datetime
import pathlib
import argparse

# 导入配置
from config import (
    API_BASE, API_KEY, MODEL_CHAT,
    DEFAULT_DIFFICULTY, DEFAULT_ROUNDS, DEFAULT_MAX_STEPS, DEFAULT_AUTO_SPEED,
    RESULTS_DIR, TEMP_DIR_PREFIX, TEXT_DISPLAY_SIZE, TEXT_DISPLAY_POS, FONT_SIZE,
    DIFFICULTY_MAP, DIFFICULTY_DESCRIPTIONS,
    DEFAULT_SEED, USE_SEQUENTIAL_SEEDS, RANDOM_SEED_RANGE
)

class ModelMazeRunner:
    """使用大模型进行分析的自动迷宫运行器"""
    
    def __init__(self, difficulty=DEFAULT_DIFFICULTY, auto_speed=DEFAULT_AUTO_SPEED, 
                 max_steps=DEFAULT_MAX_STEPS, results_dir=RESULTS_DIR,
                 seed=None, use_sequential_seeds=USE_SEQUENTIAL_SEEDS):
        # 初始化Pygame（用于音频播放和显示）
        pygame.init()
        pygame.mixer.init()
        
        # 存储游戏设置
        self.difficulty = difficulty
        self.auto_speed = auto_speed
        self.max_steps = max_steps
        self.results_dir = results_dir
        
        # Seed configuration
        self.seed = seed
        self.use_sequential_seeds = use_sequential_seeds
        self.current_seed = None
        
        # 初始化HTTP会话
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {API_KEY}"})
        
        # 获取正确的游戏资源路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        game_dir = os.path.dirname(current_dir)  # 回到上级目录 (game)
        assets_dir = os.path.join(game_dir, "assets-necessay")
        
        # 检查assets目录是否存在，如果不存在则尝试其他可能的路径
        if not os.path.exists(assets_dir):
            # 尝试相对路径
            assets_dir = os.path.join("..", "assets-necessay")
            if not os.path.exists(assets_dir):
                # 尝试当前目录下的assets
                assets_dir = "assets-necessay"
                if not os.path.exists(assets_dir):
                    print(f"警告: 未找到assets目录，音频功能可能无法正常工作")
                    assets_dir = None
        
        print(f"使用assets目录: {assets_dir}")
        
        # 创建环境，启用语音导航，设置难度，传递assets路径
        if assets_dir:
            # 设置环境变量或传递参数给MazeGymEnv
            os.environ['MAZE_ASSETS_PATH'] = assets_dir
        
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # 创建临时目录存储音频文件
        self.temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
        
        # 初始化附加显示窗口（可选，用于显示模型分析结果）
        self.text_display_size = TEXT_DISPLAY_SIZE
        self.text_display = pygame.Surface(self.text_display_size)
        self.text_display_pos = TEXT_DISPLAY_POS
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        
        # 跟踪总奖励和当前建议的动作
        self.total_reward = 0
        self.current_suggested_action = None
        self.current_step = 0
        
        # 游戏统计
        self.stats = {
            "steps": 0,
            "total_reward": 0, 
            "invalid_actions": 0  # 新增：记录无效动作次数
        }
        
        # 设置系统提示
        self.system_prompt = """
        You are a professional maze navigation intelligent agent.

        Observation information:
        1. Image - Shows a 3D view of the maze and a mini-map
        2. Audio - Provides voice navigation guidance
        3. State vector - Contains position, orientation, and target information

        Your task is to provide optimal navigation suggestions.

        Executable actions:
        - Forward distance: [-1.0, 3.0], negative values mean moving backward, positive values mean moving forward
        - Rotation angle: [-180.0, 180.0] degrees, negative values mean rotating left, positive values mean rotating right, relative to the current orientation

        Analyze each observation and provide clear action recommendations, including:
        1. A brief description of the current position and surrounding environment
        2. Suggested action (forward/backward distance and rotation angle)
        3. Reasoning for this action (e.g., avoiding walls, facing the target, etc.)

        [IMPORTANT] Your response must end with the following exact format: "Suggested action: [number] [number]"
        For example: "Suggested action: 1.0 45" or "Suggested action: 0.5 -30"
        Do not use any other formats, such as "Suggested action: move forward 1.0, rotate 45", only use the number pair without units.
        """
    
    def play_audio(self, audio_base64):
        """播放音频提示"""
        if not audio_base64:
            return
            
        try:
            # 将base64解码为音频文件
            audio_data = base64.b64decode(audio_base64)
            audio_file = os.path.join(self.temp_dir, "current_guidance.wav")
            
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            # 播放音频
            sound = pygame.mixer.Sound(audio_file)
            sound.play()
            
            # 等待音频播放完成（可选）
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"音频播放错误: {e}")
    
    def get_model_suggestion(self, observation, audio=None):
        """从模型获取动作建议"""
        try:
            # 将图像转换为base64编码
            image = observation['screen']
            img_pil = Image.fromarray(image)
            img_io = io.BytesIO()
            img_pil.save(img_io, format="JPEG")
            img_bytes = img_io.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # 构建向量状态描述
            vector_obs = observation['vector']
            state_description = (
                f"Current position: x={vector_obs[0]:.2f}, y={vector_obs[1]:.2f}\n"
                f"Current orientation: {np.degrees(vector_obs[2]):.1f}°\n"
                f"Distance to target: {vector_obs[3]:.2f}m\n"
                f"Direction to target: {np.degrees(vector_obs[4]):.2f}°\n"
                "Distance to walls: " + 
                ", ".join([f"{i*45}°: {d:.1f}m" for i, d in enumerate(vector_obs[5:])])
            )
            
            # Construct user prompt
            user_content = f"""
Please analyze the current maze environment and provide navigation suggestions.

Environment state information:
{state_description}

Please provide the following:
1. Environment analysis: Describe the current position, orientation, and relationship to the target position
2. Suggested action: Provide specific forward distance and rotation angle
3. Navigation rationale: Explain why you chose this action

Remember to end your response with the format "Suggested action: [forward distance] [rotation angle]".
For example: "Suggested action: 1.0 -45"
"""
            
            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ]
                }
            ]
            
            # 如果有音频数据，添加到消息中
            if audio is not None:
                messages[1]["content"].append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio,
                        "format": "wav",
                    },
                })
            
            # 构建完整的请求负载
            payload = {
                "model": MODEL_CHAT,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # API端点URL
            url = f"{API_BASE}/chat/completions"
            
            print("正在请求模型分析...")
            # 使用session.post()方法发送API请求
            r = self.session.post(url, json=payload, timeout=300)
            
            # 检查响应状态
            if r.status_code != 200:
                print(f"API请求失败，状态码: {r.status_code}")
                print(f"响应内容: {r.text}")
                return f"API请求失败: {r.status_code} - {r.text}", None
            
            # 解析响应JSON
            response_data = r.json()
            
            # 提取文本内容
            try:
                model_response = response_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                print(f"无法从响应中提取文本: {response_data}")
                return "无法从响应中提取文本", None
            
            print("\n模型分析结果:")
            print("-" * 60)
            print(model_response)
            print("-" * 60)
            
            # 从响应中提取建议的动作
            action = self.extract_action_from_response(model_response)
            return model_response, action
            
        except requests.RequestException as e:
            print(f"请求错误: {e}")
            return f"API请求错误: {e}", None
        except Exception as e:
            print(f"模型请求错误: {e}")
            return f"请求模型时出错: {e}", None
    
    def extract_action_from_response(self, response):
        """从模型响应中提取动作"""
        try:
            # 标准格式匹配: "建议动作: 1.0 45"
            import re
            action_match = re.search(r"建议动作:\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            if not action_match:
                action_match = re.search(r"建议动作：\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            
            if action_match:
                forward = float(action_match.group(1))
                rotation = float(action_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # 匹配带"前进"和"旋转"关键词的格式: "建议动作: 前进0.7，旋转45"
            front_turn_match = re.search(r"建议动作:[\s]*前进\s*([-+]?\d*\.?\d+)[,，]?\s*旋转\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE) 
            if not front_turn_match:
                front_turn_match = re.search(r"建议动作：[\s]*前进\s*([-+]?\d*\.?\d+)[,，]?\s*旋转\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
            
            if front_turn_match:
                forward = float(front_turn_match.group(1))
                rotation = float(front_turn_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # 匹配使用逗号分隔的格式: "建议动作: 0.7, 45"
            comma_match = re.search(r"建议动作:[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            if not comma_match:
                comma_match = re.search(r"建议动作：[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            
            if comma_match:
                forward = float(comma_match.group(1))
                rotation = float(comma_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
                
            # 匹配带单位的格式: "建议动作: 0.7米 45度"
            unit_match = re.search(r"建议动作:[\s]*([-+]?\d*\.?\d+)\s*米?\s*([-+]?\d*\.?\d+)\s*度?", response)
            if not unit_match:
                unit_match = re.search(r"建议动作：[\s]*([-+]?\d*\.?\d+)\s*米?\s*([-+]?\d*\.?\d+)\s*度?", response)
            
            if unit_match:
                forward = float(unit_match.group(1))
                rotation = float(unit_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # 如果以上都不匹配，尝试查找建议动作后的数字对
            action_line_match = re.search(r"建议动作:(.+)$|建议动作：(.+)$", response, re.MULTILINE)
            if action_line_match:
                action_line = action_line_match.group(1) or action_line_match.group(2)
                # 从这一行中提取数字
                nums = re.findall(r"[-+]?\d*\.?\d+", action_line)
                if len(nums) >= 2:
                    forward = float(nums[0])
                    rotation = float(nums[1])
                    # 检查范围是否合理
                    if -1.0 <= forward <= 3.0 and -180.0 <= rotation <= 180.0:
                        return np.array([forward, rotation], dtype=np.float32)
            
            # 备用匹配方式，查找文本中最后出现的两个数字
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            if len(numbers) >= 2:
                forward = float(numbers[-2])
                rotation = float(numbers[-1])
                # 检查范围是否合理
                if -1.0 <= forward <= 3.0 and -180.0 <= rotation <= 180.0:
                    return np.array([forward, rotation], dtype=np.float32)
            
            print("未能从响应中提取有效动作，使用默认动作")
            # 记录无效动作
            self.stats["invalid_actions"] += 1
            # 提供一个安全的默认动作
            return np.array([0.0, 0.0], dtype=np.float32)
            
        except Exception as e:
            print(f"提取动作时出错: {e}")
            # 记录无效动作
            self.stats["invalid_actions"] += 1
            # 提供一个安全的默认动作
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_user_input(self):
    
        print(f"使用模型建议的动作: {self.current_suggested_action[0]:.2f} {self.current_suggested_action[1]:.2f}")
        return self.current_suggested_action

    
    def print_observation_info(self, observation):
        """打印观察信息"""
        vector_obs = observation['vector']
        
        print("\n===== 当前状态 =====")
        print(f"位置: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"朝向: {np.degrees(vector_obs[2]):.1f}°")
        print(f"到目标距离: {vector_obs[3]:.2f}米")
        print(f"到目标角度: {np.degrees(vector_obs[4]):.1f}°")
        
        # 输出主要的墙壁距离信息
        directions = ["前方", "右前方", "右侧", "右后方", "后方", "左后方", "左侧", "左前方"]
        for i, direction in enumerate(directions):
            print(f"{direction}墙壁距离: {vector_obs[5+i]:.2f}米")
    
    def save_stats(self, round_num, terminated, truncated):
        """保存统计信息到文件"""
        # 创建保存目录（如果不存在）
        save_dir = pathlib.Path(self.results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取当前时间作为文件名一部分
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 确定当前的代码文件名（不包含.py扩展名）
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # 创建文件名
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        # 写入统计信息
        with open(filepath, 'w') as f:
            f.write("===== 导航统计 =====\n")
            f.write(f"难度级别: {self.difficulty}\n")
            f.write(f"种子值: {self.current_seed}\n")
            f.write(f"总步数: {self.stats['steps']}\n")
            f.write(f"总奖励: {self.stats['total_reward']:.2f}\n")
            f.write(f"无效动作次数: {self.stats['invalid_actions']}\n")
            f.write(f"成功到达目标: {'是' if terminated else '否'}\n")
            if truncated:
                f.write("回合被截断: 是\n")
            f.write(f"保存时间: {timestamp}\n")
        
        print(f"统计信息已保存到: {filepath}")
    
    def run_single_round(self, round_num):
        """运行单回合迷宫导航"""
        print(f"\n===== 开始第 {round_num} 回合 =====")
        print(f"难度级别: {self.difficulty}")
        
        # Set seed for this round
        if self.seed is not None:
            # Use fixed seed
            self.current_seed = self.seed
        elif self.use_sequential_seeds:
            # Use sequential seed: 0, 1, 2, ...
            self.current_seed = round_num - 1
        else:
            # Use random seed
            import random
            self.current_seed = random.randint(*RANDOM_SEED_RANGE)
        
        print(f"使用种子: {self.current_seed}")
        
        # 重置环境
        observation, info = self.env.reset(seed=self.current_seed)
        
        # 重置统计信息
        self.total_reward = 0
        self.current_step = 0
        self.stats = {
            "steps": 0,
            "total_reward": 0,
            "invalid_actions": 0
        }
        
        # 显示初始状态信息
        self.print_observation_info(observation)
        
        # 播放初始音频提示
        if 'audio' in info:
            print("播放语音导航...")
            self.play_audio(info['audio'])
        
        # 自动运行循环
        done = False
        audio_data = None
        
        while not done and self.current_step < self.max_steps:
            # 获取模型建议
            model_response, suggested_action = self.get_model_suggestion(observation, audio_data)
            self.current_suggested_action = suggested_action
            
            if suggested_action is not None:
                print(f"\n模型建议动作: 前进={suggested_action[0]:.2f}, 旋转={suggested_action[1]:.2f}°")
            else:
                self.current_suggested_action = np.array([0.0, 0.0], dtype=np.float32)
                print("\n模型无法提供明确动作建议，使用默认动作")
       
            # 执行动作
            action = self.get_user_input()
            if action is None:  # 用户中断
                return False, False
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.current_step += 1
            
            # 更新环境显示
            self.env.render()
            
            # 显示状态信息
            print(f"\n步骤 {self.current_step}/{self.max_steps}, 总奖励: {self.total_reward:.2f}")
            print(f"到目标距离: {observation['vector'][3]:.2f}米")
            
            # 播放音频提示(如果有)
            if 'audio' in info:
                try:
                    # 解码音频数据
                    audio_data = base64.b64decode(info['audio'])
                    audio_file = os.path.join(self.temp_dir, "last_guidance.wav")
                    
                    with open(audio_file, "wb") as f:
                        f.write(audio_data)
                    
                    # 播放音频
                    self.play_audio(info['audio'])
                    audio_data = info['audio']
                except Exception as e:
                    print(f"处理音频数据时出错: {e}")
                    audio_data = None
            
            # 检查是否结束
            done = terminated or truncated
            if terminated:
                print("\n恭喜! 成功到达目标!")
            elif truncated:
                print("\n回合结束!")
        
        # 更新并显示统计信息
        self.stats["steps"] = self.current_step
        self.stats["total_reward"] = self.total_reward
        
        print("\n===== 导航统计 =====")
        print(f"难度级别: {self.difficulty}")
        print(f"总步数: {self.stats['steps']}")
        print(f"总奖励: {self.stats['total_reward']:.2f}")
        print(f"无效动作次数: {self.stats['invalid_actions']}")
        print(f"成功到达目标: {'是' if terminated else '否'}")
        
        # 保存统计信息
        self.save_stats(round_num, terminated, truncated)
        
        return terminated, truncated
    
    def run(self, total_rounds=DEFAULT_ROUNDS):
        """运行多回合迷宫导航"""
        print(f"启动自动迷宫导航 - 难度: {self.difficulty}")
        print(f"自动速度: {self.auto_speed}秒/步, 最大步数: {self.max_steps}")
        print(f"将运行{total_rounds}个回合，每回合结束后会保存统计数据")
        print(f"结果保存目录: {self.results_dir}")
        
        completed_rounds = 0
        success_rounds = 0
        
        try:
            for round_num in range(1, total_rounds + 1):
                terminated, truncated = self.run_single_round(round_num)
                completed_rounds += 1
                if terminated:
                    success_rounds += 1
                
                # 在回合之间暂停一小段时间
                if round_num < total_rounds:
                    print(f"\n等待2秒后开始下一回合...")
                    time.sleep(2)
            
            # 打印总体统计
            print("\n===== 全部回合统计 =====")
            print(f"完成回合数: {completed_rounds}/{total_rounds}")
            print(f"成功到达目标回合数: {success_rounds}/{total_rounds}")
            print(f"成功率: {(success_rounds/total_rounds)*100:.1f}%")
            
        except KeyboardInterrupt:
            print("\n用户中断测试")
            print(f"已完成回合数: {completed_rounds}/{total_rounds}")
        finally:
            # 清理资源
            self.env.close()
            pygame.quit()
            
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"清理临时文件时出错: {e}")
            
            print("\n自动导航测试结束!")


def select_difficulty():
    """让用户选择迷宫难度"""
    print("请选择迷宫难度:")
    for key, desc in DIFFICULTY_DESCRIPTIONS.items():
        if key in ["1", "2", "3"]:
            continue
        number = {"easy": "1", "medium": "2", "hard": "3"}[key]
        print(f"{number}. {desc}")
    
    while True:
        choice = input("请输入选项 (1-3): ")
        if choice in DIFFICULTY_MAP:
            return DIFFICULTY_MAP[choice]
        else:
            print("无效的选择，请重新输入。")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="迷宫导航测试 - OpenAI模型")
    parser.add_argument("--difficulty", type=str, default=DEFAULT_DIFFICULTY, 
                       choices=["easy", "medium", "hard"], help="迷宫难度")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="测试轮数")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="每轮最大步数")
    parser.add_argument("--speed", type=float, default=DEFAULT_AUTO_SPEED, help="自动运行速度")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="结果保存目录")
    parser.add_argument("--seed", type=int, default=None, help="固定种子值")
    parser.add_argument("--no-sequential-seeds", action="store_true", help="禁用顺序种子模式")
    parser.add_argument("--interactive", action="store_true", help="交互式选择难度")
    
    args = parser.parse_args()
    
    # 交互式选择难度
    if args.interactive:
        difficulty = select_difficulty()
    else:
        difficulty = args.difficulty
    
    # 创建自动运行器并运行测试
    runner = ModelMazeRunner(
        difficulty=difficulty, 
        auto_speed=args.speed, 
        max_steps=args.max_steps,
        results_dir=args.results_dir,
        seed=args.seed,
        use_sequential_seeds=not args.no_sequential_seeds
    )
    runner.run(total_rounds=args.rounds)

if __name__ == "__main__":
    main()
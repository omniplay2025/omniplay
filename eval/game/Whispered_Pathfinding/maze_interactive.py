import gym
import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')  # 使用不需要图形界面的后端
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import tempfile
import os
from maze_gym_env import MazeGymEnv

class InteractiveMazeRunner:
    """交互式迷宫运行器，允许用户通过终端输入动作来控制角色"""
    
    def __init__(self, difficulty='easy'):
        # 初始化Pygame（用于音频播放）
        pygame.init()
        pygame.mixer.init()
        
        # 存储难度
        self.difficulty = difficulty
        
        # 创建环境，启用语音导航，并设置难度
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # 创建临时目录存储音频文件
        self.temp_dir = tempfile.mkdtemp(prefix="interactive_maze_")
        
        # 跟踪总奖励
        self.total_reward = 0
    
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
            
            # 等待音频播放完成
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"音频播放错误: {e}")
    
    def get_user_input(self):
        """获取用户输入的动作"""
        print("\n===== 请输入动作 =====")
        print("动作格式: [前进距离] [旋转角度]")
        print("前进距离范围: [-1.0, 3.0]，负值表示后退")
        print("旋转角度范围: [-180.0, 180.0]，负值表示向左转")
        print("示例: '1.0 45' 表示前进1.0个单位并向右旋转45度")
        print("输入'q'退出游戏")
        print("输入'r'重置游戏")
        print("输入'd'更改难度")
        
        while True:
            try:
                user_input = input("> ")
                
                # 检查特殊命令
                if user_input.lower() == 'q':
                    return None
                elif user_input.lower() == 'r':
                    return 'reset'
                elif user_input.lower() == 'd':
                    return 'change_difficulty'
                    
                # 解析输入
                values = user_input.split()
                if len(values) != 2:
                    print("错误: 请输入两个值，用空格分隔")
                    continue
                    
                forward = float(values[0])
                rotation = float(values[1])
                
                # 验证范围
                if not (-1.0 <= forward <= 3.0):
                    print("错误: 前进距离必须在[-1.0, 3.0]范围内")
                    continue
                    
                if not (-180.0 <= rotation <= 180.0):
                    print("错误: 旋转角度必须在[-180.0, 180.0]范围内")
                    continue
                    
                # 返回有效的动作
                return np.array([forward, rotation], dtype=np.float32)
                
            except ValueError:
                print("错误: 请输入有效的数值")
                continue
    
    def print_observation_info(self, observation):
        """打印观察信息"""
        vector_obs = observation['vector']
        
        print("\n===== 当前状态 =====")
        print(f"难度: {self.difficulty}")
        print(f"位置: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"朝向: {np.degrees(vector_obs[2]):.1f}°")
        print(f"到目标距离: {vector_obs[3]:.2f}米")
        print(f"到目标角度: {np.degrees(vector_obs[4]):.1f}°")
    
    def change_difficulty(self):
        """更改游戏难度"""
        print("\n选择新难度:")
        print("1. 简单 (Easy)")
        print("2. 中等 (Medium)")
        print("3. 困难 (Hard)")
        
        while True:
            choice = input("请选择 (1-3): ")
            if choice == "1":
                new_difficulty = "easy"
                break
            elif choice == "2":
                new_difficulty = "medium"
                break
            elif choice == "3":
                new_difficulty = "hard"
                break
            else:
                print("无效选择，请重新输入")
        
        # 更新难度并重置环境
        self.difficulty = new_difficulty
        observation = self.env.set_difficulty(new_difficulty)
        if observation:
            print(f"\n难度已更改为: {new_difficulty}")
            self.total_reward = 0
            return observation
        return None
    
    def run(self):
        """运行交互式迷宫环境"""
        print(f"欢迎来到交互式迷宫环境! 难度: {self.difficulty}")
        print("加载环境中...")
        
        # 重置环境
        observation, info = self.env.reset()
        
        # 显示初始状态信息
        self.print_observation_info(observation)
        
        # 播放初始音频提示
        if 'audio' in info:
            print("正在播放语音导航...")
            self.play_audio(info['audio'])
        
        # 交互式循环
        done = False
        while not done:
            # 获取用户输入的动作
            action = self.get_user_input()
            
            if action is None:  # 用户选择退出
                break
            elif action == 'reset':  # 用户选择重置
                observation, info = self.env.reset()
                self.total_reward = 0
                self.print_observation_info(observation)
                if 'audio' in info:
                    self.play_audio(info['audio'])
                continue
            elif action == 'change_difficulty':  # 用户选择更改难度
                new_observation = self.change_difficulty()
                if new_observation:
                    observation = new_observation
                    self.print_observation_info(observation)
                continue
                
            # 执行动作
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            
            # 显示当前环境
            self.env.render()
            
            # 打印信息
            self.print_observation_info(observation)
            print(f"执行动作: 前进={action[0]:.2f}, 旋转={action[1]:.2f}°")
            print(f"奖励: {reward:.2f}, 总奖励: {self.total_reward:.2f}")
            
            # 播放音频提示(如果有)
            if 'audio' in info:
                print("正在播放语音导航...")
                self.play_audio(info['audio'])
            
            # 检查是否结束
            done = terminated or truncated
            if terminated:
                print("\n恭喜! 你已到达目标!")
            elif truncated:
                print("\n回合结束!")
        
        # 清理资源
        self.env.close()
        pygame.quit()
        
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        
        print(f"\n游戏结束! 总奖励: {self.total_reward:.2f}")

def select_difficulty():
    """让用户选择迷宫难度"""
    print("请选择迷宫难度:")
    print("1. 简单 (Easy) - 墙壁较少，广阔路径")
    print("2. 中等 (Medium) - 适当障碍物，中等难度路径")
    print("3. 困难 (Hard) - 障碍物密集，更窄通道，更大迷宫")
    
    while True:
        choice = input("请输入选项 (1-3): ")
        if choice == "1":
            return "easy"
        elif choice == "2":
            return "medium"
        elif choice == "3":
            return "hard"
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    # 选择难度级别
    difficulty = select_difficulty()
    
    # 创建并运行交互式迷宫
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()
    # 创建并运行交互式迷宫
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()
    runner.run()
    # 创建并运行交互式迷宫
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()

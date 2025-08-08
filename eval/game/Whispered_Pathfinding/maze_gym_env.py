import gym
import numpy as np
import pygame
import math
import os
import random
import base64
from gym import spaces
from gtts import gTTS
import tempfile
from datetime import datetime
import threading
from maze_3d import MazeGame

class MazeGymEnv(gym.Env):
    """
    基于3D迷宫游戏的Gym环境
    
    动作空间:
    - 前进距离: [-1.0, 3.0]，负值表示后退，正值表示前进
    - 旋转角度: [-180.0, 180.0]度，负值表示向左旋转，正值表示向右旋转
    
    观察空间:
    - 玩家位置 (x, y)
    - 玩家朝向角度
    - 到目标的距离
    - 到目标的角度（相对于玩家朝向）
    - 8个方向的墙壁距离
    - 屏幕截图（RGB图像数组）
    - 音频数据 (base64编码的字符串，通过info返回)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode='rgb_array', voice_guidance=True, difficulty='easy'):
        super(MazeGymEnv, self).__init__()
        
        # 保存难度设置
        self.difficulty = difficulty
        
        # 定义动作空间：[前进距离, 旋转角度]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -180.0]), 
            high=np.array([3.0, 180.0]), 
            dtype=np.float32
        )
        
        # 定义观察空间为Dict类型，包含状态向量和屏幕图像
        self.observation_space = spaces.Dict({
            # 状态向量：[x位置, y位置, 角度, 到目标距离, 到目标角度, 8个方向的墙壁距离]
            'vector': spaces.Box(
                low=np.array([0, 0, -np.pi, 0, -np.pi] + [0] * 8),
                high=np.array([MazeGame.MAZE_SIZE, MazeGame.MAZE_SIZE, np.pi, 
                              np.sqrt(2) * MazeGame.MAZE_SIZE, np.pi] + [MazeGame.MAX_DEPTH] * 8),
                dtype=np.float32
            ),
            # 屏幕图像：RGB格式
            'screen': spaces.Box(
                low=0, 
                high=255, 
                shape=(MazeGame.SCREEN_HEIGHT, MazeGame.SCREEN_WIDTH, 3),
                dtype=np.uint8
            )
        })
        
        self.render_mode = render_mode
        self.game = None  # 将在reset中初始化
        self.window_surface = None
        
        # 语音提示相关设置
        self.voice_guidance = voice_guidance
        self.voice_temp_dir = tempfile.mkdtemp(prefix="maze_voice_")
        self.last_guidance_angle = None
        self.guidance_cooldown = 0  # 语音提示冷却时间
        self.last_audio_path = None  # 存储最近一次播放的音频文件路径
        
        # 缓存常用语音提示
        self.voice_cache = {}
        
    def encode_audio_vllm(self, audio_path):
        """
        将音频文件编码为base64字符串
        
        参数:
            audio_path: 音频文件路径
        
        返回:
            base64编码的字符串
        """
        if not audio_path or not os.path.exists(audio_path):
            return None
            
        try:
            with open(audio_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding audio: {e}")
            return None
    
    def _pregenerate_voice_guidance(self):
        """预先生成常用的语音提示并缓存"""
        # 不再预生成固定短语，因为现在每个语音提示都是基于当前位置动态生成的
        pass
        
    def _get_direction_guidance(self, relative_angle_deg, distance):
        """
        根据相对角度和距离生成精确的方向建议
        
        参数:
            relative_angle_deg: 目标相对于玩家朝向的角度（度）
            distance: 到目标的距离
        """
        # 将角度转换到(-180, 180]范围
        while relative_angle_deg > 180:
            relative_angle_deg -= 360
        while relative_angle_deg <= -180:
            relative_angle_deg += 360
        
        # 舍入角度和距离，使语音更自然
        rounded_angle = int(round(abs(relative_angle_deg) / 5) * 5)  # 舍入到最接近的5度
        rounded_distance = round(distance * 10) / 10  # 舍入到1位小数
            
        # 根据角度范围和距离生成提示
        if abs(relative_angle_deg) <= 15:
            # 几乎直线前方
            return f"The exit is {rounded_distance} meters straight ahead."
        elif abs(relative_angle_deg) <= 45:
            # 稍微偏左/右
            direction = "right" if relative_angle_deg > 0 else "left"
            return f"The exit is {rounded_distance} meters ahead, {rounded_angle} degrees to your {direction}."
        elif abs(relative_angle_deg) <= 135:
            # 大幅度向左/右转
            direction = "right" if relative_angle_deg > 0 else "left"
            return f"Turn {direction} about {rounded_angle} degrees. The exit is {rounded_distance} meters away."
        else:
            # 几乎在后方
            return f"The exit is {rounded_distance} meters behind you. Turn around."
        
    def _generate_voice_guidance(self, text, cache_only=False):
        """生成语音提示并播放"""
        if not self.voice_guidance:
            return
            
        # 检查是否已缓存
        if text in self.voice_cache:
            voice_file = self.voice_cache[text]
        else:
            # 创建唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            voice_file = os.path.join(self.voice_temp_dir, f"guidance_{timestamp}.wav")
            
            # 使用gTTS生成语音
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(voice_file)
                self.voice_cache[text] = voice_file
            except Exception as e:
                print(f"Error generating voice guidance: {e}")
                return
        
        # 保存最近一次的音频文件路径
        self.last_audio_path = voice_file
        
        # 如果只是缓存，不播放
        if cache_only:
            return
            
        # 在后台线程播放语音，避免阻塞主线程
        def play_sound():
            try:
                sound = pygame.mixer.Sound(voice_file)
                sound.play()
            except Exception as e:
                print(f"Error playing voice guidance: {e}")
                
        threading.Thread(target=play_sound).start()
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态并返回初始观察结果"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化或重置游戏，使用指定的难度
        if self.game is None:
            self.game = MazeGame(difficulty=self.difficulty)
            # 禁用原始游戏音效以避免干扰
            self.game.sound_go = None
        else:
            self.game.reset_game(difficulty=self.difficulty)
            
        # 重置语音提示状态
        self.last_guidance_angle = None
        self.guidance_cooldown = 0
        self.last_audio_path = None
        
        # 根据当前难度，更新环境观察空间中的向量上限
        current_maze_size = self.game.current_maze_size
        self.observation_space['vector'] = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, -np.pi] + [0] * 8),
            high=np.array([current_maze_size, current_maze_size, np.pi, 
                          np.sqrt(2) * current_maze_size, np.pi] + [MazeGame.MAX_DEPTH] * 8),
            dtype=np.float32
        )
        
        # 初始语音提示
        if self.voice_guidance:
            self._generate_voice_guidance(f"Welcome to the {self.difficulty} maze. Find the exit!")
        
        # 获取初始观察结果
        observation = self._get_observation()
        
        # 生成info字典，包含音频数据
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        执行动作并返回下一个状态、奖励、是否终止和额外信息
        
        参数:
            action: [前进距离, 旋转角度]
        """
        # 缓存当前位置用于计算奖励
        prev_pos = self.game.player_pos.copy()
        
        # 执行动作
        self._execute_action(action)
        
        # 获取新的观察结果
        observation = self._get_observation()
        
        # 检查是否到达目标
        dist_to_goal = np.linalg.norm(self.game.player_pos - self.game.goal_pos)
        # 根据难度设置不同的目标达成距离
        threshold = 1.0 if self.difficulty == 'hard' else 0.7
        terminated = dist_to_goal < threshold
        
        # 计算奖励
        reward = self._calculate_reward(dist_to_goal, prev_pos, terminated)
        
        # 检查是否碰到墙壁
        truncated = False  # 目前不考虑提前终止
        
        # 提供语音方向指导 - 现在每次step都会生成新提示
        self._provide_voice_guidance(observation['vector'])
        
        # 获取info字典，包含音频数据
        info = self._get_info()
        info['dist_to_goal'] = dist_to_goal
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self):
        """获取包含音频数据的info字典"""
        info = {}
        
        # 添加音频数据（如果有）
        audio_data = self.encode_audio_vllm(self.last_audio_path)
        if audio_data:
            info['audio'] = audio_data
            
        return info
    
    def _provide_voice_guidance(self, observation):
        """根据当前状态提供语音导航提示"""
        if not self.voice_guidance:
            return
            
        # 获取到目标的距离和角度（相对于玩家朝向）
        dist_to_goal = observation[3]
        relative_angle_rad = observation[4]
        relative_angle_deg = math.degrees(relative_angle_rad)
        
        # 移除冷却时间检查，确保每次都生成提示
        
        # 根据位置和方向生成语音提示
        if dist_to_goal < 2.0:
            # 非常接近目标时的特殊提示
            guidance_text = f"The exit is only {round(dist_to_goal, 1)} meters away."
        else:
            # 正常的方向指导
            guidance_text = self._get_direction_guidance(relative_angle_deg, dist_to_goal)
        
        # 生成新的语音提示
        self._generate_voice_guidance(guidance_text)
        
        # 更新最后一次提示的角度（仍然记录但不用于条件判断）
        self.last_guidance_angle = relative_angle_deg
        
        # 冷却时间设为0表示不使用冷却机制
        self.guidance_cooldown = 0

    def _execute_action(self, action):
        """
        执行给定的连续动作
        
        参数:
            action: [前进距离, 旋转角度]
        """
        move_distance, rotation_angle = action
        
        # 将角度从度转换为弧度
        rotation_radians = math.radians(rotation_angle)
        
        # 应用旋转
        self.game.player_angle += rotation_radians
        
        # 规范化角度到[-pi, pi]范围
        while self.game.player_angle > math.pi:
            self.game.player_angle -= 2 * math.pi
        while self.game.player_angle <= -math.pi:
            self.game.player_angle += 2 * math.pi
        
        # 计算移动方向
        dir_x = math.cos(self.game.player_angle)
        dir_y = math.sin(self.game.player_angle)
        
        # 应用移动（前进/后退）
        new_x = self.game.player_pos[0] + dir_x * move_distance * self.game.MOVE_SPEED * 10.0
        new_y = self.game.player_pos[1] + dir_y * move_distance * self.game.MOVE_SPEED * 10.0
        
        # 检查碰撞并更新位置
        if not self.game.is_wall(new_x, self.game.player_pos[1]):
            self.game.player_pos[0] = new_x
        if not self.game.is_wall(self.game.player_pos[0], new_y):
            self.game.player_pos[1] = new_y
    
    def _get_observation(self):
        """获取当前环境的观察向量和屏幕图像"""
        # 基本观察：玩家位置和角度
        x, y = self.game.player_pos
        angle = self.game.player_angle
        
        # 计算到目标的距离和角度
        goal_dir = self.game.goal_pos - self.game.player_pos
        distance_to_goal = np.linalg.norm(goal_dir)
        goal_angle = math.atan2(goal_dir[1], goal_dir[0])
        
        # 计算目标角度相对于玩家朝向的差值
        angle_diff = goal_angle - angle
        # 规范化角度到[-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 获取8个方向的墙壁距离
        wall_distances = []
        for i in range(8):
            ray_angle = angle + i * math.pi / 4  # 45度间隔，覆盖360度
            wall_dist, _, _ = self.game.cast_ray(ray_angle)
            wall_distances.append(wall_dist)
        
        # 组合观察向量
        vector_obs = np.array([x, y, angle, distance_to_goal, angle_diff] + wall_distances, 
                              dtype=np.float32)
        
        # 获取屏幕图像
        # 先渲染当前屏幕
        self.game.render_3d_view()
        self.game.render_mini_map()
        self.game.render_ui()
        
        # 获取屏幕截图作为RGB数组
        screen_image = pygame.surfarray.array3d(self.game.screen)
        # 调整数组形状为(高度, 宽度, 3)并转换为uint8类型
        screen_image = np.transpose(screen_image, (1, 0, 2)).astype(np.uint8)
        
        # 返回Dict形式的观察
        return {
            'vector': vector_obs,
            'screen': screen_image
        }
    
    def _calculate_reward(self, dist_to_goal, prev_pos, reached_goal):
        """
        计算奖励函数
        
        这里使用一个简单的奖励函数：
        - 到达目标：大奖励 +10
        - 接近目标：小奖励，与到目标的距离成反比
        - 远离或撞墙：小惩罚
        """
        if reached_goal:
            return 10.0  # 到达目标的奖励
        
        # 计算是否接近目标的奖励
        prev_dist = np.linalg.norm(self.game.goal_pos - prev_pos)
        reward = prev_dist - dist_to_goal  # 正值表示接近目标，负值表示远离
        
        # 添加一个基于距离的小奖励，鼓励接近目标
        distance_reward = 0.01 / (0.1 + dist_to_goal)
        
        return reward + distance_reward
    
    def render(self):
        """渲染当前游戏画面"""
        if self.render_mode == "human" and self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (MazeGame.SCREEN_WIDTH, MazeGame.SCREEN_HEIGHT)
            )
            
        # 使用游戏的渲染功能
        self.game.render_3d_view()
        self.game.render_mini_map()
        self.game.render_ui()
        
        if self.render_mode == "human":
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            # 将渲染结果转换为RGB数组
            return pygame.surfarray.array3d(
                self.game.screen
            ).swapaxes(0, 1)
    
    def close(self):
        """关闭环境和释放资源"""
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.window_surface = None
        if self.game is not None:
            self.game.running = False
            
        # 清理临时语音文件
        import shutil
        try:
            if os.path.exists(self.voice_temp_dir):
                shutil.rmtree(self.voice_temp_dir)
        except Exception as e:
            print(f"Error cleaning up voice files: {e}")
        # 清理临时语音文件
        import shutil
        try:
            if os.path.exists(self.voice_temp_dir):
                shutil.rmtree(self.voice_temp_dir)
        except Exception as e:
            print(f"Error cleaning up voice files: {e}")

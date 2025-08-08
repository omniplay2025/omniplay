import pygame
import random
import math
import time
import json
import base64
import os
import numpy as np
import gym
from gym import spaces
import io
from PIL import Image
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import sys
import importlib.util

# 添加游戏难度导入
from game_difficulty import DifficultyLevel, get_difficulty_config

# 修复文件名中的连字符问题
# 将文件重命名为更简单的名称
classic_bomberman_file = os.path.join(os.path.dirname(__file__), 'classic_bomberman-daiceshi.py')
module_name = "classic_bomberman_module"

# 直接导入模块
spec = importlib.util.spec_from_file_location(module_name, classic_bomberman_file)
game_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = game_module
spec.loader.exec_module(game_module)

# 从模块中获取所需的类和常量
Tile = game_module.Tile
Item = game_module.Item
Vector2 = game_module.Vector2
Player = game_module.Player
Bomb = game_module.Bomb
ItemDrop = game_module.ItemDrop
AssetManager = game_module.AssetManager
GameState = game_module.GameState
Renderer = game_module.Renderer
WINDOW_WIDTH = game_module.WINDOW_WIDTH
WINDOW_HEIGHT = game_module.WINDOW_HEIGHT
GRID_WIDTH = game_module.GRID_WIDTH
GRID_HEIGHT = game_module.GRID_HEIGHT
TILE_SIZE = game_module.TILE_SIZE
FPS = game_module.FPS
RED = game_module.RED
BLUE = game_module.BLUE
GREEN = game_module.GREEN
YELLOW = game_module.YELLOW
BLACK = game_module.BLACK
WHITE = game_module.WHITE

class BombermanAction(Enum):
    """动作枚举类型"""
    MOVE = 0      # 移动动作
    PLACE_BOMB = 1  # 放置炸弹动作

class BombermanEnv(gym.Env):
    """Bomberman Gym 环境"""
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': FPS
    }
    
    def __init__(self, render_mode=None, num_players=4, max_steps=1000, difficulty=DifficultyLevel.NORMAL):
        super().__init__()
        
        # 初始化 Pygame
        if not pygame.get_init():
            pygame.init()
        
        # 确保声音和字体子系统已初始化
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if not pygame.font.get_init():
            pygame.font.init()
    
        # 环境参数
        self.render_mode = render_mode
        self.num_players = num_players
        self.max_steps = max_steps
        self.current_step = 0
        self.difficulty = difficulty
        self.difficulty_config = get_difficulty_config(difficulty)
        
        # 更新基于难度的参数
        global GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
        GRID_WIDTH = self.difficulty_config.grid_width
        GRID_HEIGHT = self.difficulty_config.grid_height
        WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
        WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
        
        # 设置最大移动距离和炸弹倒计时
        self.max_move_distance = self.difficulty_config.max_move_distance
        self.bomb_countdown_steps = 3  # 炸弹爆炸倒计时（步数）
        
        # 创建游戏状态和渲染器
        self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.game_state = GameState(difficulty=difficulty)
        self.renderer = Renderer(self.screen, self.game_state)
        
        # 如果需要在窗口中显示
        if self.render_mode == 'human':
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(f"Bomberman Gym - {self.difficulty_config.name}模式")
        
        # 修改炸弹计时器为步数计时
        self._convert_bomb_timers_to_steps()
        
        # 定义动作空间和观察空间
        # 调整观察空间以适应不同大小的网格
        self.action_space = spaces.Dict({
            player_id: spaces.Dict({
                'action_type': spaces.Discrete(2),  # 0: 移动, 1: 放置炸弹
                'target_x': spaces.Discrete(GRID_WIDTH),  # 目标x坐标
                'target_y': spaces.Discrete(GRID_HEIGHT),  # 目标y坐标
            })
            for player_id in range(num_players)
        })
        
        # 观察空间: 游戏状态 + 图像 + 音频
        self.observation_space = spaces.Dict({
            'state': spaces.Dict({
                'grid': spaces.Box(low=0, high=4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.int8),
                'players': spaces.Dict({
                    player_id: spaces.Dict({
                        'position_x': spaces.Discrete(GRID_WIDTH),
                        'position_y': spaces.Discrete(GRID_HEIGHT),
                        'alive': spaces.Discrete(2),
                        'fire_power': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'bomb_count': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'speed': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'active_bombs': spaces.Box(low=0, high=8, shape=(), dtype=np.int8),
                        'trapped': spaces.Discrete(2),
                    })
                    for player_id in range(num_players)
                }),
                'bombs': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(20,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(20,), dtype=np.int8),
                    'countdown': spaces.Box(low=0, high=self.bomb_countdown_steps, shape=(20,), dtype=np.int8),
                    'owner': spaces.Box(low=0, high=num_players-1, shape=(20,), dtype=np.int8),
                    'fire_power': spaces.Box(low=1, high=8, shape=(20,), dtype=np.int8),
                    'count': spaces.Discrete(21),  # 最多20个炸弹
                }),
                'items': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(50,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(50,), dtype=np.int8),
                    'types': spaces.Box(low=0, high=2, shape=(50,), dtype=np.int8),
                    'count': spaces.Discrete(51),  # 最多50个道具
                }),
                'flames': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(100,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(100,), dtype=np.int8),
                    'count': spaces.Discrete(101),  # 最多100个火焰格子
                }),
                'game_over': spaces.Discrete(2),
            }),
            'image': spaces.Text(max_length=1000000),  # Base64 编码图像
            'audio': spaces.Text(max_length=1000000),  # Base64 编码音频
            'step': spaces.Box(low=0, high=max_steps, shape=(), dtype=np.int32),
        })
        
        # 用于存储临时文件的目录
        self.temp_dir = "/tmp/bomberman_gym"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 音频事件记录 - 修改为更详细的格式，包含玩家信息
        self.audio_events = []  # 格式: (事件类型, 玩家ID, 其他参数)
        
        # 添加事件跟踪
        self.kill_events = []  # 记录击杀事件
        self.item_pickup_events = []  # 记录道具拾取事件
        self.invalid_actions_count = {i: 0 for i in range(num_players)}
    
    def _convert_bomb_timers_to_steps(self):
        """将炸弹计时器转换为步数计时"""
        for bomb in self.game_state.bombs:
            # 将帧计时转换为步数计时
            bomb.timer = self.bomb_countdown_steps
    
    def reset(self, seed=None, options=None, difficulty=None):
        """重置环境，可选择新的难度"""
        super().reset(seed=seed)
        
        # 如果提供了新的难度，更新难度设置
        if difficulty is not None:
            self.difficulty = difficulty
            self.difficulty_config = get_difficulty_config(difficulty)
            
            # 更新全局变量
            global GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
            GRID_WIDTH = self.difficulty_config.grid_width
            GRID_HEIGHT = self.difficulty_config.grid_height
            WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
            WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
            
            # 更新移动距离
            self.max_move_distance = self.difficulty_config.max_move_distance
            
            # 重新创建屏幕和窗口
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            if self.render_mode == 'human':
                self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption(f"Bomberman Gym - {self.difficulty_config.name}模式")
        
        self.current_step = 0
        self.game_state.reset(difficulty=self.difficulty)
        self.renderer = Renderer(self.screen, self.game_state)
        self._convert_bomb_timers_to_steps()
        self.audio_events = []
        
        # 重置事件跟踪
        self.kill_events = []
        self.item_pickup_events = []
        
        # 重置无效动作计数
        self.invalid_actions_count = {i: 0 for i in range(self.num_players)}
        
        # 返回初始观察
        obs = self._get_observation()
        info = {"difficulty": self.difficulty.value}
        
        return obs, info
    
    def step(self, actions):
        """执行一步动作"""
        self.current_step += 1
        self.audio_events = []  # 清空上一步的音频事件
        
        # 处理每个玩家的动作
        for player_id, action in actions.items():
            if player_id >= len(self.game_state.players):
                continue
                
            player = self.game_state.players[player_id]
            if not player.alive:
                continue
            
            try:
                # 尝试获取动作参数
                action_type = action.get('action_type')
                target_x = action.get('target_x')
                target_y = action.get('target_y')
                
                # 检查是否缺少必要的参数
                if action_type is None or (action_type == BombermanAction.MOVE.value and (target_x is None or target_y is None)):
                    # 记录无效动作
                    self.invalid_actions_count[player_id] += 1
                    print(f"警告: 玩家 {player_id} 提供了无效动作: {action}. 应用默认动作(静止). 无效动作总数: {self.invalid_actions_count[player_id]}")
                    
                    # 提供默认动作 - 玩家保持静止
                    action_type = BombermanAction.MOVE.value
                    target_x = player.pos.x
                    target_y = player.pos.y
                
                # 执行动作
                if action_type == BombermanAction.MOVE.value:
                    self._handle_move_action(player, target_x, target_y)
                elif action_type == BombermanAction.PLACE_BOMB.value:
                    self._handle_place_bomb_action(player)
            except Exception as e:
                # 捕获所有异常，确保即使动作处理出错，游戏也能继续
                self.invalid_actions_count[player_id] += 1
                print(f"错误: 处理玩家 {player_id} 的动作时出错: {e}. 应用默认动作(静止). 无效动作总数: {self.invalid_actions_count[player_id]}")
                # 游戏继续，玩家保持静止
        
        # 在更新游戏状态前清除事件记录
        self.kill_events = []
        self.item_pickup_events = []
        
        # 更新游戏状态时，传递事件回调
        self.game_state.update_bombs = self.game_state.update_bombs_steps
        self.game_state.register_explosion_callback = self.register_explosion_event
        self.game_state.register_kill_callback = self.register_kill_event
        self.game_state.register_item_pickup_callback = self.register_item_pickup_event
        self.game_state.update()
        
        # 获取奖励、观察、结束状态和信息
        obs = self._get_observation()
        rewards = self._get_rewards()
        terminated = self.game_state.game_over
        truncated = self.current_step >= self.max_steps
        info = {
            "invalid_actions": self.invalid_actions_count.copy(),
            "kill_events": self.kill_events.copy(),
            "item_pickup_events": self.item_pickup_events.copy()
        }
        
        # 如果需要在窗口中显示
        if self.render_mode == 'human':
            self.render()
        
        return obs, rewards, terminated, truncated, info
    
    def register_explosion_event(self, bomb, affected_positions):
        """注册爆炸音效事件"""
        self.audio_events.append(('bomb_explode', bomb.owner_id, {
            'pos': (bomb.grid_pos.x, bomb.grid_pos.y),
            'fire': bomb.fire,
            'affected_positions': list(affected_positions)
        }))
        
        # 确保爆炸后火焰持续时间正确（使用步数而不是帧数）
        for pos in affected_positions:
            self.game_state.flames[pos] = 2  # 火焰持续2步
    
    def register_kill_event(self, killer_id, victim_id):
        """注册击杀事件"""
        self.kill_events.append({
            'killer': killer_id,
            'victim': victim_id
        })
        
    def register_item_pickup_event(self, player_id, item_type):
        """注册道具拾取事件"""
        self.item_pickup_events.append({
            'player': player_id,
            'item_type': item_type.value
        })
    
    def _handle_move_action(self, player, target_x, target_y):
        """处理移动动作"""
        # 检查移动距离是否在允许范围内
        current_x, current_y = player.pos.x, player.pos.y
        distance = abs(target_x - current_x) + abs(target_y - current_y)  # 曼哈顿距离
        max_distance = self.max_move_distance + (player.speed - 1)  # 基础移动距离 + 速度加成
        
        if distance > max_distance:
            # 超出移动范围，截断到最远允许距离
            if target_x > current_x:
                dx = min(target_x - current_x, max_distance)
            else:
                dx = max(target_x - current_x, -max_distance)
                
            # 调整y方向的移动距离
            remaining_distance = max_distance - abs(dx)
            if target_y > current_y:
                dy = min(target_y - current_y, remaining_distance)
            else:
                dy = max(target_y - current_y, -remaining_distance)
                
            target_x = current_x + dx
            target_y = current_y + dy
        
        # 检查目标位置是否有效
        path = self._find_path(current_x, current_y, target_x, target_y, max_distance)
        if path:
            # 移动到路径的最远有效点
            final_pos = path[-1]
            player.pos = Vector2(final_pos[0], final_pos[1])
            
            # 检查道具拾取
            self.game_state.check_item_pickup(player)
            
            # 记录移动音效事件，包含玩家ID
            self.audio_events.append(('player_walk', player.player_id, {
                'from_pos': (current_x, current_y),
                'to_pos': (final_pos[0], final_pos[1])
            }))
    
    def _find_path(self, start_x, start_y, target_x, target_y, max_steps):
        """找到从起点到目标点的路径，如果无法到达，返回最远可达点"""
        # 简化的A*寻路算法
        open_set = [(start_x, start_y)]
        closed_set = set()
        g_score = {(start_x, start_y): 0}
        came_from = {}
        
        while open_set:
            current = min(open_set, key=lambda pos: g_score[pos] + abs(pos[0] - target_x) + abs(pos[1] - target_y))
            
            if current == (target_x, target_y) or g_score[current] >= max_steps:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 检查四个方向
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if (neighbor[0] < 0 or neighbor[0] >= GRID_WIDTH or 
                    neighbor[1] < 0 or neighbor[1] >= GRID_HEIGHT):
                    continue
                
                # 检查障碍物
                if self.game_state.grid[neighbor[1]][neighbor[0]] in [Tile.HARD, Tile.SOFT, Tile.BOMB]:
                    continue
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
        
        # 如果没有找到路径，返回离目标最近的点
        if came_from:
            best_pos = max(closed_set, key=lambda pos: g_score[pos])
            path = [best_pos]
            while best_pos in came_from:
                best_pos = came_from[best_pos]
                path.append(best_pos)
            path.reverse()
            return path
        
        # 完全无法移动
        return [(start_x, start_y)]
    
    def _handle_place_bomb_action(self, player):
        """处理放置炸弹动作"""
        # 检查炸弹数量限制
        if player.active_bombs >= player.bombs:
            return
        
        # 检查当前位置是否已有炸弹
        bomb_pos = player.pos
        for bomb in self.game_state.bombs:
            if bomb.grid_pos.x == bomb_pos.x and bomb.grid_pos.y == bomb_pos.y:
                return
        
        # 放置炸弹
        bomb = Bomb(
            owner_id=player.player_id,
            grid_pos=Vector2(bomb_pos.x, bomb_pos.y),
            fire=player.fire,
            timer=self.bomb_countdown_steps  # 使用步数计时
        )
        self.game_state.bombs.append(bomb)
        self.game_state.grid[bomb_pos.y][bomb_pos.x] = Tile.BOMB
        player.active_bombs += 1
        
        # 记录放置炸弹音效事件，包含玩家ID
        self.audio_events.append(('bomb_place', player.player_id, {
            'pos': (bomb_pos.x, bomb_pos.y),
            'fire': player.fire
        }))
    
    def render(self):
        """渲染游戏画面"""
        self.renderer.render()
        
        if self.render_mode == 'human':
            # 复制到窗口显示
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            pygame.event.pump()  # 处理事件队列，防止程序无响应
        
        return self.screen
    
    def _get_observation(self):
        """获取观察状态"""
        # 游戏状态
        state = self._get_game_state()
        
        # 游戏图像
        image_base64 = self._get_image_base64()
        
        # 游戏音频
        audio_base64 = self._get_audio_base64()
        
        return {
            'state': state,
            'image': image_base64,
            'audio': audio_base64,
            'step': self.current_step
        }
    
    def _get_game_state(self):
        """获取游戏状态的结构化表示"""
        # 地图网格
        grid = np.array([[tile.value for tile in row] for row in self.game_state.grid], dtype=np.int8)
        
        # 玩家信息
        players_info = {}
        for i, player in enumerate(self.game_state.players):
            if i >= self.num_players:
                break
            players_info[i] = {
                'position_x': player.pos.x,
                'position_y': player.pos.y,
                'alive': int(player.alive),
                'fire_power': player.fire,
                'bomb_count': player.bombs,
                'speed': player.speed,
                'active_bombs': player.active_bombs,
                'trapped': int(player.trapped_ticks > 0),
            }
        
        # 炸弹信息
        bomb_positions_x = np.zeros(20, dtype=np.int8)
        bomb_positions_y = np.zeros(20, dtype=np.int8)
        bomb_countdown = np.zeros(20, dtype=np.int8)
        bomb_owner = np.zeros(20, dtype=np.int8)
        bomb_fire_power = np.zeros(20, dtype=np.int8)
        
        for i, bomb in enumerate(self.game_state.bombs[:20]):
            bomb_positions_x[i] = bomb.grid_pos.x
            bomb_positions_y[i] = bomb.grid_pos.y
            bomb_countdown[i] = bomb.timer
            bomb_owner[i] = bomb.owner_id
            bomb_fire_power[i] = bomb.fire
        
        # 道具信息
        item_positions_x = np.zeros(50, dtype=np.int8)
        item_positions_y = np.zeros(50, dtype=np.int8)
        item_types = np.zeros(50, dtype=np.int8)
        
        for i, item in enumerate(self.game_state.items[:50]):
            item_positions_x[i] = item.grid_pos.x
            item_positions_y[i] = item.grid_pos.y
            item_types[i] = item.item_type.value
        
        # 火焰信息
        flame_positions_x = np.zeros(100, dtype=np.int8)
        flame_positions_y = np.zeros(100, dtype=np.int8)
        
        # 适应新的火焰数据结构
        for i, (x, y) in enumerate(list(self.game_state.flames.keys())[:100]):
            flame_positions_x[i] = x
            flame_positions_y[i] = y
        
        return {
            'grid': grid,
            'players': players_info,
            'bombs': {
                'positions_x': bomb_positions_x,
                'positions_y': bomb_positions_y,
                'countdown': bomb_countdown,
                'owner': bomb_owner,
                'fire_power': bomb_fire_power,
                'count': len(self.game_state.bombs),
            },
            'items': {
                'positions_x': item_positions_x,
                'positions_y': item_positions_y,
                'types': item_types,
                'count': len(self.game_state.items),
            },
            'flames': {
                'positions_x': flame_positions_x,
                'positions_y': flame_positions_y,
                'count': len(self.game_state.flames),
            },
            'game_over': int(self.game_state.game_over),
        }
    
    def _get_image_base64(self):
        """获取游戏画面的base64编码"""
        # 渲染到surface
        self.renderer.render()
        
        # 将pygame surface转换为PIL Image
        image_str = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), image_str)
        
        # 保存为临时文件
        temp_image_path = os.path.join(self.temp_dir, f"frame_{self.current_step}.png")
        image.save(temp_image_path)
        
        # 编码为base64
        return self.encode_image(temp_image_path)
    
    def _get_audio_base64(self):
        """获取游戏音频的base64编码"""
        # 返回所有音频事件，而不仅仅是第一个
        if not self.audio_events:
            return ""
        
        audio_data = []
        asset_manager = self.game_state.asset_manager
        
        for event_type, player_id, params in self.audio_events:
            sound = asset_manager.get_sound(event_type)
            
            # Map event types to specific file names - 更新为WAV文件
            event_file_mapping = {
                'player_walk': 'footstep_wood_001.wav',
                'bomb_place': 'click-b.wav',
                'bomb_explode': 'explosion1.wav'
            }
            
            # Use the mapping to find the specific audio file
            audio_path = None
            if event_type in event_file_mapping:
                target_file = event_file_mapping[event_type]
                for path in asset_manager.asset_paths:
                    if path.endswith(target_file):
                        if os.path.exists(path):
                            audio_path = path
                            break
                
                if audio_path:
                    # 创建包含事件信息的数据结构
                    player_name = f"Player {player_id + 1}"
                    event_description = f"{player_name} - {event_type}"
                    
                    # 根据事件类型添加更多的描述
                    if event_type == 'player_walk':
                        from_pos = params['from_pos']
                        to_pos = params['to_pos']
                        event_description += f" from ({from_pos[0]},{from_pos[1]}) to ({to_pos[0]},{to_pos[1]})"
                    elif event_type == 'bomb_place':
                        pos = params['pos']
                        fire = params['fire']
                        event_description += f" at ({pos[0]},{pos[1]}) with fire power {fire}"
                    elif event_type == 'bomb_explode':
                        pos = params['pos']
                        fire = params['fire']
                        affected_positions = params['affected_positions']
                        affected_desc = ", ".join([f"({x},{y})" for x, y in affected_positions])
                        event_description += f" at ({pos[0]},{pos[1]}) with fire power {fire}, affected: {affected_desc}"
                    
                    # 编码音频
                    encoded_audio = self.encode_audio_vllm(audio_path)
                    
                    # 添加到结果列表
                    audio_data.append({
                        'event_type': event_type,
                        'player_id': player_id,
                        'description': event_description,
                        'audio_base64': encoded_audio,
                        'params': params
                    })
        
        # 将结果转换为JSON字符串返回
        return json.dumps(audio_data)
    
    def encode_audio_vllm(self, audio_path):
        """将音频编码为base64格式"""
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def encode_image(self, image_path):
        """将图像编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_rewards(self):
        """获取每个玩家的奖励"""
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # 这里可以实现复杂的奖励计算逻辑
        # 例如：摧毁墙壁、收集道具、击败敌人等
        
        # 简单的奖励：存活奖励
        for i, player in enumerate(self.game_state.players):
            if i >= self.num_players:
                break
                
            if player.alive:
                rewards[i] += 0.1  # 存活奖励
        
        # 胜利奖励
        if self.game_state.game_over and self.game_state.winner:
            winner_id = self.game_state.winner.player_id
            if winner_id < self.num_players:
                rewards[winner_id] += 10.0  # 胜利大奖励
        
        return rewards
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
            pygame.quit()

# 示例用法
if __name__ == "__main__":
    # 添加难度选择菜单
    print("请选择游戏难度:")
    print("1. 简单 (小地图，少障碍物)")
    print("2. 普通 (标准地图)")
    
    choice = input("请输入选择 (1/2, 默认2): ")
    difficulty = DifficultyLevel.NORMAL
    
    if choice == '1':
        difficulty = DifficultyLevel.EASY
        print("选择了简单难度 - 小地图，少障碍物")
    else:
        print("选择了普通难度 - 标准地图")
    
    env = BombermanEnv(render_mode='human', difficulty=difficulty)
    obs, info = env.reset()
    
    for _ in range(1000):
        # 随机动作
        actions = {}
        for player_id in range(env.num_players):
            if random.random() < 0.2:  # 20%概率放置炸弹
                actions[player_id] = {
                    'action_type': BombermanAction.PLACE_BOMB.value,
                    'target_x': 0,  # 放炸弹不需要目标坐标
                    'target_y': 0
                }
            else:  # 80%概率移动
                actions[player_id] = {
                    'action_type': BombermanAction.MOVE.value,
                    'target_x': random.randint(0, GRID_WIDTH-1),
                    'target_y': random.randint(0, GRID_HEIGHT-1)
                }
        
        # 执行动作并获取结果
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 如果游戏结束，重置环境
        if terminated or truncated:
            print("游戏结束，重置环境")
            obs, info = env.reset()
            # 等待一小段时间以便观察
            time.sleep(0.5)
        
        # 添加一点延迟，使演示更容易观察
        time.sleep(0.1)
    
    # 关闭环境
    env.close()

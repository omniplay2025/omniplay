import pygame
import numpy as np
import os
import json
import math
import random
from pygame.locals import *

class MazeGame:
    # 将游戏配置常量移至类级别，便于在Gym环境中访问
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    MAZE_SIZE = 8  # 基础迷宫大小，在hard难度下可能增加
    WALL_HEIGHT = 2.0
    FOV = math.pi / 3
    MAX_DEPTH = 16
    MOVE_SPEED = 0.08
    TURN_SPEED = math.pi / 30
    RAY_COUNT = 160
    
    # 难度设置
    DIFFICULTY_SETTINGS = {
        'easy': {
            'wall_remove_prob': 0.3,  # 移除墙壁的概率
            'additional_obstacles': 0,  # 额外添加的障碍物数量
            'narrow_path_factor': 0.0,  # 通道变窄因子(0不变窄)
            'dead_ends': 0,            # 死胡同数量
            'maze_size_increase': 0    # 迷宫尺寸增加量
        },
        'medium': {
            'wall_remove_prob': 0.05,
            'additional_obstacles': 12,
            'narrow_path_factor': 0.0,
            'dead_ends': 2,
            'maze_size_increase': 0
        },
        'hard': {
            'wall_remove_prob': 0.05,
            'additional_obstacles': 18,
            'narrow_path_factor': 0.25,  # 通道宽度减小25%
            'dead_ends': 5,
            'maze_size_increase': 2      # 迷宫尺寸增加2，即10x10
        }
    }
    
    def __init__(self, difficulty='easy'):
        # 初始化Pygame
        pygame.init()
        pygame.mixer.init()
        
        # 设置难度
        self.difficulty = difficulty if difficulty in self.DIFFICULTY_SETTINGS else 'easy'
        self.difficulty_config = self.DIFFICULTY_SETTINGS[self.difficulty]
        
        # 根据难度调整迷宫大小
        self.current_maze_size = self.MAZE_SIZE + self.difficulty_config['maze_size_increase']
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(f"3D Maze Adventure - {self.difficulty.capitalize()}")
        self.clock = pygame.time.Clock()
        
        # 生成迷宫
        self.maze = self.generate_simplified_maze(self.current_maze_size)
        
        # 玩家属性
        self.player_pos = np.array([1.5, 1.5])
        self.player_angle = 0
        self.player_fov = self.FOV
        
        # 设置出口位置
        self.goal_pos = np.array([self.current_maze_size - 1.5, self.current_maze_size - 1.5])
        
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 加载音效
        try:
            sound_path = os.path.join(script_dir, '..', '..', 'assets-necessary', 'kenney', 'Audio', 'Voiceover Pack', 'Audio (Female)', 'go.ogg')
            if os.path.exists(sound_path):
                self.sound_go = pygame.mixer.Sound(sound_path)
            else:
                self.sound_go = None
        except:
            self.sound_go = None
            
        # 加载目标图标纹理
        try:
            texture_path = os.path.join(script_dir, '..', '..', 'assets-necessary', 'kenney', '3D assets', 'Holiday Kit', 'Previews', 'present-a-cube.png')
            if os.path.exists(texture_path):
                self.goal_texture = pygame.image.load(texture_path)
            else:
                self.goal_texture = None
        except:
            self.goal_texture = None
        
        # 确保目标纹理正确缩放if self.goal_texture:
            goal_size = 128
            self.goal_texture = pygame.transform.scale(self.goal_texture, (goal_size, goal_size))
            if hasattr(self.goal_texture, 'convert_alpha'):
                self.goal_texture = self.goal_texture.convert_alpha()
        
        # 游戏状态
        self.running = True
        self.won = False
        self.use_lighting = True
        self.show_minimap = True
        self.show_path_hints = True
        
        # 帧计数器
        self.frame_counter = 0
        self.fps_values = []
        self.last_fps_time = pygame.time.get_ticks()
        self.current_fps = 0
        
        # 小地图缓存
        self.minimap_surface = None
        self.minimap_update_counter = 0
    
    def load_sound(self, name):
        """根据名称查找并加载声音"""
        matching_paths = [p for p in self.asset_paths if name in p]
        if matching_paths:
            try:
                return pygame.mixer.Sound(matching_paths[0])
            except:
                pass
        return None
    
    def generate_simplified_maze(self, size):
        """生成简化的迷宫，根据难度调整"""
        # 创建填满墙的迷宫 (1是墙, 0是路)
        maze = np.ones((size, size), dtype=np.int8)
        
        # 从(1,1)开始生成
        def carve_path(x, y):
            maze[y][x] = 0  # 设为路
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < size-1 and 0 < ny < size-1 and maze[ny][nx] == 1:
                    maze[y + dy//2][x + dx//2] = 0
                    carve_path(nx, ny)
        
        carve_path(1, 1)
        
        # 确保入口和出口是路
        maze[1][1] = 0
        maze[size-2][size-2] = 0
        
        # 根据难度随机移除墙壁
        wall_remove_prob = self.difficulty_config['wall_remove_prob']
        for y in range(1, size-1):
            for x in range(1, size-1):
                if maze[y][x] == 1 and random.random() < wall_remove_prob:
                    path_count = sum(maze[y+dy][x+dx] == 0 
                                     for dy in [-1, 0, 1] 
                                     for dx in [-1, 0, 1]
                                     if 0 <= x+dx < size and 0 <= y+dy < size)
                    if path_count >= 1:
                        maze[y][x] = 0
        
        # 创建从起点到终点的直接路径
        self.create_path_to_exit(maze, 1, 1, size-2, size-2)
        
        # 根据难度添加额外障碍物
        self.add_additional_obstacles(maze, size)
        
        return maze
    
    def add_additional_obstacles(self, maze, size):
        """根据难度添加额外的障碍物"""
        num_obstacles = self.difficulty_config['additional_obstacles']
        if num_obstacles <= 0:
            return
        
        # 如果是困难模式，使用高级的迷宫障碍物生成
        if self.difficulty == 'hard':
            self.create_maze_obstacles(maze, size, num_obstacles)
            return
            
        # 普通困难度的障碍物添加（原有逻辑）
        empty_cells = []
        for y in range(1, size-1):
            for x in range(1, size-1):
                # 跳过起点和终点附近
                if (abs(x - 1) + abs(y - 1) <= 2) or (abs(x - (size-2)) + abs(y - (size-2)) <= 2):
                    continue
                    
                if maze[y][x] == 0:  # 如果是路
                    empty_cells.append((x, y))
        
        # 随机选择位置添加障碍物
        random.shuffle(empty_cells)
        for i in range(min(num_obstacles, len(empty_cells))):
            x, y = empty_cells[i]
            # 检查放置障碍物后是否仍然有路径可达
            maze[y][x] = 1  # 临时设为墙
            
            # 确保放置障碍物后仍有路径从起点到终点（使用简单的连通性检查）
            if not self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                maze[y][x] = 0  # 如果阻塞了路径，则撤销障碍物

    def create_maze_obstacles(self, maze, size, num_obstacles):
        """创建迷宫风格的障碍物布局，确保存在唯一通路"""
        # 步骤1: 找出从起点到终点的一条路径（这将是我们保证存在的路径）
        path = self.find_shortest_path(maze, (1, 1), (size-2, size-2))
        if not path:
            return  # 如果找不到路径，不添加障碍物
        
        # 步骤2: 标记这条路径上的所有点及其周围一格，这些点不能放置障碍物
        protected_cells = set()
        for x, y in path:
            protected_cells.add((x, y))
            # 添加周围一格的保护区
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        protected_cells.add((nx, ny))
        
        # 步骤3: 识别适合放置障碍物的点，即不在保护区内的空白点
        candidate_cells = []
        for y in range(1, size-1):
            for x in range(1, size-1):
                # 跳过起点和终点附近
                if (abs(x - 1) + abs(y - 1) <= 2) or (abs(x - (size-2)) + abs(y - (size-2)) <= 2):
                    continue
                
                if maze[y][x] == 0 and (x, y) not in protected_cells:
                    # 检查周围是否有足够的空间放置连续障碍物
                    neighbor_spaces = []
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        dx, dy = direction
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < size and 0 <= ny < size and 
                            maze[ny][nx] == 0 and (nx, ny) not in protected_cells):
                            neighbor_spaces.append((nx, ny))
                    
                    # 只选择有至少一个可用邻居的位置作为起始点
                    if neighbor_spaces:
                        candidate_cells.append((x, y, neighbor_spaces))
        
        # 步骤4: 随机选择起始点并生成连续的障碍物模式
        obstacles_added = 0
        random.shuffle(candidate_cells)
        
        while candidate_cells and obstacles_added < num_obstacles:
            x, y, neighbors = candidate_cells.pop(0)
            
            # 尝试创建一段连续的墙（1-3格长）
            obstacle_points = [(x, y)]
            maze[y][x] = 1  # 设置第一个障碍物
            obstacles_added += 1
            
            # 随机选择一个方向，尝试扩展障碍物
            if neighbors and obstacles_added < num_obstacles:
                next_x, next_y = random.choice(neighbors)
                # 确保这个点仍然是可用的
                if maze[next_y][next_x] == 0 and (next_x, next_y) not in protected_cells:
                    obstacle_points.append((next_x, next_y))
                    maze[next_y][next_x] = 1
                    obstacles_added += 1
                    
                    # 可能继续在同一方向扩展
                    dx, dy = next_x - x, next_y - y
                    third_x, third_y = next_x + dx, next_y + dy
                    if (0 <= third_x < size and 0 <= third_y < size and 
                        maze[third_y][third_x] == 0 and 
                        (third_x, third_y) not in protected_cells and
                        obstacles_added < num_obstacles):
                        obstacle_points.append((third_x, third_y))
                        maze[third_y][third_x] = 1
                        obstacles_added += 1
            
            # 验证添加障碍物后仍存在通路
            if not self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                # 如果没有通路，则移除这些障碍物
                for ox, oy in obstacle_points:
                    maze[oy][ox] = 0
                obstacles_added -= len(obstacle_points)
                continue
                
            # 更新候选点列表，移除现在不再适合的点
            candidate_cells = [cell for cell in candidate_cells 
                               if maze[cell[1]][cell[0]] == 0 and 
                               (cell[0], cell[1]) not in protected_cells]

    def is_wall(self, x, y):
        """检查位置是否是墙"""
        grid_x, grid_y = int(x), int(y)
        if 0 <= grid_x < self.MAZE_SIZE and 0 <= grid_y < self.MAZE_SIZE:
            return self.maze[grid_y][grid_x] == 1
        return True
    
    def cast_ray(self, angle):
        """从玩家位置向特定角度投射光线"""
        # 视线方向向量
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        
        # 玩家位置
        pos_x, pos_y = self.player_pos
        map_x, map_y = int(pos_x), int(pos_y)
        
        # DDA算法参数
        delta_dist_x = abs(1 / dir_x) if dir_x != 0 else float('inf')
        delta_dist_y = abs(1 / dir_y) if dir_y != 0 else float('inf')
        
        if dir_x < 0:
            step_x = -1
            side_dist_x = (pos_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - pos_x) * delta_dist_x
            
        if dir_y < 0:
            step_y = -1
            side_dist_y = (pos_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - pos_y) * delta_dist_y
            
        # 执行DDA算法
        hit = False
        side = 0  # 0表示x侧面，1表示y侧面
        
        while not hit and (abs(map_x - pos_x) < self.MAX_DEPTH or abs(map_y - pos_y) < self.MAX_DEPTH):
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            
            if 0 <= map_x < self.MAZE_SIZE and 0 <= map_y < self.MAZE_SIZE:
                if self.maze[map_y][map_x] == 1:
                    hit = True
            else:
                hit = True
        
        # 计算精确距离
        if side == 0:
            wall_dist = (map_x - pos_x + (1 - step_x) / 2) / (dir_x if abs(dir_x) > 1e-6 else 1e-6)
        else:
            wall_dist = (map_y - pos_y + (1 - step_y) / 2) / (dir_y if abs(dir_y) > 1e-6 else 1e-6)
            
        # 确保wall_dist始终为正值且有合理上限
        wall_dist = max(0.1, min(wall_dist, self.MAX_DEPTH))
        
        return wall_dist, side, (map_x, map_y)
    
    def render_3d_view(self):
        """渲染3D视图"""
        # 清除屏幕
        self.screen.fill((0, 0, 0))
        
        # 绘制天空和地面
        sky_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT//2)
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT//2, self.SCREEN_WIDTH, self.SCREEN_HEIGHT//2)
        pygame.draw.rect(self.screen, (100, 150, 200), sky_rect)
        pygame.draw.rect(self.screen, (50, 100, 50), ground_rect)
        
        # 保存z-buffer用于目标指示器的遮挡检测
        z_buffer = [float('inf')] * self.SCREEN_WIDTH
        
        # 光线投射参数
        ray_step = self.SCREEN_WIDTH / self.RAY_COUNT
        wall_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # 投射光线
        for i in range(self.RAY_COUNT):
            x = int(i * ray_step)
            
            # 计算光线方向
            camera_x = 2 * x / self.SCREEN_WIDTH - 1
            ray_angle = self.player_angle + camera_x * self.player_fov / 2
            
            # 投射光线
            wall_dist, side, wall_pos = self.cast_ray(ray_angle)
            
            # 更新z-buffer
            end_x = int((i + 1) * ray_step) if i < self.RAY_COUNT - 1 else self.SCREEN_WIDTH
            for fill_x in range(x, end_x):
                z_buffer[min(fill_x, self.SCREEN_WIDTH - 1)] = wall_dist
            
            # 计算墙的高度
            line_height = int(self.SCREEN_HEIGHT / wall_dist * self.WALL_HEIGHT)
            line_height = min(line_height, self.SCREEN_HEIGHT * 3)
                
            # 计算绘制位置
            draw_start = max(0, -line_height // 2 + self.SCREEN_HEIGHT // 2)
            draw_end = min(self.SCREEN_HEIGHT - 1, line_height // 2 + self.SCREEN_HEIGHT // 2)
            
            # 使用纯色绘制墙壁
            strip_width = int(ray_step) + 1
            
            # 根据墙面类型选择颜色
            base_color = (160, 160, 160) if side == 0 else (130, 130, 130)
            
            # 应用光照
            if self.use_lighting:
                shade = max(0.3, min(1.0, 1.0 - wall_dist/self.MAX_DEPTH))
                color = tuple(int(c * shade) for c in base_color)
            else:
                color = base_color
            
            # 绘制墙
            pygame.draw.rect(wall_surface, color, (x, draw_start, strip_width, draw_end - draw_start))
        
        # 渲染墙面
        self.screen.blit(wall_surface, (0, 0))
        
        # 更新帧计数器
        self.frame_counter += 1
        
        # 渲染目标指示器
        self.render_goal_indicator(z_buffer)
        
        # 渲染路径提示
        if self.show_path_hints:
            self.render_path_hints()
    
    def render_goal_indicator(self, z_buffer):
        """渲染目标指示器"""
        # 计算目标方向
        goal_dir = self.goal_pos - self.player_pos
        goal_angle = math.atan2(goal_dir[1], goal_dir[0])
        
        # 计算角度差
        angle_diff = goal_angle - self.player_angle
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        # 计算距离
        dist = np.linalg.norm(goal_dir)
        
        # 如果在视野范围内，且未被墙壁遮挡
        if abs(angle_diff) < self.player_fov / 2:
            # 计算屏幕x位置
            screen_x = int(self.SCREEN_WIDTH / 2 + (angle_diff / max(self.player_fov/2, 1e-6)) * (self.SCREEN_WIDTH / 2))
            screen_x = max(0, min(screen_x, self.SCREEN_WIDTH - 1))
            
            # 检查遮挡
            if dist >= z_buffer[screen_x]:
                return
            
            # 计算图标大小
            size = min(100, int(2000 / dist))
            pulse = (math.sin(self.frame_counter / 10) + 1) / 2
            size_with_pulse = int(size * (0.9 + 0.1 * pulse))
            
            # 调整位置到地面
            ground_y_pos = int(self.SCREEN_HEIGHT * 0.75)
            height_adj = int((1.0 - min(1.0, dist / self.MAX_DEPTH)) * self.SCREEN_HEIGHT * 0.25)
            icon_y_pos = ground_y_pos - height_adj - size_with_pulse // 2
            
            # 绘制图标
            if self.goal_texture:
                scaled_goal = pygame.transform.scale(self.goal_texture, (size_with_pulse, size_with_pulse))
                self.screen.blit(scaled_goal, (screen_x - size_with_pulse/2, icon_y_pos))
                
                # 添加光环
                glow_size = size_with_pulse + 10
                glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (0, 255, 0, 50), (glow_size//2, glow_size//2), glow_size//2)
                self.screen.blit(glow_surface, (screen_x - glow_size/2, icon_y_pos))
                
                # 绘制光柱
                light_pillar_height = int(size_with_pulse * 0.7)
                light_pillar_surface = pygame.Surface((size_with_pulse//3, light_pillar_height), pygame.SRCALPHA)
                pillar_gradient = [(0, 255, 0, int(200 * (1 - i/light_pillar_height))) for i in range(light_pillar_height)]
                for i, color in enumerate(pillar_gradient):
                    pygame.draw.line(light_pillar_surface, color, (size_with_pulse//6, i), (size_with_pulse//6, i), size_with_pulse//3)
                self.screen.blit(light_pillar_surface, (screen_x - size_with_pulse//6, icon_y_pos + size_with_pulse))
            else:
                color = (0, int(200 + 55 * pulse), 0)
                pygame.draw.circle(self.screen, color, (screen_x, icon_y_pos + size_with_pulse/2), size_with_pulse/2)
            
            # 显示距离
            font = pygame.font.SysFont(None, 24)
            dist_text = font.render(f"{dist:.1f}m", True, (255, 255, 255))
            text_y_pos = icon_y_pos + size_with_pulse + light_pillar_height + 5
            self.screen.blit(dist_text, (screen_x - dist_text.get_width()/2, text_y_pos))
    
    def render_path_hints(self):
        """渲染路径提示"""
        goal_dir = self.goal_pos - self.player_pos
        goal_dist = np.linalg.norm(goal_dir)
        
        if 2.0 < goal_dist < 10.0:
            # 箭头参数
            arrow_color = (0, 220, 0, 100)
            arrow_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT * 3 // 4)
            arrow_size = 40
            
            # 计算箭头角度
            goal_angle = math.atan2(goal_dir[1], goal_dir[0])
            relative_angle = goal_angle - self.player_angle
            while relative_angle > math.pi: relative_angle -= 2 * math.pi
            while relative_angle < -math.pi: relative_angle += 2 * math.pi
            
            # 计算箭头终点
            end_x = arrow_pos[0] + int(math.cos(relative_angle) * arrow_size)
            end_y = arrow_pos[1] + int(math.sin(relative_angle) * arrow_size)
            
            # 计算箭头头部
            head_angle1 = relative_angle + math.radians(150)
            head_angle2 = relative_angle + math.radians(210)
            head_len = 15
            head_x1 = end_x + int(math.cos(head_angle1) * head_len)
            head_y1 = end_y + int(math.sin(head_angle1) * head_len)
            head_x2 = end_x + int(math.cos(head_angle2) * head_len)
            head_y2 = end_y + int(math.sin(head_angle2) * head_len)
            
            # 绘制箭头
            pygame.draw.line(self.screen, arrow_color, arrow_pos, (end_x, end_y), 5)
            pygame.draw.line(self.screen, arrow_color, (end_x, end_y), (head_x1, head_y1), 5)
            pygame.draw.line(self.screen, arrow_color, (end_x, end_y), (head_x2, head_y2), 5)
    
    def render_mini_map(self):
        """渲染小地图"""
        if not self.show_minimap:
            return
        
        # 每3帧更新一次迷你地图
        map_size = 150
        cell_size = map_size // self.current_maze_size  # 使用当前迷宫大小
        
        if self.minimap_surface is None or self.minimap_update_counter % 3 == 0:
            self.minimap_surface = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
            self.minimap_surface.fill((0, 0, 0, 180))
            
            # 绘制墙壁
            for y in range(self.current_maze_size):
                for x in range(self.current_maze_size):
                    if x < len(self.maze[0]) and y < len(self.maze) and self.maze[y][x] == 1:
                        color = (150, 150, 200, 200) if x == 0 or y == 0 or x == self.current_maze_size-1 or y == self.current_maze_size-1 else (200, 200, 200, 200)
                        pygame.draw.rect(self.minimap_surface, color, (x * cell_size, y * cell_size, cell_size, cell_size))
        
        # 绘制玩家和目标
        player_layer = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
        player_layer.fill((0, 0, 0, 0))
        
        # 绘制玩家位置
        px = int(self.player_pos[0] * cell_size)
        py = int(self.player_pos[1] * cell_size)
        
        # 绘制视锥
        left_angle = self.player_angle - self.player_fov/2
        right_angle = self.player_angle + self.player_fov/2
        left_dx = math.cos(left_angle) * 30
        left_dy = math.sin(left_angle) * 30
        right_dx = math.cos(right_angle) * 30
        right_dy = math.sin(right_angle) * 30
        
        points = [(px, py), (px + int(left_dx), py + int(left_dy)), (px + int(right_dx), py + int(right_dy))]
        pygame.draw.polygon(player_layer, (255, 255, 0, 40), points)
        
        # 绘制玩家
        pygame.draw.circle(player_layer, (255, 50, 50, 200), (px, py), cell_size//2)
        
        # 绘制朝向指示
        end_x = px + int(cell_size * math.cos(self.player_angle))
        end_y = py + int(cell_size * math.sin(self.player_angle))
        pygame.draw.line(player_layer, (255, 0, 0, 200), (px, py), (end_x, end_y), 2)
        
        # 绘制目标
        gx = int(self.goal_pos[0] * cell_size)
        gy = int(self.goal_pos[1] * cell_size)
        pulse = (math.sin(self.frame_counter / 10) + 1) / 2
        goal_radius = int(cell_size * (1.0 + 0.3 * pulse))
        
        pygame.draw.circle(player_layer, (0, 255, 0, 50), (gx, gy), goal_radius + 6)
        pygame.draw.circle(player_layer, (0, 255, 0, 100), (gx, gy), goal_radius + 3)
        pygame.draw.circle(player_layer, (0, 255, 0, 200), (gx, gy), goal_radius)
        
        # 绘制小地图
        combined_map = self.minimap_surface.copy()
        combined_map.blit(player_layer, (0, 0))
        
        border = pygame.Surface((map_size + 4, map_size + 4), pygame.SRCALPHA)
        border.fill((255, 255, 255, 100))
        self.screen.blit(border, (self.SCREEN_WIDTH - map_size - 14, 8))
        self.screen.blit(combined_map, (self.SCREEN_WIDTH - map_size - 12, 10))
        
        self.minimap_update_counter += 1
    
    def update_fps(self):
        """计算和显示FPS"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fps_time > 1000:
            self.current_fps = len(self.fps_values)
            self.fps_values = []
            self.last_fps_time = current_time
        else:
            self.fps_values.append(1)
        
        font = pygame.font.SysFont(None, 24)
        fps_text = font.render(f"FPS: {self.current_fps}", True, (255, 255, 255))
        self.screen.blit(fps_text, (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 30))
    
    def handle_input(self):
        """处理用户输入"""
        keys = pygame.key.get_pressed()
        
        # 旋转
        if keys[K_LEFT] or keys[K_a]:
            self.player_angle -= self.TURN_SPEED
                
        if keys[K_RIGHT] or keys[K_d]:
            self.player_angle += self.TURN_SPEED

        
        # 移动计算
        dir_x, dir_y = math.cos(self.player_angle), math.sin(self.player_angle)
        right_x = math.cos(self.player_angle + math.pi/2)
        right_y = math.sin(self.player_angle + math.pi/2)
        
        # 前后移动
        if keys[K_UP] or keys[K_w]:
            new_x = self.player_pos[0] + dir_x * self.MOVE_SPEED
            new_y = self.player_pos[1] + dir_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
                
        if keys[K_DOWN] or keys[K_s]:
            new_x = self.player_pos[0] - dir_x * self.MOVE_SPEED
            new_y = self.player_pos[1] - dir_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
        
        # 左右平移
        if keys[K_q]:
            new_x = self.player_pos[0] - right_x * self.MOVE_SPEED
            new_y = self.player_pos[1] - right_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
                
        if keys[K_e]:
            new_x = self.player_pos[0] + right_x * self.MOVE_SPEED
            new_y = self.player_pos[1] + right_y * self.MOVE_SPEED
            if not self.is_wall(new_x, self.player_pos[1]):
                self.player_pos[0] = new_x
            if not self.is_wall(self.player_pos[0], new_y):
                self.player_pos[1] = new_y
        
        # 切换选项
        for event in pygame.event.get(KEYDOWN):
            if event.key == K_l:
                self.use_lighting = not self.use_lighting
            elif event.key == K_m:
                self.show_minimap = not self.show_minimap
            elif event.key == K_h:
                self.show_path_hints = not self.show_path_hints
    
    def check_goal(self):
        """检查是否到达目标"""
        dist_to_goal = np.linalg.norm(self.player_pos - self.goal_pos)
        # 根据难度设置不同的目标达成距离
        threshold = 1.0 if self.difficulty == 'hard' else 0.7
        if dist_to_goal < threshold:
            self.won = True
    
    def render_ui(self):
        """渲染UI元素"""
        # 显示位置信息
        font = pygame.font.SysFont(None, 24)
        pos_text = font.render(f"Position: ({self.player_pos[0]:.1f}, {self.player_pos[1]:.1f})", True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 10))
        
        # 如果胜利则显示信息
        if self.won:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            large_font = pygame.font.SysFont(None, 64)
            win_text = large_font.render("Congratulations! Exit found!", True, (255, 255, 0))
            text_rect = win_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 50))
            self.screen.blit(win_text, text_rect)
            
            if (self.frame_counter // 30) % 2 == 0:
                restart_text = font.render("Press R to restart or ESC to exit", True, (255, 255, 255))
                restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 + 50))
                self.screen.blit(restart_text, restart_rect)
        
        # 显示FPS
        self.update_fps()
    
    def reset_game(self, difficulty=None):
        """重置游戏，可以选择更改难度"""
        if difficulty and difficulty in self.DIFFICULTY_SETTINGS:
            self.difficulty = difficulty
            self.difficulty_config = self.DIFFICULTY_SETTINGS[difficulty]
            pygame.display.set_caption(f"3D Maze Adventure - {self.difficulty.capitalize()}")
            
            # 更新迷宫大小
            self.current_maze_size = self.MAZE_SIZE + self.difficulty_config['maze_size_increase']
            # 更新出口位置
            self.goal_pos = np.array([self.current_maze_size - 1.5, self.current_maze_size - 1.5])
            
        self.maze = self.generate_simplified_maze(self.current_maze_size)
        self.player_pos = np.array([1.5, 1.5])
        self.player_angle = 0
        self.won = False
        if self.sound_go:
            self.sound_go.play()
    
    def run(self):
        """主游戏循环"""
        if self.sound_go:
            self.sound_go.play()
        
        while self.running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_r:
                        self.reset_game()
            
            # 游戏逻辑
            if not self.won:
                self.handle_input()
                self.check_goal()
            
            # 渲染
            self.render_3d_view()
            self.render_mini_map()
            self.render_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

    def create_path_to_exit(self, maze, start_x, start_y, end_x, end_y):
        """创建一条从起点到终点的路径"""
        # 使用简单的方法创建路径 - 先水平移动，再垂直移动
        current_x, current_y = start_x, start_y
        
        # 先水平移动到与终点相同的x坐标
        while current_x < end_x:
            current_x += 1
            maze[current_y][current_x] = 0  # 设为路径
    
        # 再垂直移动到终点
        while current_y < end_y:
            current_y += 1
            maze[current_y][current_x] = 0  # 设为路径
    
        return maze

    def find_shortest_path(self, maze, start, end):
        """使用BFS算法找到从start到end的最短路径"""
        queue = [(start, [start])]  # 元素为(当前位置, 从起点到当前位置的路径)
        visited = set([start])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 四个方向：下、右、上、左
        size = len(maze)
        
        while queue:
            (x, y), path = queue.pop(0)  # 取队首元素
            
            # 如果到达终点
            if (x, y) == end:
                return path
                
            # 探索四个方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # 如果新位置合法且是路径且未访问过
                if (0 <= nx < size and 0 <= ny < size and 
                    maze[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
        
        return None  # 如果没有找到路径

    def narrow_paths(self, maze, size):
        """根据难度调整通道宽度"""
        narrow_factor = self.difficulty_config.get('narrow_path_factor', 0.0)
        if narrow_factor <= 0:
            return
            
        # 创建障碍物标记数组
        obstacles = np.zeros((size, size), dtype=np.int8)
        
        # 对于迷宫中的每个路径单元格
        for y in range(1, size-1):
            for x in range(1, size-1):
                if maze[y][x] == 0:  # 如果是路径
                    # 检查这个单元格是否可以变成障碍物（变窄通道）
                    if random.random() < narrow_factor:
                        # 临时将此单元格标记为墙
                        maze[y][x] = 1
                        
                        # 检查此变更是否会阻断路径
                        if self.is_path_exists(maze.copy(), (1, 1), (size-2, size-2)):
                            # 如果不阻断，保留标记
                            obstacles[y][x] = 1
                        else:
                            # 否则恢复为路径
                            maze[y][x] = 0
                
        # 应用最终的障碍物标记
        for y in range(size):
            for x in range(size):
                if obstacles[y][x] == 1:
                    maze[y][x] = 1
        
        return maze
    
    def is_path_exists(self, maze, start, end):
        """
        检查在迷宫中是否存在从start到end的路径
        
        参数:
            maze: 迷宫矩阵
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        返回:
            bool: 如果存在路径则返回True，否则返回False
        """
        # 使用广度优先搜索算法
        queue = [start]
        visited = set([start])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
                
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and
                    maze[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        
        return False

"""
炸弹人游戏难度配置模块
定义不同难度级别的游戏参数
"""
from enum import Enum
from dataclasses import dataclass


class DifficultyLevel(Enum):
    """游戏难度级别"""
    EASY = 'easy'       # 简单
    NORMAL = 'normal'   # 普通


@dataclass
class DifficultyConfig:
    """难度配置数据类"""
    name: str           # 难度名称
    grid_width: int     # 网格宽度
    grid_height: int    # 网格高度
    soft_wall_chance: float  # 软墙生成概率 
    clear_radius: int   # 玩家初始位置的清除半径
    player_speed: int   # 玩家基础速度
    max_move_distance: int  # 最大移动距离
    item_drop_chance: float  # 道具掉落概率


# 预定义的难度配置
DIFFICULTY_CONFIGS = {
    DifficultyLevel.EASY: DifficultyConfig(
        name="简单",
        grid_width=9,           # 更小的地图: 9x7
        grid_height=7,
        soft_wall_chance=0.3,   # 减少软墙(障碍物)生成概率
        clear_radius=2,         # 更大的初始清除区域
        player_speed=2,         # 更快的基础速度
        max_move_distance=6,    # 更大的移动距离
        item_drop_chance=0.5,   # 提高道具掉落概率 (原先是0.2)
    ),
    DifficultyLevel.NORMAL: DifficultyConfig(
        name="普通", 
        grid_width=13,
        grid_height=11,
        soft_wall_chance=0.6,
        clear_radius=1,
        player_speed=1,
        max_move_distance=5,
        item_drop_chance=0.2,   # 保持原来的道具掉落概率
    )
}


def get_difficulty_config(difficulty: DifficultyLevel = DifficultyLevel.NORMAL) -> DifficultyConfig:
    """获取指定难度的配置"""
    return DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS[DifficultyLevel.NORMAL])

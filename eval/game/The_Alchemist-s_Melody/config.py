"""
The Alchemist's Melody - 音乐智能体游戏配置文件
包含游戏环境、智能体和API的所有配置选项
"""
import os
from typing import Dict, Any, Optional

class Config:
    """主配置类，包含所有游戏和智能体配置"""
    
    # =================================
    # 游戏基础配置
    # =================================
    GAME_NAME = "The Alchemist's Melody"
    GAME_VERSION = "1.0.0"
    
    # 难度设置
    DIFFICULTY_LEVELS = {
        "easy": {"sequence_length": 3, "score_multiplier": 1},
        "normal": {"sequence_length": 5, "score_multiplier": 2}, 
        "hard": {"sequence_length": 7, "score_multiplier": 3}
    }
    DEFAULT_DIFFICULTY = "normal"
    
    # 颜色到动作的映射
    COLOR_ID_MAP = {
        "BLUE": 0,     # Sol
        "RED": 1,      # Do
        "GREEN": 2,    # Fa  
        "YELLOW": 3,   # Mi
        "ORANGE": 4,   # Re
        "PURPLE": 5,   # La
        "GREY": 6,     # Ti/Si
    }
    
    # 音符显示名称
    NOTE_DISPLAY_NAMES = {
        "do": "Do",
        "re": "Re", 
        "mi": "Mi",
        "fa": "Fa",
        "so": "Sol",
        "la": "La",
        "si": "Si"
    }
    
    # =================================
    # 环境配置
    # =================================
    class Environment:
        # 图像和音频设置
        SCREEN_WIDTH = 800
        SCREEN_HEIGHT = 600
        IMG_SIZE = (224, 224)  # 观测图像分辨率
        FPS = 60
        
        # 音频设置
        AUDIO_SR = 16000  # 采样率 16kHz
        AUDIO_CHANNELS = 1  # 单声道
        AUDIO_DURATION = 1  # 每步音频时长(秒)
        
        # 数据保存设置
        SAVE_DATA = True
        SAVE_SEQUENCE = True
        SAVE_DIR = "game_data/caclu"
        
        # 自动模式设置
        AUTO_START_ENABLED = True
        MAX_STEPS_PER_EPISODE = 25
    
    # =================================
    # 智能体配置
    # =================================
    class Agent:
        # 基础设置
        VERBOSE = True
        USE_LOCAL_FALLBACK = True
        MAX_RETRIES = 3
        
        # 对话策略配置
        CONVERSATION_STRATEGY = "hybrid"  # "native", "rag", "hybrid"
        NATIVE_WINDOW_SIZE = 8  # 原生对话保留轮数
        RAG_RETRIEVAL_COUNT = 3  # RAG检索相关轮数
        COMPRESS_OLD_ROUNDS = True
        MULTIMODAL_SUMMARY = True
        
        # 记忆管理
        MAX_NATIVE_HISTORY = 8
        MAX_TOTAL_MEMORY = 50
        
        # 文本输出管理
        SAVE_TEXT_OUTPUTS = True
        TEXT_OUTPUT_DIR = "agent_outputs"
    
    # =================================
    # API配置
    # =================================
    class API:
        # 主要API设置 (用户需要填写)
        BASE_URL = ""  # 填写您的API基础URL
        API_KEY = ""   # 填写您的API密钥
        MODEL_CHAT = "gemini-pro-2.5"
        
        # 备用API设置 (百川模型)
        BAICHUAN_BASE_URL = ""  # 百川API服务器地址
        BAICHUAN_ENABLED = False
        
        # 请求设置
        TIMEOUT = 300  # 请求超时时间(秒)
        MAX_TOKENS = 10000
        TEMPERATURE = 0.1
        
        # 重试设置
        RETRY_ATTEMPTS = 3
        RETRY_DELAY = 1  # 重试间隔(秒)
    
    # =================================
    # 路径配置
    # =================================
    class Paths:
        # 项目根目录
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 游戏资源路径
        GAME_DIR = os.path.join(PROJECT_ROOT, "eval", "game", "The_Alchemist-s_Melody")
        ASSETS_DIR = os.path.join(GAME_DIR, "assets")
        MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
        
        # 数据保存路径
        DATA_DIR = os.path.join(GAME_DIR, "game_data")
        LOG_DIR = os.path.join(GAME_DIR, "logs")
        DEBUG_DIR = os.path.join(GAME_DIR, "debug_logs")
        
        # 序列数据路径
        SEQUENCE_DIR = os.path.join(DATA_DIR, "sequences")
        SCORES_DIR = os.path.join(DATA_DIR, "scores")
        
        @classmethod
        def ensure_directories(cls):
            """确保所有必要的目录存在"""
            dirs_to_create = [
                cls.DATA_DIR,
                cls.LOG_DIR, 
                cls.DEBUG_DIR,
                cls.SEQUENCE_DIR,
                cls.SCORES_DIR
            ]
            
            for directory in dirs_to_create:
                os.makedirs(directory, exist_ok=True)
                
            print(f"Ensured directories exist: {dirs_to_create}")
    
    # =================================
    # 评分系统配置
    # =================================
    class Scoring:
        # 基础分数设置
        BASE_SCORE = 1000
        
        # 奖励系统
        POSITIVE_FEEDBACK_REWARD = 0.5  # 正确音符奖励
        EXPLORATION_REWARD = 0.1        # 探索奖励
        SEQUENCE_ADVANCE_REWARD = 1.0   # 序列前进奖励
        
        # 惩罚系统
        SEQUENCE_RESET_PENALTY = 1.0    # 序列重置惩罚
        MISTAKE_PENALTY_BASE = 150      # 错误基础惩罚
        
        # 完成奖励
        PERFECT_PLAY_BONUS = 500        # 完美游戏奖励
        COMPLETION_MULTIPLIER = 100     # 完成奖励倍数
        
        # 评分等级
        SCORE_RATINGS = {
            "excellent": 0.9,  # 90%以上得分
            "great": 0.6,      # 60%以上得分
            "good": 0.3,       # 30%以上得分
            "practice": 0.0    # 其他
        }
    
    # =================================
    # 调试和日志配置
    # =================================
    class Debug:
        # 日志级别
        LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
        
        # 调试选项
        SAVE_API_REQUESTS = True
        SAVE_GAME_STATES = True
        SAVE_DECISION_HISTORY = True
        
        # 详细输出选项
        SHOW_MODEL_RESPONSES = True
        SHOW_GAME_STATE_DETAILS = True
        SHOW_MEMORY_STATS = True
        
        # 性能监控
        MONITOR_RESPONSE_TIMES = True
        MONITOR_MEMORY_USAGE = True
    
    # =================================
    # 实验配置
    # =================================
    class Experiment:
        # 实验设置
        NUM_EPISODES = 10
        MAX_STEPS_PER_EPISODE = 25
        
        # 评估指标
        TRACK_COMPLETION_RATE = True
        TRACK_LEARNING_PROGRESS = True
        TRACK_EFFICIENCY_METRICS = True
        
        # 自动化设置
        AUTO_SAVE_RESULTS = True
        AUTO_GENERATE_REPORTS = True

# =================================
# 配置验证和工具函数
# =================================

def validate_config() -> bool:
    """验证配置的有效性"""
    errors = []
    
    # 检查API配置
    if not Config.API.BASE_URL:
        errors.append("API.BASE_URL 未设置")
    if not Config.API.API_KEY:
        errors.append("API.API_KEY 未设置")
    
    # 检查路径配置
    if not os.path.exists(Config.Paths.PROJECT_ROOT):
        errors.append(f"项目根目录不存在: {Config.Paths.PROJECT_ROOT}")
    
    # 检查难度配置
    if Config.DEFAULT_DIFFICULTY not in Config.DIFFICULTY_LEVELS:
        errors.append(f"默认难度 {Config.DEFAULT_DIFFICULTY} 不在可用难度列表中")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    return True

def load_config_from_env() -> None:
    """从环境变量加载配置"""
    # API配置
    Config.API.BASE_URL = os.getenv("ALCHEMIST_API_BASE", Config.API.BASE_URL)
    Config.API.API_KEY = os.getenv("ALCHEMIST_API_KEY", Config.API.API_KEY)
    Config.API.MODEL_CHAT = os.getenv("ALCHEMIST_MODEL", Config.API.MODEL_CHAT)
    
    # 百川API配置
    Config.API.BAICHUAN_BASE_URL = os.getenv("BAICHUAN_API_BASE", Config.API.BAICHUAN_BASE_URL)
    Config.API.BAICHUAN_ENABLED = os.getenv("BAICHUAN_ENABLED", "false").lower() == "true"
    
    # 调试配置
    Config.Debug.LOG_LEVEL = os.getenv("LOG_LEVEL", Config.Debug.LOG_LEVEL)
    Config.Agent.VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
    
    print("从环境变量加载配置完成")

def print_config_summary() -> None:
    """打印配置摘要"""
    print(f"""
=== {Config.GAME_NAME} v{Config.GAME_VERSION} 配置摘要 ===

游戏配置:
  - 默认难度: {Config.DEFAULT_DIFFICULTY}
  - 可用颜色: {len(Config.COLOR_ID_MAP)}
  - 自动开始: {Config.Environment.AUTO_START_ENABLED}

环境配置:
  - 图像分辨率: {Config.Environment.IMG_SIZE}
  - 音频采样率: {Config.Environment.AUDIO_SR}Hz
  - 保存数据: {Config.Environment.SAVE_DATA}

智能体配置:
  - 对话策略: {Config.Agent.CONVERSATION_STRATEGY}
  - 记忆窗口: {Config.Agent.NATIVE_WINDOW_SIZE}
  - 最大重试: {Config.Agent.MAX_RETRIES}

API配置:
  - 模型: {Config.API.MODEL_CHAT}
  - 超时时间: {Config.API.TIMEOUT}s
  - API已配置: {'是' if Config.API.API_KEY else '否'}

路径配置:
  - 数据目录: {Config.Paths.DATA_DIR}
  - 调试目录: {Config.Paths.DEBUG_DIR}

调试配置:
  - 日志级别: {Config.Debug.LOG_LEVEL}
  - 详细输出: {Config.Agent.VERBOSE}
""")

# 初始化配置
def initialize_config():
    """初始化配置系统"""
    print(f"初始化 {Config.GAME_NAME} 配置...")
    
    # 加载环境变量
    load_config_from_env()
    
    # 确保目录存在
    Config.Paths.ensure_directories()
    
    # 验证配置
    if validate_config():
        print("配置初始化成功")
        if Config.Agent.VERBOSE:
            print_config_summary()
        return True
    else:
        print("配置初始化失败")
        return False

# 如果直接运行此文件，执行配置测试
if __name__ == "__main__":
    initialize_config()

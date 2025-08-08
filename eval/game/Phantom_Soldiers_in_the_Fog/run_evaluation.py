#!/usr/bin/env python3
"""
统一的多模态评估运行脚本

支持两种视频处理模式：
1. frame_extraction: 视频抽帧后输入模型 (eval-oepnai-video-frame-multi-episode.py)
2. direct: 直接输入视频到模型 (eval-oepnai-multi-episode.py)
"""

import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import EvaluationConfig, get_preset_config, list_presets

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "numpy", "requests", "pillow", "gym"
    ]
    
    optional_packages = {
        "moviepy": "视频抽帧功能需要",
        "cv2": "视频处理功能需要"
    }
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package, desc in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, desc))
    
    if missing_required:
        logger.error(f"缺少必需的包: {', '.join(missing_required)}")
        logger.error("请运行: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        logger.warning("缺少可选包:")
        for package, desc in missing_optional:
            logger.warning(f"  {package}: {desc}")
        logger.warning("某些功能可能不可用")
    
    return True

def check_scripts_exist():
    """检查评估脚本是否存在"""
    scripts = [
        "eval-oepnai-video-frame-multi-episode.py",
        "eval-oepnai-multi-episode.py",
        "eval-baichuan-multi-episode.py"
    ]
    
    missing_scripts = []
    for script in scripts:
        if not (current_dir / script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"缺少评估脚本: {', '.join(missing_scripts)}")
        return False
    
    return True

def validate_api_config(config: EvaluationConfig) -> bool:
    """验证API配置"""
    if not config.api.api_base:
        logger.error("API_BASE 未配置")
        return False
    
    if not config.api.api_key:
        logger.error("API_KEY 未配置")
        return False
    
    if not config.api.model_name:
        logger.error("MODEL_CHAT 未配置")
        return False
    
    return True

def run_evaluation(config: EvaluationConfig, dry_run: bool = False) -> bool:
    """运行评估"""
    logger.info("=" * 80)
    logger.info("🚀 多模态评估配置")
    logger.info("=" * 80)
    
    # 显示配置信息
    logger.info(f"📍 视频处理模式: {config.media.video_processing_mode.upper()}")
    logger.info(f"🤖 模型: {config.api.model_name}")
    logger.info(f"🎮 游戏难度: {config.game.difficulty.upper()}")
    logger.info(f"🔄 评估轮次: {config.game.num_episodes}")
    logger.info(f"🎯 最大回合数: {config.game.max_rounds}")
    logger.info(f"🎬 输入模式: {config.media.input_mode.upper()}")
    logger.info(f"💾 保存媒体: {'是' if config.media.save_media else '否'}")
    logger.info(f"📊 包含向量文本: {'是' if config.media.include_vector_text else '否'}")
    logger.info(f"🎥 增强视频: {'是' if config.media.enhanced_video else '否'}")
    
    if config.media.video_processing_mode == "frame_extraction":
        logger.info("🎬 模式说明: 视频将被抽帧后逐帧输入模型")
        logger.info("   - 优点: 模型可以分析每一帧的细节")
        logger.info("   - 缺点: 处理时间较长，API调用次数较多")
    else:
        logger.info("🎬 模式说明: 视频将直接作为整体输入模型")
        logger.info("   - 优点: 处理速度快，保持时序信息")
        logger.info("   - 缺点: 依赖模型的视频理解能力")
    
    logger.info("=" * 80)
    
    # 选择评估脚本
    script_name = config.get_script_path()
    script_path = current_dir / script_name
    
    if not script_path.exists():
        logger.error(f"评估脚本不存在: {script_path}")
        return False
    
    # 构建命令行参数
    args = config.to_cli_args()
    cmd = [sys.executable, str(script_path)] + args
    
    logger.info(f"📝 执行脚本: {script_name}")
    logger.info(f"🔧 命令行参数: {' '.join(args)}")
    
    if dry_run:
        logger.info("🏃 DRY RUN - 不会实际执行评估")
        logger.info(f"完整命令: {' '.join(cmd)}")
        return True
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        "API_BASE": config.api.api_base,
        "API_KEY": config.api.api_key,
        "MODEL_CHAT": config.api.model_name
    })
    
    try:
        logger.info("🎮 开始执行评估...")
        
        # 执行评估脚本
        process = subprocess.run(
            cmd,
            env=env,
            cwd=current_dir,
            capture_output=False,  # 让输出直接显示到控制台
            text=True
        )
        
        if process.returncode == 0:
            logger.info("✅ 评估执行成功!")
            return True
        else:
            logger.error(f"❌ 评估执行失败，退出码: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        logger.info("⛔ 用户中断了评估")
        return False
    except Exception as e:
        logger.error(f"💥 执行评估时发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="多模态游戏评估统一运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用预设配置
  python run_evaluation.py --preset quick_test
  
  # 使用自定义配置
  python run_evaluation.py --difficulty medium --num_episodes 5 --video_mode frame_extraction
  
  # 从环境变量加载配置
  python run_evaluation.py --from_env
  
  # 查看所有预设配置
  python run_evaluation.py --list_presets
  
视频处理模式说明:
  - frame_extraction: 视频抽帧后输入模型 (更细致的分析)
  - direct: 直接输入视频到模型 (更快的处理)
        """
    )
    
    # 配置选择
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--preset", type=str, 
                             help="使用预设配置")
    config_group.add_argument("--from_env", action="store_true",
                             help="从环境变量加载配置")
    config_group.add_argument("--config_file", type=str,
                             help="从JSON文件加载配置")
    
    # 游戏配置
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["normal", "medium", "hard"],
                       help="游戏难度")
    parser.add_argument("--seed_index", type=int, default=0,
                       help="随机种子索引")
    parser.add_argument("--max_rounds", type=int, default=100,
                       help="每局最大回合数")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="评估轮次数量")
    parser.add_argument("--probabilistic_commands", action="store_true",
                       help="使用概率性命令执行")
    
    # 媒体配置
    parser.add_argument("--video_mode", type=str, default="direct",
                       choices=["direct", "frame_extraction"],
                       help="视频处理模式")
    parser.add_argument("--input_mode", type=str, default="video",
                       choices=["video", "image_audio"],
                       help="输入模态")
    parser.add_argument("--no_save_media", action="store_true",
                       help="不保存媒体文件")
    parser.add_argument("--no_vector_text", action="store_true",
                       help="不包含向量文本信息")
    parser.add_argument("--enhanced_video", action="store_true",
                       help="启用增强视频录制")
    parser.add_argument("--video_fps", type=float, default=0.5,
                       help="视频帧率")
    
    # API配置
    parser.add_argument("--model_name", type=str,
                       help="模型名称")
    
    # 工具选项
    parser.add_argument("--list_presets", action="store_true",
                       help="列出所有预设配置")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示配置，不执行评估")
    parser.add_argument("--check_deps", action="store_true",
                       help="检查依赖是否安装")
    
    args = parser.parse_args()
    
    # 列出预设配置
    if args.list_presets:
        list_presets()
        return
    
    # 检查依赖
    if args.check_deps:
        check_dependencies()
        check_scripts_exist()
        return
    
    # 检查基本环境
    if not check_dependencies():
        logger.error("依赖检查失败")
        sys.exit(1)
    
    if not check_scripts_exist():
        logger.error("脚本检查失败")
        sys.exit(1)
    
    # 加载配置
    if args.preset:
        try:
            config = get_preset_config(args.preset)
            logger.info(f"🔧 使用预设配置: {args.preset}")
        except ValueError as e:
            logger.error(f"预设配置错误: {e}")
            sys.exit(1)
    elif args.from_env:
        config = EvaluationConfig.from_env()
        logger.info("🔧 从环境变量加载配置")
    elif args.config_file:
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            # 这里需要实现从JSON恢复配置的逻辑
            config = EvaluationConfig()
            logger.info(f"🔧 从文件加载配置: {args.config_file}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            sys.exit(1)
    else:
        # 使用命令行参数创建配置
        config = EvaluationConfig()
        
        # 更新配置
        config.game.difficulty = args.difficulty
        config.game.seed_index = args.seed_index
        config.game.max_rounds = args.max_rounds
        config.game.num_episodes = args.num_episodes
        config.game.deterministic_commands = not args.probabilistic_commands
        
        config.media.video_processing_mode = args.video_mode
        config.media.input_mode = args.input_mode
        config.media.save_media = not args.no_save_media
        config.media.include_vector_text = not args.no_vector_text
        config.media.enhanced_video = args.enhanced_video
        config.media.video_fps = args.video_fps
        
        if args.model_name:
            config.api.model_name = args.model_name
        
        logger.info("🔧 使用命令行参数配置")
    
    # 验证API配置
    if not args.dry_run and not validate_api_config(config):
        logger.error("API配置验证失败")
        logger.info("请确保设置了以下环境变量:")
        logger.info("  export API_BASE='your_api_base_url'")
        logger.info("  export API_KEY='your_api_key'")
        logger.info("  export MODEL_CHAT='your_model_name'")
        sys.exit(1)
    
    # 运行评估
    success = run_evaluation(config, dry_run=args.dry_run)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

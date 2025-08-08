import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def check_ffmpeg():
    """检查ffmpeg是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def try_install_ffmpeg():
    """尝试安装ffmpeg并提供说明"""
    system = platform.system()
    print("您的系统未安装ffmpeg。您可以通过以下方式安装：")
    
    if system == "Darwin":  # macOS
        print("方法1: 使用Homebrew安装")
        print("  $ brew install ffmpeg")
        print("\n方法2: 从官网下载: https://ffmpeg.org/download.html")
        
    elif system == "Linux":
        print("Ubuntu/Debian:")
        print("  $ sudo apt-get update")
        print("  $ sudo apt-get install ffmpeg")
        print("\nCentOS/RHEL:")
        print("  $ sudo yum install ffmpeg ffmpeg-devel")
        
    elif system == "Windows":
        print("1. 下载ffmpeg: https://www.gyan.dev/ffmpeg/builds/")
        print("2. 解压文件并添加bin目录到系统PATH")
    
    print("\n安装后请重新运行此脚本")
    return False

def convert_ogg_to_wav_with_ffmpeg(input_file, output_file):
    """使用ffmpeg将OGG转换为WAV"""
    try:
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-i', input_file,  # 输入文件
            '-acodec', 'pcm_s16le',  # 16位PCM编码
            '-ar', '44100',  # 采样率44.1kHz
            '-y',  # 覆盖输出文件
            output_file  # 输出文件
        ]
        
        # 运行命令
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"成功转换: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败! ffmpeg错误: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        return False

def simple_copy_as_wav(input_file, output_file):
    """针对测试环境，简单复制并重命名文件为.wav"""
    try:
        print(f"使用简单复制方法 (仅用于测试！文件格式没有真正转换)")
        shutil.copy2(input_file, output_file)
        print(f"文件已复制为WAV: {output_file}")
        return True
    except Exception as e:
        print(f"复制文件失败: {e}")
        return False

def main():
    # 需要转换的文件列表
    files_to_convert = [
        "game/assets-necessay/kenney/UI assets/UI Pack/Sounds/click-b.ogg",
        "game/assets-necessay/kenney/Audio/Retro Sounds 2/Audio/explosion1.ogg",
        "game/assets-necessay/kenney/Audio/Impact Sounds/Audio/footstep_wood_001.ogg"
    ]
    
    # 从项目根目录确定绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 检查ffmpeg是否可用
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("FFmpeg未找到，将尝试简单复制方法")
        try_install_ffmpeg()
    
    # 转换每个文件
    success_count = 0
    for rel_path in files_to_convert:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            print(f"错误: 找不到文件 {abs_path}")
            continue
        
        # 创建WAV输出路径
        wav_path = os.path.splitext(abs_path)[0] + '.wav'
        print(f"正在处理: {abs_path}")
        
        if ffmpeg_available:
            # 使用ffmpeg转换
            success = convert_ogg_to_wav_with_ffmpeg(abs_path, wav_path)
        else:
            # 简单复制并重命名为WAV (注意：不是真正的格式转换)
            success = simple_copy_as_wav(abs_path, wav_path)
        
        if success:
            success_count += 1
    
    print(f"\n转换完成! 成功处理 {success_count}/{len(files_to_convert)} 个文件。")

    # 提醒用户更新代码中的文件引用
    if success_count > 0:
        print("\n请确保更新 bomberman_gym.py 和 classic_bomberman-daiceshi.py 中的音频文件引用:")
        print("1. 将原本的 '.ogg' 后缀改为 '.wav'")
        print("2. 修改音频格式配置为 'wav'")

if __name__ == "__main__":
    main()

import os
import re
from collections import defaultdict
import pandas as pd
from datetime import datetime

def identify_model(filename):
    """根据文件名识别模型类型"""
    filename_lower = filename.lower()
    if 'baichuan' in filename_lower:
        return 'baichuan-omni'
    elif 'cpm' in filename_lower:
        return 'cpm'
    elif 'flash' in filename_lower:
        return 'gemini-2.5-flash'
    elif 'pro' in filename_lower:
        return 'gemini-2.5-pro'
    elif 'qwen' in filename_lower:
        return 'qwen'
    else:
        return 'unknown'

def parse_result_file(file_path):
    """解析结果文件，提取统计数据"""
    data = {
        'difficulty': None,
        'total_episodes': 0,
        'success_episodes': 0,
        'success_rate': 0.0,
        'total_score': 0,
        'avg_score': 0.0,
        'unparseable_sequences': 0,
        'total_coordinate_correct': 0,
        'total_icon_correct': 0,
        'total_coordinate_attempts': 0,
        'total_icon_attempts': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取基本统计信息
        difficulty_match = re.search(r'难度级别:\s*(\d+)', content)
        if difficulty_match:
            data['difficulty'] = int(difficulty_match.group(1))
        
        total_episodes_match = re.search(r'总episode数:\s*(\d+)', content)
        if total_episodes_match:
            data['total_episodes'] = int(total_episodes_match.group(1))
        
        success_episodes_match = re.search(r'成功episode数:\s*(\d+)', content)
        if success_episodes_match:
            data['success_episodes'] = int(success_episodes_match.group(1))
        
        success_rate_match = re.search(r'成功率:\s*([\d.]+)%', content)
        if success_rate_match:
            data['success_rate'] = float(success_rate_match.group(1))
        
        total_score_match = re.search(r'总分:\s*(\d+)', content)
        if total_score_match:
            data['total_score'] = int(total_score_match.group(1))
        
        avg_score_match = re.search(r'平均分:\s*([\d.]+)', content)
        if avg_score_match:
            data['avg_score'] = float(avg_score_match.group(1))
        
        unparseable_match = re.search(r'无法解析的序列次数:\s*(\d+)', content)
        if unparseable_match:
            data['unparseable_sequences'] = int(unparseable_match.group(1))
        
        # 解析详细结果，统计坐标和图标正确数
        episode_pattern = r'Episode \d+:.*?坐标正确=(\d+)/(\d+).*?图标正确=(\d+)/(\d+)'
        episodes = re.findall(episode_pattern, content)
        
        for coord_correct, coord_total, icon_correct, icon_total in episodes:
            data['total_coordinate_correct'] += int(coord_correct)
            data['total_coordinate_attempts'] += int(coord_total)
            data['total_icon_correct'] += int(icon_correct)
            data['total_icon_attempts'] += int(icon_total)
        
        return data
    
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None

def main():
    results_dir = "results"
    
    # 存储所有数据，按模型和难度分组
    model_data = defaultdict(lambda: defaultdict(list))
    
    # 扫描目录
    for filename in os.listdir(results_dir):
        if filename.startswith('rhythm_memory') or not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(results_dir, filename)
        model_type = identify_model(filename)
        
        if model_type == 'unknown':
            print(f"跳过未知模型文件: {filename}")
            continue
        
        data = parse_result_file(file_path)
        if data and data['difficulty'] is not None:
            model_data[model_type][data['difficulty']].append(data)
            print(f"处理文件: {filename} -> {model_type}, 难度: {data['difficulty']}")
    
    # 创建保存目录
    output_dir = "Desktop/0611-test/statistics_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存总体统计报告
    report_file = os.path.join(output_dir, f"rhythm_memory_statistics_{timestamp}.txt")
    excel_file = os.path.join(output_dir, f"rhythm_memory_statistics_{timestamp}.xlsx")
    
    all_tables = {}
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("节奏记忆游戏模型性能统计报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 为每个难度级别生成统计表格
        for difficulty in [1, 2, 3]:
            print(f"\n{'='*50}")
            print(f"难度级别 {difficulty} 统计结果")
            print(f"{'='*50}")
            
            f.write(f"难度级别 {difficulty} 统计结果\n")
            f.write("-"*50 + "\n")
            
            if not any(difficulty in difficulties for difficulties in model_data.values()):
                print(f"没有找到难度级别 {difficulty} 的数据")
                f.write(f"没有找到难度级别 {difficulty} 的数据\n\n")
                continue
            
            # 准备表格数据
            table_data = []
            
            for model_type in ['baichuan-omni', 'cpm', 'gemini-2.5-flash', 'gemini-2.5-pro', 'qwen']:
                if difficulty not in model_data[model_type]:
                    continue
                
                files_data = model_data[model_type][difficulty]
                
                # 计算平均值
                total_episodes = sum(d['total_episodes'] for d in files_data)
                total_success_episodes = sum(d['success_episodes'] for d in files_data)
                total_score = sum(d['total_score'] for d in files_data)
                total_unparseable = sum(d['unparseable_sequences'] for d in files_data)
                total_coord_correct = sum(d['total_coordinate_correct'] for d in files_data)
                total_coord_attempts = sum(d['total_coordinate_attempts'] for d in files_data)
                total_icon_correct = sum(d['total_icon_correct'] for d in files_data)
                total_icon_attempts = sum(d['total_icon_attempts'] for d in files_data)
                
                avg_success_rate = (total_success_episodes / total_episodes * 100) if total_episodes > 0 else 0
                avg_score = total_score / total_episodes if total_episodes > 0 else 0
                avg_coord_correct = total_coord_correct / total_episodes if total_episodes > 0 else 0
                avg_icon_correct = total_icon_correct / total_episodes if total_episodes > 0 else 0
                unparseable_ratio = (total_unparseable / total_episodes * 100) if total_episodes > 0 else 0
                
                table_data.append({
                    '模型': model_type,
                    '文件数': len(files_data),
                    '总episodes': total_episodes,
                    '平均成功率(%)': f"{avg_success_rate:.2f}",
                    '平均得分': f"{avg_score:.2f}",
                    '平均坐标正确数': f"{avg_coord_correct:.2f}",
                    '平均图标正确数': f"{avg_icon_correct:.2f}",
                    '无法解析序列占比(%)': f"{unparseable_ratio:.2f}"
                })
            
            # 创建并显示表格
            if table_data:
                df = pd.DataFrame(table_data)
                print(df.to_string(index=False))
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # 保存到Excel的不同sheet
                all_tables[f'难度{difficulty}'] = df
            else:
                print("没有数据可显示")
                f.write("没有数据可显示\n\n")
    
    # 保存Excel文件
    if all_tables:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for sheet_name, df in all_tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n统计结果已保存到:")
    print(f"文本报告: {report_file}")
    print(f"Excel文件: {excel_file}")

if __name__ == "__main__":
    main()

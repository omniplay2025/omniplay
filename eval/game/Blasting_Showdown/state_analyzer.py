import json
import os
import glob
import re
from collections import defaultdict
from datetime import datetime

class BombermanStatsAnalyzer:
    def __init__(self):
        self.model_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0, 'kills': 0, 'deaths': 0, 'items': 0}))
        self.player_id_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0, 'kills': 0, 'deaths': 0, 'items': 0, 'models': defaultdict(int)}))
        
    def extract_difficulty_from_filename(self, filename):
        """从文件名中提取难度信息"""
        # 尝试匹配常见的难度模式
        difficulty_patterns = [
            r'easy|简单',
            r'medium|normal|中等|普通',
            r'hard|difficult|困难',
            r'expert|专家',
            r'easy_(\d+)',
            r'medium_(\d+)',
            r'hard_(\d+)'
        ]
        
        filename_lower = filename.lower()
        
        for pattern in difficulty_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                if 'easy' in pattern or '简单' in pattern:
                    return 'easy'
                elif 'medium' in pattern or 'normal' in pattern or '中等' in pattern or '普通' in pattern:
                    return 'medium'
                elif 'hard' in pattern or 'difficult' in pattern or '困难' in pattern:
                    return 'hard'
                elif 'expert' in pattern or '专家' in pattern:
                    return 'expert'
        
        # 如果没有匹配到特定难度，返回默认值
        return 'unknown'
        
    def extract_difficulty_from_data_or_filename(self, data, filename):
        """从数据或文件名中提取难度信息，优先使用数据中的难度字段"""
        # 首先尝试从数据中获取难度
        if 'difficulty' in data:
            return data['difficulty']
        
        # 如果数据中没有难度字段，则从文件名推断
        return self.extract_difficulty_from_filename(filename)
        
    def analyze_episode_format(self, data, difficulty, is_stats_file=False):
        """分析单集格式的数据 (bomberman_episode_*.json)"""
        player_mapping = data.get('player_mapping', {})
        episode_stats = data.get('episode_stats', {})
        
        for model_name, stats in episode_stats.items():
            player_id = str(stats.get('player_id', ''))
            
            # 更新模型统计（按难度分类）
            self.model_stats[model_name][difficulty]['games'] += 1
            self.model_stats[model_name][difficulty]['kills'] += stats.get('kills', 0)
            self.model_stats[model_name][difficulty]['deaths'] += stats.get('deaths', 0)
            self.model_stats[model_name][difficulty]['items'] += stats.get('items_collected', 0)
            if stats.get('won', False):
                self.model_stats[model_name][difficulty]['wins'] += 1
                
            # 只有非stats文件才更新player_id统计
            if not is_stats_file:
                self.player_id_stats[player_id][difficulty]['games'] += 1
                self.player_id_stats[player_id][difficulty]['kills'] += stats.get('kills', 0)
                self.player_id_stats[player_id][difficulty]['deaths'] += stats.get('deaths', 0)
                self.player_id_stats[player_id][difficulty]['items'] += stats.get('items_collected', 0)
                self.player_id_stats[player_id][difficulty]['models'][model_name] += 1
                if stats.get('won', False):
                    self.player_id_stats[player_id][difficulty]['wins'] += 1
    
    def analyze_stats_format(self, data, difficulty):
        """分析统计格式的数据 (bomberman_stats_*.json)"""
        models = data.get('models', {})
        wins = data.get('wins', {})
        player_stats = data.get('player_stats', {})
        
        for player_id, model_name in models.items():
            stats = player_stats.get(player_id, {})
            win_count = wins.get(player_id, 0)
            episodes = stats.get('episodes', [])
            game_count = len(episodes)
            
            # 只更新模型统计，不更新player_id统计（按难度分类）
            self.model_stats[model_name][difficulty]['games'] += game_count
            self.model_stats[model_name][difficulty]['wins'] += win_count
            self.model_stats[model_name][difficulty]['kills'] += stats.get('kills', 0)
            self.model_stats[model_name][difficulty]['deaths'] += stats.get('deaths', 0)
            self.model_stats[model_name][difficulty]['items'] += stats.get('items_collected', 0)
    
    def process_files(self, result_dir):
        """处理结果目录中的所有JSON文件"""
        json_files = glob.glob(os.path.join(result_dir, "*.json"))
        
        for file_path in json_files:
            try:
                filename = os.path.basename(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从数据或文件名中提取难度信息
                difficulty = self.extract_difficulty_from_data_or_filename(data, filename)
                
                # 判断是否为stats文件
                is_stats_file = "stats" in filename.lower()
                
                # 判断文件格式类型
                if 'episode_stats' in data and 'player_mapping' in data:
                    # 单集格式
                    self.analyze_episode_format(data, difficulty, is_stats_file)
                elif 'models' in data and 'player_stats' in data:
                    # 统计格式（这类文件本身就是stats文件）
                    self.analyze_stats_format(data, difficulty)
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    
    def calculate_win_rates(self, stats):
        """计算胜率"""
        result = {}
        for key, difficulty_data in stats.items():
            result[key] = {}
            for difficulty, data in difficulty_data.items():
                win_rate = (data['wins'] / data['games'] * 100) if data['games'] > 0 else 0
                result_data = {
                    'games': data['games'],
                    'wins': data['wins'],
                    'win_rate': round(win_rate, 2),
                    'kills': data['kills'],
                    'deaths': data['deaths'],
                    'items_collected': data['items'],
                    'kd_ratio': round(data['kills'] / max(data['deaths'], 1), 2)
                }
                
                # 如果有models字段，添加模型出现次数统计
                if 'models' in data:
                    result_data['models'] = dict(data['models'])
                    
                result[key][difficulty] = result_data
        return result
    
    def generate_report(self, output_dir):
        """生成统计报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算统计结果
        model_report = self.calculate_win_rates(self.model_stats)
        player_id_report = self.calculate_win_rates(self.player_id_stats)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型统计
        model_file = os.path.join(output_dir, f"model_stats_{timestamp}.json")
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_statistics': model_report
            }, f, indent=2, ensure_ascii=False)
        
        # 保存player_id统计
        player_file = os.path.join(output_dir, f"player_id_stats_{timestamp}.json")
        with open(player_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'player_id_statistics': player_id_report
            }, f, indent=2, ensure_ascii=False)
        
        # 生成汇总报告
        summary_file = os.path.join(output_dir, f"summary_report_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("炸弹人游戏统计报告 (按难度分类)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("模型统计:\n")
            f.write("-" * 30 + "\n")
            for model, difficulty_stats in model_report.items():
                f.write(f"{model}:\n")
                for difficulty, stats in difficulty_stats.items():
                    f.write(f"  难度 [{difficulty}]:\n")
                    f.write(f"    游戏场次: {stats['games']}\n")
                    f.write(f"    胜利次数: {stats['wins']}\n")
                    f.write(f"    胜率: {stats['win_rate']}%\n")
                    f.write(f"    击杀数: {stats['kills']}\n")
                    f.write(f"    死亡数: {stats['deaths']}\n")
                    f.write(f"    K/D比: {stats['kd_ratio']}\n")
                    f.write(f"    收集道具: {stats['items_collected']}\n")
                f.write("\n")
            
            f.write("玩家ID统计:\n")
            f.write("-" * 30 + "\n")
            for player_id, difficulty_stats in player_id_report.items():
                f.write(f"Player {player_id}:\n")
                for difficulty, stats in difficulty_stats.items():
                    f.write(f"  难度 [{difficulty}]:\n")
                    f.write(f"    游戏场次: {stats['games']}\n")
                    f.write(f"    胜利次数: {stats['wins']}\n")
                    f.write(f"    胜率: {stats['win_rate']}%\n")
                    f.write(f"    击杀数: {stats['kills']}\n")
                    f.write(f"    死亡数: {stats['deaths']}\n")
                    f.write(f"    K/D比: {stats['kd_ratio']}\n")
                    f.write(f"    收集道具: {stats['items_collected']}\n")
                    if 'models' in stats:
                        f.write(f"    模型出现次数:\n")
                        for model, count in stats['models'].items():
                            f.write(f"      {model}: {count}次\n")
                f.write("\n")
        
        print(f"统计报告已生成:")
        print(f"- 模型统计: {model_file}")
        print(f"- 玩家ID统计: {player_file}")
        print(f"- 汇总报告: {summary_file}")

def main():
    analyzer = BombermanStatsAnalyzer()
    
    # 处理结果文件
    result_dir = ""
    output_dir = ""
    
    print("开始分析炸弹人游戏数据...")
    analyzer.process_files(result_dir)
    
    print("生成统计报告...")
    analyzer.generate_report(output_dir)
    
    print("分析完成！")

if __name__ == "__main__":
    main()

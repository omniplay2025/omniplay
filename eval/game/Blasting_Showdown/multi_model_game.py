import os
import time
import json
import argparse
import datetime
import random
from typing import Dict, List
import copy

# 导入游戏环境和AI控制器
from bomberman_gym import BombermanEnv, BombermanAction
from ai_player_controller import AIPlayerController
from game_difficulty import DifficultyLevel

class MultiModelGame:
    """多模型炸弹人游戏控制器"""
    
    def __init__(self, model_configs, episodes=3, steps_per_episode=300, delay=0.3, difficulty=DifficultyLevel.NORMAL):
        """
        初始化多模型游戏
        
        Args:
            model_configs: 模型配置列表 [{"api_base": "...", "api_key": "...", "model": "...", "player_id": 0}, ...]
            episodes: 游戏局数
            steps_per_episode: 每局最大步数
            delay: 每步延迟时间(秒)
            difficulty: 游戏难度
        """
        self.original_model_configs = copy.deepcopy(model_configs)  # 保存原始配置
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.delay = delay
        self.difficulty = difficulty
        
        # 验证配置
        assert len(model_configs) <= 4, "最多支持4个模型"
        
        # 创建游戏环境
        self.env = BombermanEnv(render_mode='human', difficulty=difficulty)
        
        # 存储每个episode的模型映射关系
        self.episode_model_mappings = []
        
        # 添加统计数据结构 - 按模型名称而不是player_id
        self.model_stats = {}
        for config in self.original_model_configs:
            model_name = config["model"]
            self.model_stats[model_name] = {
                "kills": 0,           # 总击杀数
                "deaths": 0,          # 总死亡数
                "items_collected": 0, # 总收集道具数
                "wins": 0,           # 总胜场数
                "episodes": []        # 每回合的统计
            }
    
    def _shuffle_player_assignments(self, episode_num):
        """打乱玩家编号分配"""
        # 获取所有模型配置，不包含player_id
        models = []
        for config in self.original_model_configs:
            models.append({
                "api_base": config["api_base"],
                "api_key": config["api_key"],
                "model": config["model"],
                "description": config.get("description", config["model"])
            })
        
        # 打乱模型顺序 - 确保每次都有不同的随机种子
        random.seed(time.time() + episode_num)  # 使用时间和episode号作为种子
        random.shuffle(models)
        
        # 重新分配player_id
        shuffled_configs = []
        episode_mapping = {}
        
        for i, model_config in enumerate(models):
            new_config = copy.deepcopy(model_config)
            new_config["player_id"] = i
            shuffled_configs.append(new_config)
            
            # 记录映射关系：player_id -> model_name
            episode_mapping[i] = model_config["model"]
        
        # 保存这个episode的映射关系
        self.episode_model_mappings.append({
            "episode": episode_num + 1,  # 修改这里，使用episode_num + 1
            "mapping": episode_mapping
        })
        
        print(f"\n第 {episode_num+1} 局玩家分配:")
        for player_id, model_name in episode_mapping.items():
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"  玩家{player_id+1}: {model_short}")
        
        return shuffled_configs
    
    def _create_controllers(self, model_configs):
        """根据当前配置创建AI控制器"""
        controllers = {}
        for config in model_configs:
            player_id = config["player_id"]
            controllers[player_id] = AIPlayerController(
                config["api_base"], 
                config["api_key"],
                config["model"]
            )
        return controllers
    
    def run_game(self):
        """运行游戏"""
        try:
            overall_wins = {config["model"]: 0 for config in self.original_model_configs}
            
            for episode in range(self.episodes):
                print(f"\n====== 第 {episode+1}/{self.episodes} 局游戏 ======")
                
                # 打乱玩家分配
                current_configs = self._shuffle_player_assignments(episode)
                self.controllers = self._create_controllers(current_configs)
                
                obs, info = self.env.reset()
                
                # 为当前回合初始化每个模型的统计数据
                episode_stats = {}
                for config in current_configs:
                    model_name = config["model"]
                    episode_stats[model_name] = {
                        "player_id": config["player_id"],
                        "kills": 0,
                        "deaths": 0,
                        "items_collected": 0,
                        "won": False
                    }
                
                ep_start_time = time.time()
                for step in range(self.steps_per_episode):
                    step_start_time = time.time()
                    
                    # 获取所有活跃玩家的决策
                    actions = {}
                    for player_id, controller in self.controllers.items():
                        # 检查玩家是否存活
                        if obs['state']['players'][player_id]['alive'] == 0:
                            continue
                        
                        # 获取AI决策
                        action = controller.get_decision(player_id, obs)
                        actions[player_id] = action
                    
                    # 执行动作
                    obs, rewards, terminated, truncated, info = self.env.step(actions)
                    
                    # 更新统计数据
                    self._update_episode_stats(info, episode_stats, current_configs)
                    
                    # 计算步骤耗时
                    step_time = time.time() - step_start_time
                    print(f"回合 {step+1} 耗时: {step_time:.2f}秒")

                    # 在图形界面中添加延迟以便观察
                    if step_time < self.delay:
                        time.sleep(self.delay - step_time)
                    
                    # 检查游戏是否结束
                    if terminated:
                        # 找出获胜者
                        winner_player_id = None
                        for pid, p_info in obs['state']['players'].items():
                            if p_info['alive'] == 1:
                                winner_player_id = int(pid)
                                break
                        
                        if winner_player_id is not None:
                            # 找到获胜模型
                            winner_model = None
                            for config in current_configs:
                                if config["player_id"] == winner_player_id:
                                    winner_model = config["model"]
                                    break
                            
                            if winner_model:
                                episode_stats[winner_model]["won"] = True
                                overall_wins[winner_model] += 1
                                model_short = winner_model.split('/')[-1] if '/' in winner_model else winner_model
                                print(f"游戏结束! 玩家{winner_player_id+1}({model_short})获胜!")
                        else:
                            print("游戏结束! 平局!")
                        break
                    elif truncated:
                        print(f"达到最大步数 {self.steps_per_episode}!")
                        break
                
                # 显示该回合的统计信息
                self._print_episode_stats(episode, episode_stats)
                
                # 更新总体统计数据
                self._update_overall_stats(episode_stats)
                
                # 计算局时
                ep_time = time.time() - ep_start_time
                print(f"第 {episode+1} 局游戏完成，耗时: {ep_time:.2f}秒")
                
                # 每个episode结束后保存统计结果
                self._save_episode_results(episode, episode_stats, overall_wins)
            
            # 输出总统计信息
            print("\n====== 游戏总统计 ======")
            self._print_overall_stats(overall_wins)
            
            # 保存最终统计结果
            self._save_final_results(overall_wins)
            
        except KeyboardInterrupt:
            print("用户中断，结束游戏")
        except Exception as e:
            print(f"游戏过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保正确关闭环境
            self.env.close()

    def _update_episode_stats(self, info, episode_stats, current_configs):
        """更新episode统计信息"""
        # 创建player_id到model_name的映射
        player_to_model = {config["player_id"]: config["model"] for config in current_configs}
        
        # 处理击杀事件
        for kill_event in info.get("kill_events", []):
            killer_id = kill_event["killer"]
            victim_id = kill_event["victim"]
            
            # 确保不统计自杀
            if killer_id != victim_id:
                # 更新击杀者的击杀数
                if killer_id in player_to_model:
                    killer_model = player_to_model[killer_id]
                    episode_stats[killer_model]["kills"] += 1
            
            # 更新被击杀者的死亡数
            if victim_id in player_to_model:
                victim_model = player_to_model[victim_id]
                episode_stats[victim_model]["deaths"] += 1
        
        # 处理道具收集事件
        for item_event in info.get("item_pickup_events", []):
            player_id = item_event["player"]
            if player_id in player_to_model:
                model_name = player_to_model[player_id]
                episode_stats[model_name]["items_collected"] += 1
    
    def _update_overall_stats(self, episode_stats):
        """更新总体统计数据"""
        for model_name, stats in episode_stats.items():
            if model_name in self.model_stats:
                self.model_stats[model_name]["kills"] += stats["kills"]
                self.model_stats[model_name]["deaths"] += stats["deaths"]
                self.model_stats[model_name]["items_collected"] += stats["items_collected"]
                if stats["won"]:
                    self.model_stats[model_name]["wins"] += 1
                
                # 保存episode详细数据
                self.model_stats[model_name]["episodes"].append({
                    "episode": len(self.model_stats[model_name]["episodes"]) + 1,
                    "player_id": stats["player_id"],
                    "kills": stats["kills"],
                    "deaths": stats["deaths"],
                    "items_collected": stats["items_collected"],
                    "won": stats["won"]
                })

    def _print_episode_stats(self, episode_num, episode_stats):
        """打印单个回合的统计信息"""
        print(f"\n--- 第 {episode_num+1} 局统计 ---")
        print(f"{'模型':^20}|{'玩家':^6}|{'击杀':^6}|{'死亡':^6}|{'道具':^6}|{'胜利':^6}")
        print("-" * 60)
        
        for model_name, stats in episode_stats.items():
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
            model_display = model_short[:20] if len(model_short) > 20 else model_short
            won_display = "是" if stats["won"] else "否"
            print(f"{model_display:^20}|{stats['player_id']+1:^6}|{stats['kills']:^6}|{stats['deaths']:^6}|{stats['items_collected']:^6}|{won_display:^6}")
    
    def _print_overall_stats(self, overall_wins):
        """打印总体统计信息"""
        print(f"\n--- 总体数据统计 ({self.episodes}局) ---")
        print(f"{'模型':^20}|{'总击杀':^6}|{'总死亡':^6}|{'总道具':^6}|{'胜场':^6}|{'胜率':^8}")
        print("-" * 70)
        
        for model_name, stats in self.model_stats.items():
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
            model_display = model_short[:20] if len(model_short) > 20 else model_short
            wins = stats["wins"]
            win_rate = f"{(wins/self.episodes)*100:.1f}%" if self.episodes > 0 else "0.0%"
            print(f"{model_display:^20}|{stats['kills']:^6}|{stats['deaths']:^6}|{stats['items_collected']:^6}|{wins:^6}|{win_rate:^8}")

    def _save_episode_results(self, episode_num, episode_stats, overall_wins):
        """保存单个episode的统计结果"""
        # 创建result目录（如果不存在）
        result_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # 准备episode统计数据
        episode_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode": episode_num + 1,
            "difficulty": self.difficulty.value,
            "steps_per_episode": self.steps_per_episode,
            "player_mapping": self.episode_model_mappings[episode_num]["mapping"],  # 确保索引正确
            "episode_stats": episode_stats,
            "current_overall_wins": overall_wins,
            "debug_mappings_count": len(self.episode_model_mappings),  # 添加调试信息
            "debug_episode_num": episode_num
        }
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bomberman_episode_{episode_num+1}_{timestamp}.json"
        filepath = os.path.join(result_dir, filename)
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        print(f"第 {episode_num+1} 局统计结果已保存到: {filepath}")
        print(f"调试信息: mappings总数={len(self.episode_model_mappings)}, 当前episode={episode_num}")

    def _save_final_results(self, overall_wins):
        """保存最终统计结果"""
        # 创建result目录（如果不存在）
        result_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # 准备完整的统计数据
        full_stats = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "difficulty": self.difficulty.value,
            "episodes": self.episodes,
            "steps_per_episode": self.steps_per_episode,
            "episode_mappings": self.episode_model_mappings,
            "model_stats": self.model_stats,
            "overall_wins": overall_wins,
            "original_configs": self.original_model_configs
        }
        
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bomberman_final_stats_{timestamp}.json"
        filepath = os.path.join(result_dir, filename)
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n最终统计结果已保存到: {filepath}")

# 解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模型炸弹人对战")
    parser.add_argument("--config", type=str, default="model_config_example.json", help="模型配置文件路径（JSON格式）")
    parser.add_argument("--episodes", type=int, default=3, help="游戏局数")
    parser.add_argument("--steps", type=int, default=300, help="每局最大步数")
    parser.add_argument("--delay", type=float, default=0.3, help="每步延迟时间(秒)")
    parser.add_argument("--difficulty", type=str, choices=['easy', 'normal'], 
                       default='normal', help="游戏难度: easy, normal")
    
    args = parser.parse_args()
    
    # 转换难度字符串为枚举类型
    difficulty_map = {
        'easy': DifficultyLevel.EASY,
        'normal': DifficultyLevel.NORMAL
    }
    difficulty = difficulty_map.get(args.difficulty, DifficultyLevel.NORMAL)
    
    # 读取配置文件
    try:
        with open(args.config, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        print("配置文件格式示例:")
        print("""
        [
            {
                "api_base": "https://api.example.com",
                "api_key": "your_api_key_1",
                "model": "model_name_1",
                "player_id": 0
            },
            {
                "api_base": "https://api.example.com",
                "api_key": "your_api_key_2",
                "model": "model_name_2",
                "player_id": 1
            }
        ]
        """)
        exit(1)
    
    # 不要在这里设置固定的随机种子，让每次运行都有不同的结果
    # random.seed(42)  # 注释掉这行
    
    # 创建并运行游戏
    game = MultiModelGame(
        model_configs=model_configs,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        delay=args.delay,
        difficulty=difficulty
    )
    
    game.run_game()
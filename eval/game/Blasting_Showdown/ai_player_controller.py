import base64
import json
import os
import time
import requests
import numpy as np
from typing import Dict, List, Any, Tuple
import argparse
import sys

# 导入游戏环境
from bomberman_gym import BombermanEnv, BombermanAction
import re

class AIPlayerController:
    """AI玩家控制器，使用API控制游戏角色"""
    
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base
        self.model_name = model_name
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
        
        # 初始化系统提示词
        self.system_prompts = {
            0: self._create_system_prompt(0, RED=255, BLUE=0, GREEN=0),    # 红色玩家
            1: self._create_system_prompt(1, RED=0, BLUE=255, GREEN=0),    # 蓝色玩家
            2: self._create_system_prompt(2, RED=0, BLUE=0, GREEN=255),    # 绿色玩家
            3: self._create_system_prompt(3, RED=255, BLUE=255, GREEN=0)   # 黄色玩家
        }
        
        # 存储上一步的游戏状态，用于生成摘要
        self.previous_states = {i: None for i in range(4)}
        self.game_history = {i: [] for i in range(4)}
    
    def _create_system_prompt(self, player_id: int, **kwargs) -> str:
        """Create system prompt"""
        color_name = ["Red", "Blue", "Green", "Yellow"][player_id]
        return f"""You are an AI player playing Bomberman game, playing as the {color_name} character (Player {player_id+1}).
    You need to make intelligent decisions based on current game state information, game screen images, and sound events.

    Game Rules:
    1. You can move on the map or place bombs
    2. Bombs create cross-shaped explosions that can destroy soft walls and hit players
    3. Soft walls (brown blocks) have a chance to drop power-ups when destroyed: increase fire power (explosion range), increase bomb count, or improve movement speed
    4. Players hit by flames will be eliminated, your goal is to defeat other players and survive as much as possible
    5. The last surviving player wins

    Map Elements:
    - Empty space: Can move freely
    - Soft walls (brown): Can be destroyed by bombs
    - Hard walls (gray): Cannot be destroyed or passed through
    - Bombs: Will explode after being placed, creating cross-shaped flames
    - Flames: Will damage players and destroy soft walls
    - Power-ups: Enhance player abilities

    After analyzing images, sounds, and game state information, make the best decision:
    1. Move to safe positions, avoid being hit by bombs
    2. Strategically place bombs to destroy soft walls or defeat opponents
    3. Collect valuable power-ups to enhance abilities
    4. Predict opponent actions and react accordingly

    Please return your decision in JSON format:
    For movement: {{"action_type": 0, "target_x": <target x coordinate>, "target_y": <target y coordinate>}}, for example: {{"action_type": 0, "target_x": 1, "target_y": 2}} but please ensure the target coordinates are within map boundaries and do not exceed maximum movement distance.
    For placing bomb: {{"action_type": 1, "target_x": 0, "target_y": 0}}

    Ensure the return format strictly follows requirements, only return one valid JSON object, do not add other explanatory text.
    """
    
    def get_decision(self, player_id: int, obs: Dict) -> Dict:
        """获取AI决策"""
        try:
            # 获取基本信息
            game_image = obs.get('image', '')
            audio_data = obs.get('audio', '')
            
            # 解析音频事件
            audio_events = self._parse_audio_data(audio_data)
            
            # 生成用户提示内容
            user_content = self._create_user_content(player_id, obs, audio_events)
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": self.system_prompts[player_id]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content}
                    ]
                }
            ]
            
            # 添加图像数据
            if game_image:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{game_image}"},
                })
            
            # 添加音频数据 - 优化版本
            if audio_events:
                # 添加总体音频事件说明
                if len(audio_events) > 0:
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"当前回合共有 {len(audio_events)} 个音频事件:"
                    })
                
                # 遍历所有音频事件
                for event_index, event in enumerate(audio_events):
                    player_name = f"玩家{event['player_id'] + 1}"
                    event_type = event['event_type']
                    
                    # 根据事件类型构建友好的中文描述
                    event_description = ""
                    if event_type == 'player_walk':
                        event_description = f"{player_name}移动脚步声"
                    elif event_type == 'bomb_place':
                        event_description = f"{player_name}放置炸弹声"
                    elif event_type == 'bomb_explode':
                        event_description = f"{player_name}的炸弹爆炸声"
                    else:
                        event_description = f"{player_name}的{event_type}声音"
                    
                    # 添加音频描述文本
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"音频{event_index+1}: {event_description}"
                    })
                    
                    # 添加详细信息
                    if 'description' in event:
                        messages[1]["content"].append({
                            "type": "text", 
                            "text": f"详情: {event['description']}"
                        })
                    
                    # 添加实际音频数据
                    audio_b64 = event.get('audio_base64', '')
                    if audio_b64:
                        # 确定音频格式
                        audio_format = "wav"  # 默认格式
                        if "footstep" in event_type or "walk" in event_type:
                            audio_format = "wav"
                        elif "explosion" in event_type or "explode" in event_type:
                            audio_format = "wav"
                        elif "click" in event_type or "place" in event_type:
                            audio_format = "wav"
                        
                        messages[1]["content"].append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        })
            
            # 构建完整请求
            payload = {
                "model": self.model_name,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # 发送API请求
            print(f"正在请求玩家{player_id+1}的决策...")
            r = self.session.post(f"{self.api_base}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            
            response_data = r.json()
            action_text = response_data['choices'][0]['message']['content']
            
            # 解析返回的JSON
            try:
                # 从文本中提取JSON部分
                action_text = action_text.strip()
                if action_text.startswith("```json"):
                    action_text = action_text.split("```json")[1].split("```")[0].strip()
                elif action_text.startswith("```"):
                    action_text = action_text.split("```")[1].split("```")[0].strip()
                
                action = json.loads(action_text)
                
                # 记录到历史
                action_desc = f"move to({action['target_x']},{action['target_y']})" if action['action_type'] == 0 else "放置炸弹"
                self.game_history[player_id].append({
                    'step': obs['step'],
                    'action': action_desc
                })
                
                print(f"玩家{player_id+1}决定: {action_desc}")
                return action
            except json.JSONDecodeError:
                print(f"解析JSON失败，返回默认动作。原始响应: {action_text}")
                # 返回默认动作
                return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}
                
        except Exception as e:
            print(f"玩家{player_id+1}决策出错: {e}")
            # 出错时，返回一个安全的默认动作
            return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}

    # 添加辅助方法
    def _parse_audio_data(self, audio_data: str) -> List[Dict]:
        """解析音频数据"""
        try:
            if not audio_data:
                return []
            return json.loads(audio_data)
        except Exception as e:
            print(f"解析音频数据出错: {e}")
            return []
    
    def _create_user_content(self, player_id: int, obs: Dict, game_events: List[Dict]) -> str:
        """Create user prompt content"""
        # Get player information
        player_info = obs['state']['players'][player_id]
        position_x = player_info['position_x']
        position_y = player_info['position_y']
        
        # Create game events description
        game_events_description = self._format_game_events(game_events)
        
        # Create state changes description (compared to previous step)
        state_changes = self._create_state_changes_description(player_id, obs)
        
        # Other players' positions
        other_players = []
        for pid, p_info in obs['state']['players'].items():
            if int(pid) != player_id and p_info['alive'] == 1:
                other_players.append(f"Player{int(pid)+1}: ({p_info['position_x']}, {p_info['position_y']})")
        
        # Bomb information collection
        bombs = []
        danger_zones = set()  # Store all danger zone coordinates
        for i in range(obs['state']['bombs']['count']):
            x = obs['state']['bombs']['positions_x'][i]
            y = obs['state']['bombs']['positions_y'][i]
            timer = obs['state']['bombs']['countdown'][i]
            owner = obs['state']['bombs']['owner'][i]
            fire = obs['state']['bombs']['fire_power'][i]
            bombs.append(f"Bomb: position({x},{y}), countdown {timer} steps, owner Player{owner+1}, fire power {fire}")
            
            # Calculate danger zones for this bomb
            # Center point
            danger_zones.add((x, y))
            # Four directions
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                for dist in range(1, fire + 1):
                    danger_x = x + dx * dist
                    danger_y = y + dy * dist
                    # Check boundaries
                    if 0 <= danger_x < 13 and 0 <= danger_y < 11:
                        danger_zones.add((danger_x, danger_y))
        
        # Check if player is in danger zone
        player_in_danger = (position_x, position_y) in danger_zones
        
        # Collect nearest bombs
        bombs = []
        for i in range(obs['state']['bombs']['count']):
            x = obs['state']['bombs']['positions_x'][i]
            y = obs['state']['bombs']['positions_y'][i]
            timer = obs['state']['bombs']['countdown'][i]
            owner = obs['state']['bombs']['owner'][i]
            fire = obs['state']['bombs']['fire_power'][i]
            bombs.append(f"Bomb: position({x},{y}), countdown {timer} steps, owner Player{owner+1}, fire power {fire}")
        
        # Create game history summary
        history_summary = ""
        if len(self.game_history[player_id]) > 0:
            recent_history = self.game_history[player_id][-5:]  # Last 5 records
            history_summary = "Recent action history:\n" + "\n".join([f"- Round {h['step']}: {h['action']}" for h in recent_history])
        
        # Calculate current player's maximum movement distance
        max_move_distance = 5 + (player_info['speed'] - 1)  # Base distance 5 + speed bonus
        
        # 构建完整的用户提示
        return f"""Current game state analysis - Round {obs['step']}:

    You are Player{player_id+1}, {"Alive" if player_info['alive'] == 1 else "Dead"}
    Your position: ({position_x}, {position_y})
    Your attributes:
    - Fire power: {player_info['fire_power']} (bomb explosion range)
    - Bomb count: {player_info['bomb_count']} (maximum simultaneous bombs)
    - Currently placed bombs: {player_info['active_bombs']}
    - Movement speed: {player_info['speed']}
    - Trapped status: {"Yes" if player_info['trapped'] == 1 else "No"}

    {"⚠️ WARNING: You are currently in bomb explosion range! Evacuate immediately!" if player_in_danger else ""}

    Movement limitations:
    - Your maximum movement distance is {max_move_distance} tiles (Manhattan distance)
    - This is base distance (5 tiles) plus speed attribute bonus (speed value - 1)
    - You cannot pass through walls or bombs - bombs become obstacles after placement
    - If target position exceeds movement range, you will move to the farthest reachable point

    Other players' positions:
    {chr(10).join(other_players) if other_players else "No other surviving players"}

    Danger zone warnings:
    {chr(10).join([f"⚠️ Bomb at position({x},{y}), {timer} steps until explosion, fire range {fire} tiles, will affect horizontal area from ({x-fire},{y}) to ({x+fire},{y}) and vertical area from ({x},{y-fire}) to ({x},{y+fire})!" 
        for i, (x, y, timer, owner, fire) in enumerate([(obs['state']['bombs']['positions_x'][i], 
                     obs['state']['bombs']['positions_y'][i], 
                     obs['state']['bombs']['countdown'][i],
                     obs['state']['bombs']['owner'][i],
                     obs['state']['bombs']['fire_power'][i]) 
                       for i in range(obs['state']['bombs']['count'])])]) if bombs else "Currently no bomb threats on the field"}

    Important reminder: Bombs become obstacles after placement and cannot be passed through! Consider this when planning routes.

    {state_changes}

    {game_events_description}

    {history_summary}

    Please analyze the attached game screen image and sound events, assess the current situation, and decide whether to move to a safe position, place a bomb, or collect power-ups.
    Prioritize safety! Stay away from bomb explosion zones, especially those with short countdowns.
    Return a JSON action in the correct format, for example {{"action_type": 0, "target_x": 5, "target_y": 3}} to move to position (5,3).
    """
    
    def _format_game_events(self, game_events: List[Dict]) -> str:
        """Format game events description"""
        if not game_events:
            return "No special game events this turn."
        
        events_str = "Game events this turn:"
        for event in game_events:
            event_type = event['event_type']
            player_name = f"Player{event['player_id']+1}"
            
            if event_type == 'player_walk':
                from_pos = event['params']['from_pos']
                to_pos = event['params']['to_pos']
                events_str += f"\n- {player_name} moved from ({from_pos[0]},{from_pos[1]}) to ({to_pos[0]},{to_pos[1]})"
            
            elif event_type == 'bomb_place':
                pos = event['params']['pos']
                fire = event['params']['fire']
                events_str += f"\n- {player_name} placed a bomb with fire power {fire} at ({pos[0]},{pos[1]})"
            
            elif event_type == 'bomb_explode':
                pos = event['params']['pos']
                fire = event['params']['fire']
                affected = len(event['params']['affected_positions'])
                events_str += f"\n- {player_name}'s bomb exploded at ({pos[0]},{pos[1]}) with fire power {fire}, affecting {affected} positions"
        
        return events_str
    
    def _create_state_changes_description(self, player_id: int, obs: Dict) -> str:
        """Create description of state changes compared to previous step"""
        if not self.previous_states[player_id]:
            self.previous_states[player_id] = obs
            return "This is the first round of the game."
        
        prev = self.previous_states[player_id]
        changes = []
        
        # Check player state changes
        curr_player = obs['state']['players'][player_id]
        prev_player = prev['state']['players'][player_id]
        
        if curr_player['fire_power'] > prev_player['fire_power']:
            changes.append(f"Your fire power increased: {prev_player['fire_power']} → {curr_player['fire_power']}")
        
        if curr_player['bomb_count'] > prev_player['bomb_count']:
            changes.append(f"Your bomb count increased: {prev_player['bomb_count']} → {curr_player['bomb_count']}")
        
        if curr_player['speed'] > prev_player['speed']:
            changes.append(f"Your movement speed improved: {prev_player['speed']} → {curr_player['speed']}")
        
        if curr_player['trapped'] > prev_player['trapped']:
            changes.append("You are trapped by bomb flames! Need teammate rescue or will die soon.")
        
        if curr_player['trapped'] < prev_player['trapped'] and prev_player['trapped'] == 1:
            changes.append("You successfully escaped from trapped status!")
        
        # Check bomb count changes
        prev_bombs = prev['state']['bombs']['count']
        curr_bombs = obs['state']['bombs']['count']
        if curr_bombs > prev_bombs:
            changes.append(f"Bomb count on field increased: {prev_bombs} → {curr_bombs}")
        elif curr_bombs < prev_bombs:
            changes.append(f"Bombs exploded, bomb count decreased: {prev_bombs} → {curr_bombs}")
        
        # Check other players' survival status
        for pid, p_info in prev['state']['players'].items():
            pid = int(pid)
            if pid != player_id:
                prev_alive = p_info['alive']
                curr_alive = obs['state']['players'][pid]['alive']
                if prev_alive == 1 and curr_alive == 0:
                    changes.append(f"Player{pid+1} has been eliminated!")
        
        # Update state
        self.previous_states[player_id] = obs
        
        if not changes:
            return "No significant state changes compared to previous step."
        return "State changes:\n- " + "\n- ".join(changes)
    
    def _parse_audio_data(self, audio_data: str) -> List[Dict]:
        """解析音频数据"""
        try:
            if not audio_data:
                return []
            return json.loads(audio_data)
        except Exception as e:
            print(f"解析音频数据出错: {e}")
            return []
    
    def get_decision(self, player_id: int, obs: Dict) -> Dict:
        """获取AI决策"""
        try:
            # 获取基本信息
            game_image = obs.get('image', '')
            audio_data = obs.get('audio', '')
            
            # 解析音频事件
            audio_events = self._parse_audio_data(audio_data)
            
            # 生成用户提示内容
            user_content = self._create_user_content(player_id, obs, audio_events)
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": self.system_prompts[player_id]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content}
                    ]
                }
            ]
            
            # 添加图像数据
            if game_image:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{game_image}"},
                })
            
            # 添加音频数据 - 优化版本
            if audio_events:
                # 添加总体音频事件说明
                if len(audio_events) > 0:
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"当前回合共有 {len(audio_events)} 个音频事件:"
                    })
                
                # 遍历所有音频事件
                for event_index, event in enumerate(audio_events):
                    player_name = f"玩家{event['player_id'] + 1}"
                    event_type = event['event_type']
                    
                    # 根据事件类型构建友好的中文描述
                    event_description = ""
                    if event_type == 'player_walk':
                        event_description = f"{player_name}移动脚步声"
                    elif event_type == 'bomb_place':
                        event_description = f"{player_name}放置炸弹声"
                    elif event_type == 'bomb_explode':
                        event_description = f"{player_name}的炸弹爆炸声"
                    else:
                        event_description = f"{player_name}的{event_type}声音"
                    
                    # 添加音频描述文本
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"音频{event_index+1}: {event_description}"
                    })
                    
                    # 添加详细信息
                    if 'description' in event:
                        messages[1]["content"].append({
                            "type": "text", 
                            "text": f"详情: {event['description']}"
                        })
                    
                    # 添加实际音频数据
                    audio_b64 = event.get('audio_base64', '')
                    if audio_b64:
                        # 确定音频格式
                        audio_format = "wav"  # 默认格式
                        if "footstep" in event_type or "walk" in event_type:
                            audio_format = "wav"
                        elif "explosion" in event_type or "explode" in event_type:
                            audio_format = "wav"
                        elif "click" in event_type or "place" in event_type:
                            audio_format = "wav"
                        
                        messages[1]["content"].append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        })
            
            # 构建完整请求
            payload = {
                "model": self.model_name,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # 发送API请求
            print(f"正在请求玩家{player_id+1}的决策...")
            r = self.session.post(f"{self.api_base}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            
            response_data = r.json()
            action_text = response_data['choices'][0]['message']['content']
            
            # 解析返回的JSON
            try:
                # 从文本中提取JSON部分
                action_text = action_text.strip()
                action_json = None
                
                # 方法1: 检查是否是代码块格式
                if "```json" in action_text:
                    action_json = action_text.split("```json")[1].split("```")[0].strip()
                elif "```" in action_text:
                    action_json = action_text.split("```")[1].split("```")[0].strip()
                
                # 方法2: 直接寻找JSON对象模式 {...}
                if not action_json:
                    json_pattern = re.search(r'\{(?:[^{}]|"[^"]*")*\}', action_text)
                    if json_pattern:
                        action_json = json_pattern.group(0)
                
                # 解析JSON
                if action_json:
                    action = json.loads(action_json)
                else:
                    # 如果没找到格式化的JSON，尝试直接解析整个响应
                    action = json.loads(action_text)
                
                # 验证关键字段
                required_fields = ["action_type", "target_x", "target_y"]
                if not all(field in action for field in required_fields):
                    raise ValueError(f"缺少必要的动作字段: {required_fields}")
                
                # 记录到历史
                action_desc = f"move to({action['target_x']},{action['target_y']})" if action['action_type'] == 0 else "放置炸弹"
                self.game_history[player_id].append({
                    'step': obs['step'],
                    'action': action_desc
                })
                
                print(f"玩家{player_id+1}决定: {action_desc}")
                return action
            except (json.JSONDecodeError, ValueError) as e:
                print(f"解析JSON失败: {e}，返回默认动作。原始响应: {action_text[:100]}...")
                # 返回默认动作
                return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}
        except Exception as e:
            print(f"玩家{player_id+1}决策出错: {e}")
            # 出错时，返回一个安全的默认动作
            return {"action_type": 0, "target_x": obs['state']['players'][player_id]['position_x'], "target_y": obs['state']['players'][player_id]['position_y']}

def run_ai_game(api_base, api_key, model_name, episodes=3, steps_per_episode=300, delay=0.1):
    """运行AI控制的游戏"""
    # 初始化游戏环境
    env = BombermanEnv(render_mode='human')
    
    # 创建AI控制器
    controller = AIPlayerController(api_base, api_key, model_name)
    
    try:
        for episode in range(episodes):
            print(f"\n====== 第 {episode+1} 局游戏 ======")
            obs, info = env.reset()
            
            for step in range(steps_per_episode):
                # 获取所有活跃玩家的决策
                actions = {}
                for player_id in range(env.num_players):
                    # 检查玩家是否存活
                    if obs['state']['players'][player_id]['alive'] == 0:
                        continue
                    
                    # 获取AI决策
                    action = controller.get_decision(player_id, obs)
                    actions[player_id] = action
                
                # 执行动作
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # 在图形界面中添加稍微的延迟以便观察
                time.sleep(delay)
                
                # 检查游戏是否结束
                if terminated:
                    # 找出获胜者
                    winner_id = None
                    for pid, p_info in obs['state']['players'].items():
                        if p_info['alive'] == 1:
                            winner_id = int(pid)
                            break
                    
                    if winner_id is not None:
                        print(f"游戏结束! 玩家{winner_id+1}获胜!")
                    else:
                        print("游戏结束! 平局!")
                    break
                elif truncated:
                    print(f"达到最大步数 {steps_per_episode}!")
                    break
    
    except KeyboardInterrupt:
        print("用户中断，结束游戏")
    except Exception as e:
        print(f"游戏过程中出现错误: {e}")
    finally:
        # 确保正确关闭环境
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI控制炸弹人游戏")
    parser.add_argument("--api-base", type=str, required=True, help="API基础URL")
    parser.add_argument("--api-key", type=str, required=True, help="API密钥")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--episodes", type=int, default=3, help="游戏局数")
    parser.add_argument("--steps", type=int, default=300, help="每局最大步数")
    parser.add_argument("--delay", type=float, default=0.1, help="每步延迟时间(秒)")
    
    args = parser.parse_args()
    
    run_ai_game(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        delay=args.delay
    )

import gym
import numpy as np
import time
import random
from maze_gym_env import MazeGymEnv

def random_agent_demo():
    """使用随机动作演示迷宫Gym环境"""
    # 创建环境，启用语音导航
    env = MazeGymEnv(render_mode="human", voice_guidance=True)
    observation, info = env.reset()
    
    # 检查并显示初始音频数据
    if 'audio' in info:
        print(f"Initial audio guidance available (length: {len(info['audio'])//1000}KB)")
    
    # 跟踪总奖励
    total_reward = 0
    
    # 运行一个回合
    done = False
    while not done:
        # 随机选择动作
        action = env.action_space.sample()
        
        # 可选：限制随机动作的范围，使演示更加平滑
        action[0] = max(-0.5, min(1.0, action[0]))  # 限制前进距离
        action[1] = max(-45, min(45, action[1]))    # 限制旋转角度
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 渲染
        env.render()
        
        # 获取向量状态和屏幕图像
        vector_obs = observation['vector']
        screen_image = observation['screen']
        
        # 显示当前状态
        print(f"动作: 前进={action[0]:.2f}, 旋转={action[1]:.2f}°")
        print(f"位置: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f}), 朝向: {np.degrees(vector_obs[2]):.2f}°")
        print(f"到目标距离: {vector_obs[3]:.2f}, 角度: {np.degrees(vector_obs[4]):.2f}°")
        print(f"屏幕图像形状: {screen_image.shape}")
        print(f"奖励: {reward:.2f}, 总奖励: {total_reward:.2f}")
        
        # 显示音频状态
        if 'audio' in info:
            print(f"语音导航已更新 (数据长度: {len(info['audio'])//1000}KB)")
        
        print(f"目标状态: {'已达到' if terminated else '未达到'}")
        print("-" * 40)
        
        # 控制速度，给语音提示留出足够时间
        time.sleep(0.2)
        
        # 检查是否结束
        done = terminated or truncated
    
    env.close()
    print(f"回合结束! 总奖励: {total_reward:.2f}")

def keyboard_agent_demo():
    """使用键盘控制演示迷宫Gym环境"""
    print("键盘控制演示:")
    print("  W键 - 前进")
    print("  S键 - 后退")
    print("  A/D键 - 左/右旋转")
    print("  H键 - 请求语音指引")
    print("  Q键 - 退出")
    
    import pygame
    
    # 创建环境，启用语音导航
    env = MazeGymEnv(render_mode="human", voice_guidance=True)
    observation, info = env.reset()
    
    # 初始化pygame以获取键盘输入
    pygame.init()
    
    # 跟踪总奖励
    total_reward = 0
    
    # 运行一个回合
    running = True
    done = False
    clock = pygame.time.Clock()
    manual_guidance_cooldown = 0
    
    while running and not done:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_h and manual_guidance_cooldown <= 0:
                    # 手动请求语音导航
                    env._provide_voice_guidance(observation['vector'])
                    manual_guidance_cooldown = 60  # 设置冷却时间
        
        # 减少冷却时间
        if manual_guidance_cooldown > 0:
            manual_guidance_cooldown -= 1
            
        # 获取按键状态
        keys = pygame.key.get_pressed()
        
        # 根据按键生成动作
        forward = 0.0
        rotation = 0.0
        
        if keys[pygame.K_w]:
            forward = 1.0
        if keys[pygame.K_s]:
            forward = -0.5
        if keys[pygame.K_a]:
            rotation = -30.0
        if keys[pygame.K_d]:
            rotation = 30.0
        
        action = np.array([forward, rotation], dtype=np.float32)
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 获取向量状态
        vector_obs = observation['vector']
        
        # 显示当前位置与目标信息
        print(f"\rPos: ({vector_obs[0]:.1f}, {vector_obs[1]:.1f}) | Target: {vector_obs[3]:.1f}m {np.degrees(vector_obs[4]):.0f}° | Reward: {total_reward:.1f}     ", end="")
        
        # 控制速度
        clock.tick(30)
        
        # 检查是否结束
        done = terminated or truncated
    
    print("\n")
    env.close()
    pygame.quit()
    print(f"回合结束! 总奖励: {total_reward:.2f}")

if __name__ == "__main__":
    # 允许用户选择演示模式
    print("请选择演示模式:")
    print("1. 随机动作演示")
    print("2. 键盘控制演示")
    choice = input("> ")
    
    if choice == "1":
        random_agent_demo()
    elif choice == "2":
        keyboard_agent_demo()
    else:
        print("无效选择，默认使用随机动作演示")
        random_agent_demo()

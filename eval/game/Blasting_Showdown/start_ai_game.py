#!/usr/bin/env python
import os
import sys
import argparse
import subprocess

def main():
    """Start AI Bomberman Game"""
    parser = argparse.ArgumentParser(description="Start AI Bomberman Game")
    parser.add_argument("--config", type=str, default="model_config_example.json", 
                      help="Model configuration file path, default is model_config_example.json")
    parser.add_argument("--episodes", type=int, default=3, help="Number of game episodes, default is 3")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per episode, default is 300")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay time per step (seconds), default is 0.5")
    parser.add_argument("--difficulty", type=str, choices=['easy', 'normal'], 
                       default='easy', help="Game difficulty: easy, normal")

    args = parser.parse_args()
    
    # Check if files exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    multi_model_game_path = os.path.join(script_dir, "multi_model_game.py")
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} does not exist")
        return
        
    if not os.path.exists(multi_model_game_path):
        print(f"Error: multi_model_game.py not found. Please ensure this file has been created.")
        return
    
    # Build command
    cmd = [
        sys.executable,
        multi_model_game_path,
        "--config", config_path,
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--delay", str(args.delay),
        "--difficulty", args.difficulty
    ]
    
    print("Starting AI Bomberman Game...")
    print(f"Configuration file: {config_path}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes: {args.episodes}")
    print(f"Maximum steps per episode: {args.steps}")
    print(f"Delay per step: {args.delay} seconds")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nGame interrupted")
    except Exception as e:
        print(f"Runtime error: {e}")

if __name__ == "__main__":
    main()

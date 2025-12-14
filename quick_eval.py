"""Quick eval script - just 2 tasks, 1 episode each for fast verification."""
import sys
from pathlib import Path

# Add LIBERO to path
if Path("/tmp/LIBERO").exists():
    sys.path.insert(0, "/tmp/LIBERO")

import os
import numpy as np
from PIL import Image
import torch

def quick_eval(checkpoint_path=None, n_tasks=2, n_episodes=1):
    """
    Quick eval on first n_tasks of libero_spatial.
    If checkpoint_path is None, uses vanilla (untrained) model.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tinyvla import TinyVLA, ActionEnsemble
    
    # Load model
    print(f"\n{'='*60}")
    if checkpoint_path:
        print(f"Loading TRAINED model from: {checkpoint_path}")
    else:
        print("Loading VANILLA (untrained) model")
    print(f"{'='*60}\n")
    
    vla = TinyVLA(model_name="qwen3-2b")
    if checkpoint_path:
        vla.load(checkpoint_path)
    vla.model.eval()
    
    # Load benchmark
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    max_steps = 220
    
    print(f"Evaluating {n_tasks} tasks, {n_episodes} episodes each\n")
    
    results = {}
    total_success = 0
    total_episodes = 0
    
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)
        
        print(f"Task {task_id}: {task.name[:50]}...")
        print(f"  Instruction: {instruction[:60]}...")
        
        successes = []
        
        for episode in range(min(n_episodes, len(init_states))):
            task_bddl = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(episode)
            env.reset()
            obs = env.set_init_state(init_states[episode])
            
            ensemble = ActionEnsemble(horizon=vla.horizon)
            action_horizon = 5
            action_i = action_horizon
            done = False
            
            for step in range(max_steps):
                if action_i >= action_horizon:
                    img1 = obs['agentview_image'][::-1, ::-1]
                    img2 = obs['robot0_eye_in_hand_image'][::-1, ::-1]
                    
                    pil1 = Image.fromarray(img1).resize((224, 224))
                    pil2 = Image.fromarray(img2).resize((224, 224))
                    
                    with torch.no_grad():
                        actions, text = vla.generate([pil1], [pil2], [instruction])
                    
                    # Print first prediction for debugging
                    if step == 0 and episode == 0:
                        print(f"  First prediction: {text[0][:50]}...")
                    
                    ensemble.add(actions[0])
                    action_i = 0
                
                action = ensemble.get_action()
                action[-1] = 1.0 if action[-1] > 0 else -1.0
                
                obs, reward, done, info = env.step(action.tolist())
                action_i += 1
                
                if done:
                    break
            
            successes.append(float(done))
            env.close()
            print(f"  Episode {episode}: {'SUCCESS' if done else 'FAIL'} (steps: {step+1})")
        
        success_rate = np.mean(successes) * 100
        results[task.name] = success_rate
        total_success += sum(successes)
        total_episodes += len(successes)
        print(f"  Task success: {success_rate:.0f}%\n")
    
    mean_success = total_success / max(total_episodes, 1) * 100
    print(f"{'='*60}")
    print(f"OVERALL SUCCESS RATE: {mean_success:.1f}%")
    print(f"({'VANILLA' if not checkpoint_path else 'TRAINED'} model)")
    print(f"{'='*60}")
    
    return mean_success

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=1)
    args = parser.parse_args()
    
    quick_eval(args.checkpoint, args.n_tasks, args.n_episodes)

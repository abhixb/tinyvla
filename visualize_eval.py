"""Visualize evaluation and save GIFs."""
import sys
from pathlib import Path

if Path("/tmp/LIBERO").exists():
    sys.path.insert(0, "/tmp/LIBERO")

import os
import argparse
import numpy as np
from PIL import Image
import torch
import imageio

def visualize_eval(checkpoint_path, n_tasks=2, n_episodes=1, output_dir="eval_gifs"):
    """Run eval and save GIFs of each episode."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tinyvla import TinyVLA, ActionEnsemble

    Path(output_dir).mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    vla = TinyVLA(model_name="qwen2.5-3b")
    vla.load(checkpoint_path)
    vla.model.eval()

    # Load benchmark
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    max_steps = 220

    print(f"\nRecording {n_tasks} tasks, {n_episodes} episodes each")
    print(f"Saving GIFs to {output_dir}/")
    print(f"Image tiling: {vla.tile_images}\n")

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)

        short_name = task.name[:40]
        print(f"Task {task_id}: {short_name}...")

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
            frames = []

            for step in range(max_steps):
                # Capture frame
                frame = obs['agentview_image'][::-1, ::-1].copy()
                frames.append(frame)

                if action_i >= action_horizon:
                    img1 = obs['agentview_image'][::-1, ::-1]
                    img2 = obs['robot0_eye_in_hand_image'][::-1, ::-1]

                    pil1 = Image.fromarray(img1).resize((224, 224))
                    pil2 = Image.fromarray(img2).resize((224, 224))

                    # Tile images if model expects tiled input
                    if vla.tile_images:
                        tiled = np.concatenate([np.array(pil1), np.array(pil2)], axis=1)
                        input_image = Image.fromarray(tiled)
                    else:
                        input_image = pil1

                    with torch.no_grad():
                        actions, text = vla.generate([input_image], [instruction])

                    # Print VLM output
                    print(f"    Step {step:3d} | VLM: {text[0][:60]}...")
                    print(f"            | Action[0]: [{', '.join(f'{a:.2f}' for a in actions[0][0])}]")

                    ensemble.add(actions[0])
                    action_i = 0

                action = ensemble.get_action()
                action[-1] = 1.0 if action[-1] > 0 else -1.0

                obs, reward, done, info = env.step(action.tolist())
                action_i += 1

                if done:
                    # Add a few more frames to show success
                    for _ in range(10):
                        frames.append(obs['agentview_image'][::-1, ::-1].copy())
                    break

            env.close()

            # Save GIF
            status = "SUCCESS" if done else "FAIL"
            gif_path = f"{output_dir}/task{task_id}_ep{episode}_{status}.gif"

            # Downsample frames for smaller GIF (every 2nd frame)
            frames_down = frames[::2]
            imageio.mimsave(gif_path, frames_down, fps=15, loop=0)

            print(f"  Episode {episode}: {status} ({len(frames)} frames) -> {gif_path}")

    print(f"\nDone! GIFs saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="eval_gifs")
    args = parser.parse_args()

    visualize_eval(args.checkpoint, args.n_tasks, args.n_episodes, args.output_dir)

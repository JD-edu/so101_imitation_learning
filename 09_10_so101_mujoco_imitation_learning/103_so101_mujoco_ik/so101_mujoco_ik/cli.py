from __future__ import annotations

import argparse
import os


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SO-101 MuJoCo + LeRobot ACT pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run one IK expert demonstration")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--headless", action="store_true")
    data = sub.add_parser("generate", help="generate a local LeRobot dataset")
    data.add_argument("--root", default="data/so101_pick_place")
    data.add_argument("--repo-id", default="local/so101_mujoco_pick_place")
    data.add_argument("--episodes", type=int, default=100)
    data.add_argument("--seed", type=int, default=0)
    data.add_argument("--ui", action="store_true")
    train = sub.add_parser("train", help="train LeRobot ACT")
    train.add_argument("--dataset-root", default="data/so101_pick_place")
    train.add_argument("--repo-id", default="local/so101_mujoco_pick_place")
    train.add_argument("--output", default="outputs/act_so101")
    train.add_argument("--steps", type=int, default=100_000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--device")
    infer = sub.add_parser("infer", help="run an ACT checkpoint in MuJoCo")
    infer.add_argument("--checkpoint", default="outputs/act_so101")
    infer.add_argument("--episodes", type=int, default=10)
    infer.add_argument("--seed", type=int, default=1000)
    infer.add_argument("--ui", action="store_true")
    infer.add_argument("--device")
    infer.add_argument("--max-steps", type=int, default=240)
    return p


def main() -> None:
    args = parser().parse_args()
    if not getattr(args, "ui", False) and getattr(args, "headless", True):
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.command == "demo":
        from .env import SO101PickPlaceEnv
        from .expert import IKPickPlaceExpert

        env = SO101PickPlaceEnv(seed=args.seed)
        # Demo trajectory generation only needs state/action values. Creating
        # an offscreen mujoco.Renderer here and a GLFW context immediately
        # afterwards can leave GLFW terminated before replay starts. Dataset
        # generation still captures images through its dedicated path.
        frames = IKPickPlaceExpert(env).generate(capture=False)
        print(f"generated {len(frames)} frames; final block position={env.block_position.round(3)}")
        if not args.headless:
            from .viewer import replay_actions

            env.reset()
            replay_actions(env, [f.action for f in frames])
        env.close()
    elif args.command == "generate":
        from .dataset import create_dataset

        create_dataset(args.root, args.repo_id, args.episodes, args.seed, show_ui=args.ui)
    elif args.command == "train":
        from .train import train_act

        train_act(args.dataset_root, args.repo_id, args.output, args.steps, args.batch_size, args.device)
    elif args.command == "infer":
        from .inference import run_inference

        run_inference(args.checkpoint, args.episodes, args.seed, not args.ui, args.device, args.max_steps)


if __name__ == "__main__":
    main()

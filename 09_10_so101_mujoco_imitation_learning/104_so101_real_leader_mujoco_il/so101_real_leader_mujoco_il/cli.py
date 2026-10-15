from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real SO-101 leader + MuJoCo follower + LeRobot ACT"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="calibrate the hardware leader")
    calibrate.add_argument("--port", default="/dev/ttyUSB0")
    calibrate.add_argument("--baudrate", type=int, default=1_000_000)
    calibrate.add_argument("--xml")
    calibrate.add_argument("--output")

    collect = commands.add_parser("collect", help="record hardware-leader demonstrations")
    collect.add_argument("--dataset-root", required=True)
    collect.add_argument("--repo-id", default="local/so101_real_leader_mujoco")
    collect.add_argument("--leader-port", default="/dev/ttyUSB0")
    collect.add_argument("--calibration")
    collect.add_argument("--xml")
    collect.add_argument("--fps", type=int, default=30)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--recover-empty", action="store_true")
    collect.add_argument(
        "--aruco-state",
        action="store_true",
        help="store detected ArUco geometry as observation.environment_state",
    )
    collect.add_argument("--aruco-id", type=int, default=23)
    collect.add_argument(
        "--assist-view",
        action="store_true",
        help="open an operator-only front-oblique MuJoCo window",
    )

    inspect = commands.add_parser("inspect", help="inspect a recorded episode")
    from .inspect_data import add_arguments

    add_arguments(inspect)

    train = commands.add_parser("train", help="train official LeRobot ACT")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--repo-id", default="local/so101_real_leader_mujoco")
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=100_000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--device")

    infer = commands.add_parser("infer", help="run a saved ACT policy in MuJoCo")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device")
    infer.add_argument("--seed", type=int, default=1000)
    infer.add_argument("--xml")
    infer.add_argument("--aruco-id", type=int, default=23)
    infer.add_argument(
        "--assist-view",
        action="store_true",
        help="open an operator-only front-oblique MuJoCo window",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        from .calibration import LeaderCalibrator, save_calibration
        from .runtime import DEFAULT_CALIBRATION, DEFAULT_XML

        output = args.output or DEFAULT_CALIBRATION
        calibration = LeaderCalibrator(
            args.xml or DEFAULT_XML, args.port, args.baudrate
        ).run()
        backup = save_calibration(calibration, output)
        print(f"calibration saved: {output}")
        if backup is not None:
            print(f"previous calibration backup: {backup}")
    elif args.command == "collect":
        from .dataset import collect_dataset
        from .runtime import DEFAULT_CALIBRATION, DEFAULT_XML

        collect_dataset(
            root=args.dataset_root,
            repo_id=args.repo_id,
            leader_port=args.leader_port,
            calibration_path=args.calibration or DEFAULT_CALIBRATION,
            xml_path=args.xml or DEFAULT_XML,
            fps=args.fps,
            seed=args.seed,
            recover_empty=args.recover_empty,
            aruco_state=args.aruco_state,
            aruco_id=args.aruco_id,
            assist_view=args.assist_view,
        )
    elif args.command == "inspect":
        from .inspect_data import inspect_dataset

        inspect_dataset(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            episode=args.episode,
            delay_ms=args.delay_ms,
        )
    elif args.command == "train":
        from .train import train_act

        train_act(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            output_dir=args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        from .inference import run_inference
        from .runtime import DEFAULT_XML

        run_inference(
            args.checkpoint,
            xml_path=args.xml or DEFAULT_XML,
            device=args.device,
            seed=args.seed,
            aruco_id=args.aruco_id,
            assist_view=args.assist_view,
        )


if __name__ == "__main__":
    main()

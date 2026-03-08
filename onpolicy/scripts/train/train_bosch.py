#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.bosch import BoschEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    """
    Add Bosch-specific arguments on top of the common config.
    """
    parser.add_argument("--num_lines", type=int, default=6)
    parser.add_argument("--num_products", type=int, default=3)
    parser.add_argument("--num_periods", type=int, default=24)
    parser.add_argument("--num_periods", type=int, default=24)
    parser.add_argument("--capacity_per_line", type=float, default=100.0)
    parser.add_argument("--max_lot_size", type=int, default=10)

    parser.add_argument("--holding_cost", type=float, default=1.0)
    parser.add_argument("--backlog_cost", type=float, default=10.0)
    parser.add_argument("--production_cost", type=float, default=1.0)
    parser.add_argument("--setup_cost", type=float, default=2.0)
    parser.add_argument("--pm_cost", type=float, default=20.0)
    parser.add_argument("--cm_cost", type=float, default=40.0)
    parser.add_argument("--alpha_cost_weight", type=float, default=0.1)
    parser.add_argument("--hazard_rate", type=float, default=1e-3)

    # Time-based parameters
    parser.add_argument("--pm_time", type=float, default=0.0)
    parser.add_argument("--cm_time", type=float, default=0.0)

    # Comma-separated lists for per-product parameters
    parser.add_argument(
        "--processing_time",
        type=str,
        default="1.0",
        help="Per-product processing time (hours per unit), comma-separated or scalar.",
    )
    parser.add_argument(
        "--mean_demand",
        type=str,
        default="10.0",
        help="Per-product mean demand per period, comma-separated or scalar.",
    )

    # Optional scalar or matrices for setup and production
    parser.add_argument(
        "--setup_time",
        type=float,
        default=0.0,
        help="Base setup time (hours) for switching between different products.",
    )
    parser.add_argument(
        "--setup_cost_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup cost matrix (row-major).",
    )
    parser.add_argument(
        "--setup_time_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup time matrix (row-major).",
    )
    parser.add_argument(
        "--production_cost_matrix",
        type=str,
        default=None,
        help="Flattened line×product production cost matrix (row-major).",
    )
    parser.add_argument(
        "--eligibility_matrix",
        type=str,
        default=None,
        help="Flattened line×product eligibility matrix (0/1, row-major).",
    )
    parser.add_argument(
        "--max_actions_per_period",
        type=int,
        default=8,
        help="Maximum number of machine micro-actions per period.",
    )

    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Fixed env name for this problem
    all_args.env_name = "BOSCH"

    # For this hierarchical setup we use RMAPPo with separated policies per agent.
    if all_args.algorithm_name == "rmappo":
        print(
            "Using rmappo; setting use_recurrent_policy=True and use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "Using mappo; setting use_recurrent_policy=False and use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("Only rmappo/mappo are supported for BOSCH.")

    # We need different policies per agent; force non-shared policy.
    if all_args.share_policy:
        print("BOSCH env requires non-shared policies; overriding share_policy=False.")
        all_args.share_policy = False

    # Centralized critic by default
    all_args.use_centralized_V = True

    # Align episode_length (max timesteps) with micro-step horizon
    all_args.episode_length = all_args.num_periods * all_args.max_actions_per_period

    # cuda setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb / logging
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            group="bosch_parallel_lines",
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 1 + all_args.num_lines

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # Always use separated runner (one policy per agent)
    from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + "/summary.json")
        )
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])


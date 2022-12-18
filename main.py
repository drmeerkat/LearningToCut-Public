import gym
import os
import torch
import argparse
import numpy as np
import torch.optim as optim

from datetime import datetime
from distutils.util import strtobool

from agent import PPOAgent, train, evaluate
from utils import make_vec_env, generate_wandb_path
from environments import CustomSyncVecEnv

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='learn2cutAgent',
        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=1e-6,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1234,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=150000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, program will run in debug mode and print debug info on screen")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="finalproject",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="orcs4529",
        help="the entity (team) of wandb's project")
    parser.add_argument("--config", type=str, choices=['easy', 'hard', 'custom'], default='easy',
        help="the type of environment being used for training and testing")
    parser.add_argument("--eva-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, will only evaluate the agent given weights loading dirs ")
    parser.add_argument("--load-dir", type=str, default='./saved_weights',
        help="Use with eva-only toggled. This is the directory where all the weights to be evaluated are stored")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=50,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.015,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.log_interval = 5
    if args.debug:
        args.exp_name += '_debug'
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.track:
        import wandb
        wandb.login(key="92768651ba1654eb9429de983b3d41468d1214d6")
        run=wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tags=["training-" + args.config], dir=generate_wandb_path())

    # Set up environments
    # Setup: You may generate your own instances on which you train the cutting agent.
    custom_config = {
        "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
        "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
        "timelimit"       : 50,                             # the maximum horizon length is 50
        "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
    }

    # Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
    easy_config = {
        "load_dir"        : 'instances/train_10_n60_m60',
        "idx_list"        : list(range(10)),
        "timelimit"       : 50,
        "reward_type"     : 'obj'
    }

    # Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
    hard_config = {
        "load_dir"        : 'instances/train_100_n60_m60',
        "idx_list"        : list(range(99)),
        "timelimit"       : 50,
        "reward_type"     : 'obj'
    }

    configs = {
        'easy': easy_config,
        'hard': hard_config,
        'custom': custom_config
    }

    # Based on your environment config setup
    NUM_VARS = 60 + 1
    NUM_CONSTRAINTS = 60

    train_envs = CustomSyncVecEnv([
        make_vec_env(configs[args.config], NUM_VARS, NUM_CONSTRAINTS, args.seed + i) for i in range(args.num_envs)
    ])
    eva_env = make_vec_env(configs[args.config], NUM_VARS, NUM_CONSTRAINTS, args.seed)()

    # Set up agents
    agent = PPOAgent(NUM_VARS, embed_dim=256)
    agent.train()
    agent.to(device)

    if not args.eva_only:
        # Training
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        print(f'Start to train on task setting: {args.config}.')
        train(
            envs=train_envs,
            evaluate_env=eva_env,
            agent=agent,
            args=args,
            device=device,
            timestamp=timestamp,
            optimizer=optimizer,
        )

        print(f'Start to evaluate on task setting: {args.config}.')
        agent.evaluate()
        eva_return = evaluate(
            env=eva_env,
            agent=agent,
            args=args,
            device=device,
            ep=100
        )

        print(f"Evaluation performance under {args.config} config: {eva_return:.3f}")

    else:
        print(f'Start to evaluate on task setting: {args.config}.')
        # easy: scratch/explogs/LearningToCut/checkpoints/20221212_2335_learn2cutAgent_easy/
        # hard: scratch/explogs/LearningToCut/checkpoints/20221213_1353_learn2cutAgent_hard
        wt_dirs = sorted(os.listdir(args.load_dir), key=lambda x:int(x.split('.')[0]))
        if args.track:
            from collections import deque
            eva_returns = deque(maxlen=100)
        for wt in wt_dirs:
            agent.load_state_dict(torch.load(os.path.join(args.load_dir, wt)))
            agent.eval()
            agent.to(device)
            print(f'Start to evaluate weight {wt}.')
            # run 10 episodes for each weight
            for ep in range(10):
                eva_return = evaluate(
                    env=eva_env,
                    agent=agent,
                    args=args,
                    device=device,
                    ep=1
                )

                print(f"Evaluation performance for wt {wt} in ep {ep}: {eva_return:.3f}")
                if args.track:
                    eva_returns.append(eva_return)
                    wandb.log({f"Evaluation Reward ({args.config})": eva_return})
                    if len(eva_returns) == eva_returns.maxlen:
                        wandb.log({f"Evaluation Reward Moving Avg ({args.config})": sum(eva_returns)/eva_returns.maxlen})
                # break

import os
import torch
import gym
import numpy as np

from environments import make_multiple_env, ObsPadding


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def generate_log_path(sub_dir_name):
    proj_name = os.path.split(os.path.split(os.path.abspath(__file__))[0])[1]
    scratch_path = os.path.join(os.path.expanduser('~'), 'scratch', 'explogs')
    sub_dir_path = os.path.join(scratch_path, proj_name, sub_dir_name)
    print(f'Creating log path: {sub_dir_path}')
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    return sub_dir_path


def generate_wandb_path():
    proj_name = os.path.split(os.path.split(os.path.abspath(__file__))[0])[1]
    scratch_path = os.path.join(os.path.expanduser('~'), 'scratch', 'explogs')

    return os.path.join(scratch_path, proj_name)


def make_vec_env(config, n_vars, n_cons, seed):
    def make_vec_env_helper():
        env = make_multiple_env(**config, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ObsPadding(env, n_vars, n_cons, config['timelimit'])
        return env
    
    return make_vec_env_helper
"""
Date: 2025.04.16
Description:
    DTG_IRRL framework algorithm based on a 4 DOF robotic arm.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import time
import math
import yaml
import torch
import socket
import warnings
import webbrowser
import random
warnings.filterwarnings("ignore")

sys.path.append('../../')
import numpy as np
from torch import nn
from tensorboard import program
from stable_baselines3 import PPO
from robot_gym.envs.DTG_IRRL.KirinEnv import KirinEnv 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from robot_gym.envs.DTG_IRRL.Custom import CustomCallback, CustomWrappedEnv, CustomActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# Assign tensorboard port
def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# create tensorboard monitor
def CreateTensorboard(save_path):
    log_dir = save_path + '/tensorboard/'
    port = get_available_port()
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, 
                       '--port', str(port), '--reload_interval', "10.0"])
    url = tb.launch()
    print("[MUJOCO] Tensorboard session created: "+url)
    webbrowser.open_new(url)

    return log_dir


if __name__=="__main__":
    Dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # config file
    yaml_dir = Dirpath + '/robot_gym/envs/DTG_IRRL/config/badm_cfg.yaml'
    with open(yaml_dir,'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_stage = cfg['general']['train_stage']
    algo = cfg['general']['algo']

    # set randmon seed
    seed = int(time.time() * 1e6) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # xml robot model file
    xml_dir = Dirpath + '/resources/KirinApollo_scene.xml'
    monitor_kwargs = {'info_keywords': cfg['environment']['reward_name']}

    # Is multi-threaded training used
    if cfg['general']['multi']:
        n_envs = cfg['general']['multi_env']             
        vec_env_cls = SubprocVecEnv
        print('\n\n*********************   Multi-Threaded Computing  *********************\n\n')
    else:
        n_envs = cfg['general']['n_envs']
        vec_env_cls = DummyVecEnv    
        print('\n\n*********************   Single-Threaded Computing  *********************\n\n')


    # 实例化矢量环境
    vec_env = make_vec_env(
        lambda: CustomWrappedEnv(KirinEnv(xml_dir, yaml_dir, Dirpath, mode='train')), 
        n_envs=n_envs, 
        monitor_kwargs = monitor_kwargs,
        vec_env_cls=vec_env_cls
    )
    
    # Create vector environment
    time_now = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())
    save_path = Dirpath+'/data/' + train_stage + '/' + time_now
    tb_log_dir = CreateTensorboard(save_path)
    callback = CustomCallback(
        verbose = 0,
        log_dir = save_path,
        log_every_episode = cfg['general']['log_every_episode'],
        info_keywords = cfg['environment']['reward_name'],
        video_enable = cfg['general']['video_enable'],
        print_note = '训练一个羽毛球运动控制器',
        filepath = Dirpath
    )
        
    # Imitation training stage
    if train_stage == 'Imitation':
        print('\n\n*********************   MLP Policy  *********************\n\n')      
        # 网络架构
        net_scale = cfg['general']['mlp_net_scale']
        if algo == "PPO":
            print('\n\n*********************  Initial Training  *********************\n\n')      
            # network parameters
            policy_kwargs = dict(
                net_arch = dict(pi=[net_scale, net_scale], vf=[net_scale, net_scale]),
                log_std_init = cfg['general']['log_std_init'],
                log_std_min = cfg['general']['log_std_min'],
            )
            # PPO parameters
            model = PPO(
                policy = CustomActorCriticPolicy, 
                policy_kwargs = policy_kwargs,
                env = vec_env, 
                verbose=1, 
                tensorboard_log= tb_log_dir,
                n_steps = math.ceil(cfg['general']['max_time_env'] / cfg['general']['ctrl_dt'] / cfg['general']['nminibatches']) * cfg['general']['nminibatches'],
                batch_size= math.ceil(cfg['general']['max_time_env'] / cfg['general']['ctrl_dt'] / cfg['general']['nminibatches']) * cfg['general']['n_envs'],
                n_epochs = cfg['general']['n_epochs'],
                gamma = cfg['general']['gamma'],
                gae_lambda = cfg['general']['gae_lambda'],
                learning_rate = cfg['general']['learning_rate'],
                clip_range=0.3,
                max_grad_norm=0.5,
            )

    # Relaxation training stage      
    elif train_stage == 'Relaxation':
        print('\n\n*********************   Continue Learning  *********************\n\n')
        
        # load imitation taining results
        load_dir = '/2025.04.08-16.46.11/'
        load_episode = 16000

        print("Relaxation based results: ", load_dir)

        if algo == "PPO":
            model = PPO.load(Dirpath + '/data/Imitation' + load_dir + str(load_episode))

        learning_rate = cfg['general']['learning_rate']
        model.env = vec_env
        model.tensorboard_log = tb_log_dir
        model.learning_rate = learning_rate
        
        # adjust lr parameters
        model.lr_schedule = get_schedule_fn(learning_rate)
    
    if algo == "PPO":
        model.learn(
            total_timesteps = model.n_envs * model.n_steps * cfg['general']['n_episodes'],
            callback = callback
        )
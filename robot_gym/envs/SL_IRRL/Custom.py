import time
import os
import numpy as np
import pandas as pd
import shutil
import copy
import gymnasium as gym
from stable_baselines3.common.utils import  safe_mean
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from gymnasium.wrappers import RecordVideo



class MyLog:
    def __init__(self,info_keywords):
        self.ep_len_mean = []
        self.ep_rew_mean = []      
        if info_keywords is not None:
            self.ep_info_mean = dict.fromkeys(info_keywords,np.array([]))


# Tensorboard callback function
class SummaryWriterCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.num_envs = 0
    
# Custom callback function
# Store training process data and network, as well as displaying tensor soap data
class CustomCallback(BaseCallback):
    
    def __init__(self, verbose=0, log_dir = None, log_every_episode = 100, info_keywords=None, video_enable=False, print_note ='无', filepath=None):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_every_episode = log_every_episode              # Save once every log_overy_episode episode
        self.iteration = 0
        self.recording = False
        self.info_keywords = info_keywords
        self.MyLog = MyLog(self.info_keywords)
        self.video_enable = video_enable
        self.print_note = print_note
        self.num_envs = 0
        self.reward = {}
        self.filepath = filepath

    def _on_training_start(self):
        env = self.model.get_env()
        self.num_envs = env.num_envs
        self._log_freq = 10  # log every 10 calls
        for name in self.info_keywords:
            self.reward[name] = 0

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self):
        if self.n_calls % self._log_freq == 0:
            infos = self.locals['infos']
            for name in self.reward:
                self.reward[name] = 0
            
            for i in range(self.num_envs-1):
                r = copy.deepcopy(infos[i+1])
                for name in self.info_keywords:
                   self.reward[name] += r[name]
            for name in self.info_keywords:
                self.reward[name] = self.reward[name] / self.num_envs
                self.tb_formatter.writer.add_scalar("rewards/{}".format(name), self.reward[name], self.n_calls)
        return True
    
    def _init_callback(self):      
        time_now = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())
        self.time_begin = time_now
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # config保存
        cfg_path = self.filepath + '/robot_gym/envs/SL_IRRL/config/badm_cfg.yaml'
        # train保存
        train_path = self.filepath + '/robot_gym/scripts/train.py' 
        # env save
        source_path = self.filepath + '/robot_gym/envs/SL_IRRL/'

        print(cfg_path)
        print(train_path)
        # create folder
        if not os.path.exists(self.log_dir + '/config/'):
            os.makedirs(self.log_dir + '/config/')
        
        if not os.path.exists(self.log_dir + '/env/'):
            os.makedirs(self.log_dir + '/env/')

        shutil.copy(cfg_path, self.log_dir + '/config/')
        shutil.copy(train_path, self.log_dir + '/env/')
        shutil.copy(source_path + 'Custom.py', self.log_dir + '/env/')
        shutil.copy(source_path + 'KirinEnv.py', self.log_dir + '/env/')

        self.model.save(self.log_dir + '/0')
  

    def _on_rollout_start(self):
        
        print('save path:   ', self.log_dir)
        print('training info:    ', self.print_note)

        if self.iteration:
            self.MyLog.ep_rew_mean.append(safe_mean([ep_info["r"]/ep_info["l"] for ep_info in self.model.ep_info_buffer]))
            self.MyLog.ep_len_mean.append(safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
            
            if self.info_keywords is not None:
                for key in self.info_keywords:
                    val = safe_mean([ep_info[key] for ep_info in self.model.ep_info_buffer])
                    self.MyLog.ep_info_mean[key] = np.append(self.MyLog.ep_info_mean[key],val)

        self.iteration += 1
        if self.iteration % self.log_every_episode ==0:
            # network save
            self.model.save(self.log_dir + '/' + str(self.iteration))
            
            if self.video_enable:
                self.recording = True
            
            # save training data
            file = self.log_dir + '/data.csv'
            self.MyLog.ep_info_mean.update(
                {
                    'ep_len_mean': self.MyLog.ep_len_mean,
                    'ep_rew_mean': self.MyLog.ep_rew_mean
                }
            )
            data = pd.DataFrame(self.MyLog.ep_info_mean)
            data.to_csv(file)


    def _on_training_end(self):
        # Storage network at the end of training
        self.model.save(self.log_dir + '/final')
        time_now = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())
        
        print('\n\n','Begin Time:',self.time_begin,'\n\n')
        print('\n\n','End Time:',time_now,'\n\n')


# Customized wrapped environment
class CustomWrappedEnv(RecordVideo):
    def __init__(
        self,
        env: gym.Env,
        log_every_episode: int = 100,
    ):

        self.log_every_episode = log_every_episode
        video_folder = '../data'
        RecordVideo.__init__(self, env = env, video_folder = video_folder)
        self.recording = False

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = super().step(action)
        return observations, rewards, terminateds, truncateds, infos
    
    def reset(self, **kwargs):
        ob, info = super().reset(**kwargs)
        return ob, info
    

    def _video_enabled(self):
        return False

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any
from torch import nn
import torch as th

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        log_std_min: float = -1.6,
        log_std_init: float = -0.1,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,      
        *args,
        **kwargs,
    ):
        self.log_std_min = log_std_min

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            log_std_init = log_std_init,
            *args,
            **kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # We performed a clip on log_std and fed it to the sample after the clip

        mean_actions = self.action_net(latent_pi)

        self.log_std_clipped = th.clip(self.log_std,min=self.log_std_min,max=0)


        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std_clipped)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std_clipped, latent_pi)
        else:
            raise ValueError("Invalid action distribution")
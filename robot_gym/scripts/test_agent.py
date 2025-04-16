import sys
sys.path.append('../src')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import torch
import math
import yaml
import random
import warnings
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import matplotlib as mpl
from matplotlib.collections import LineCollection
sys.path.append('../../')
warnings.filterwarnings("ignore")


class Motor:
    def __init__(self):
        self.torque = []
        self.vel = []
        self.low = []
        self.up = []
        pass

class Reward:
    def __init__(self, reward_name):
        self.reward_name = reward_name
        self.reward = {}
        for name in self.reward_name:
            self.reward[name] = []
        pass

class DataStates:
    def __init__(self):
        self.arm_joint_pos = []
        self.arm_joint_vel = []
        self.ball_pos = []
        self.ball_vel = []
        self.end_pos = []
        self.end_vel = []
        self.end_acc = []
        self.pos_mimic = []
        self.vel_mimic = []
        pass

class Data:
    def __init__(self,env):
        self.reset(env)
        pass

    def reset(self,env):
        self.reward_name = ['JointRef', 'JointDotRef', 'TargetPoint', 'Power',
                  'Smooth', 'Smooth_vel', 'Collision', 'Touch', 'Terminated']
        self.joint_name = ['ShoulderRoll', 'ShoulderPitch', 'Elbow', 'WristPitch']
        self.env = env
        self.time = []
        self.motor = [Motor() for _ in range(env.model.nu)]
        self.states = DataStates()
        self.Reward = Reward(self.reward_name)
        self.total_reward = []
        self.ob = []
        self.action = []
        self.ref = []
        self.arm_dof = env.model.nu
        self.action_std = np.array([0.25, 0.25, 0.25, 0.25])
        self.action_mean = np.array([0.0, 2.8, 1.2, 0.0])

    def add_frame(self,env,info,reward,obs,a):
        
        self.time.append(env.data.time)

        mimic_joint = []
        mimic_vel = []
        for i in range(env.model.nu):
            self.motor[i].torque.append(env.Motor.torque_clip[i])
            self.motor[i].vel.append(env.Motor.PID[i].fdb_dot)
            self.motor[i].low.append(env.Motor.low_bound[i])
            self.motor[i].up.append(env.Motor.up_bound[i])
            mimic_joint.append(env.Mimic.joint[i]) 
            mimic_vel.append(env.Mimic.joint_dot[i]) 

        # joint angel ref
        self.states.pos_mimic.append(mimic_joint)
        self.states.vel_mimic.append(mimic_vel)

        # end vel and acc of robotic arm
        vel = env.data.sensor('EndVel').data
        acc = env.data.sensor('EndAcc').data
        self.states.end_vel.append(list(vel))
        self.states.end_acc.append(list(acc))

        # joint angel and angular velocity
        self.states.arm_joint_pos.append(list(env.data.qpos[0:env.model.nu]))
        self.states.arm_joint_vel.append(list(env.data.qvel[0:env.model.nu]))

        # ball pos and vel
        self.states.ball_pos.append(list(env.data.qpos[env.model.nu:(env.model.nu+3)]))
        self.states.ball_vel.append(list(env.data.qvel[env.model.nu:(env.model.nu+3)]))

        # reward
        for name in self.reward_name:
            self.Reward.reward[name].append(info[name])
        self.total_reward.append(reward)

        self.ob.append(obs)
        self.action.append(a)
        ref_tmp = a*self.action_std + self.action_mean
        self.ref.append(ref_tmp)

# -----------------------Visualization----------------------------------------
class Visualization:
    def __init__(self, Data):
        self.Data = Data

        self.params = {
            'text.usetex': True,
            'font.size': 6,
            'font.family': 'Times New Roman',
            'image.cmap': 'YlGnBu',
            'axes.titlesize': 8,
            'legend.fontsize': 6,
            'axes.labelsize': 6,
            'lines.linewidth': 1.5,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'axes.titlepad': 2.0,
            'axes.labelpad': 2.0,
            'xtick.major.pad': 0.5,
            'ytick.major.pad': 0.5,
            'lines.markersize': 2,
            'figure.subplot.wspace': 0.5,
            'figure.subplot.hspace': 0.5,}
        mpl.rcParams.update(self.params)
        pass

    # Joint angle Plot
    def JointStatePlot(self):
        data = self.Data
        JointAngle = np.asarray(data.states.arm_joint_pos)
        JointAngleMimic = np.asarray(data.states.pos_mimic)

        cols = math.ceil(data.arm_dof/2)
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)

        # print(JointAngle.shape,cols)
        
        TitileLabel = data.joint_name
        # print(TitileLabel)
        [axs[math.floor(i/cols)][i%cols].plot(data.time, JointAngle[:, i], label="Real Angle") for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].plot(data.time, JointAngleMimic[:, i], linestyle='--', label="Ref Angle") for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)') for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_ylabel('Angle (rad)') for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_title(TitileLabel[i]) for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].legend() for i in range(data.arm_dof)]
        # plt.legend()

    # Joint angular velocity Plot
    def JointVelPlot(self):
        data = self.Data
        JointVel = np.asarray(data.states.arm_joint_vel)

        cols = math.ceil(data.arm_dof/2)
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)
        
        TitileLabel = data.joint_name
        [axs[math.floor(i/cols)][i%cols].plot(data.time, JointVel[:, i], label="Real Angle Vel") for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)') for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_ylabel('Angle Vel (rad/s)') for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].set_title(TitileLabel[i]) for i in range(data.arm_dof)]
        [axs[math.floor(i/cols)][i%cols].legend() for i in range(data.arm_dof)]

    # Ball state Plot
    def BallStatePlot(self):
        data = self.Data
        BallPos = np.asarray(data.states.ball_pos)
        BallVel = np.asarray(data.states.ball_vel)

        fig, axs = plt.subplots(1, 2, figsize=(3.5, 3.0), dpi=300)
        
        BallLabel = ['x', 'y', 'z']
        YLabel = ['Pos (m)', 'Vel (m/s)']
        [axs[0].plot(data.time, BallPos[:, i], label=BallLabel[i]) for i in range(3)]
        [axs[1].plot(data.time, BallVel[:, i], label=BallLabel[i]) for i in range(3)]
        [axs[i].set_xlabel('time (s)') for i in range(2)]
        [axs[i].set_ylabel(YLabel[i]) for i in range(2)]
        [axs[i].legend() for i in range(2)]

    # Joint torque Plot
    def TorqueCurvePlot(self):
        data = self.Data

        TitleLabel = data.joint_name
        cols = math.ceil(data.arm_dof/2)
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)
        for i in range(data.arm_dof):
            torque = np.asarray(data.motor[i].torque)
            torque_up = np.asarray(data.motor[i].up)
            torque_low = np.asarray(data.motor[i].low)

            axs[math.floor(i/cols)][i%cols].plot(data.time, torque)
            axs[math.floor(i/cols)][i%cols].plot(data.time, torque_up, color='grey',linestyle='--')
            axs[math.floor(i/cols)][i%cols].plot(data.time, torque_low, color='grey',linestyle='--')

            axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)')
            axs[math.floor(i/cols)][i%cols].set_ylabel('Torque (N.m)')
            axs[math.floor(i/cols)][i%cols].set_title(TitleLabel[i])
    
    # Motor External Curve Plot
    def MotorExternalCurve(self):
        env = self.Data.env
        data = self.Data

        TitleLabel = data.joint_name
        cols = math.ceil(data.arm_dof/2)
        cols = 2
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)
        for i in range(4):
            edge_x = [
                [ -env.Motor.max_speed[i], env.Motor.critical_speed[i] ],
                [ env.Motor.critical_speed[i], env.Motor.max_speed[i] ],
                [ env.Motor.max_speed[i],env.Motor.max_speed[i] ],
                [ env.Motor.max_speed[i],-env.Motor.critical_speed[i] ],
                [ -env.Motor.critical_speed[i],-env.Motor.max_speed[i] ],
                [ -env.Motor.max_speed[i],-env.Motor.max_speed[i]]
            ]
            edge_y = [
                [ env.Motor.torque_lim[i],env.Motor.torque_lim[i] ],
                [ env.Motor.torque_lim[i],0 ],
                [ 0,-env.Motor.torque_lim[i] ],
                [ -env.Motor.torque_lim[i],-env.Motor.torque_lim[i] ],
                [ -env.Motor.torque_lim[i], 0],
                [ 0,env.Motor.torque_lim[i]]
            ]
            for j in range(len(edge_x)):
                axs[math.floor(i/cols)][i%cols].plot(edge_x[j],edge_y[j],linestyle='--', color='grey', lw=1)
            
            torque = np.asarray(data.motor[i].torque)
            speed = np.asarray(data.motor[i].vel)

            points = np.array([speed, torque]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc_array = range(np.size(torque)-1)
            lc = LineCollection(segments, array=lc_array)
            axs[math.floor(i/cols)][i%cols].add_collection(lc)
            axs[math.floor(i/cols)][i%cols].set_xlabel('Angular Vel (rad/s)')
            axs[math.floor(i/cols)][i%cols].set_ylabel('Torque (N.m)')
            axs[math.floor(i/cols)][i%cols].set_title(TitleLabel[i])

            pass

    # Reward Plot
    def RewardPlot(self):
        data = self.Data
        reward = data.Reward.reward
        lens = len(data.reward_name)
        cols = math.ceil(lens/2)
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)
        
        TitileLabel = data.reward_name
        [axs[math.floor(i/cols)][i%cols].plot(data.time, reward[data.reward_name[i]]) for i in range(lens)]
        [axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)') for i in range(lens)]
        axs[0][0].set_ylabel('Reward')
        axs[1][0].set_ylabel('Reward')
        [axs[math.floor(i/cols)][i%cols].set_title(TitileLabel[i]) for i in range(lens)]

    # action and ref plot
    def ActionAndRefPlot(self):
        data = self.Data
        action = data.action
        action = np.asarray(action)
        action_1st = action.copy()
        action_2nd = action.copy()
        action_3rd = action.copy()
        ref = data.ref
        ref = np.asarray(ref)

        for i in range(len(data.time)-1):
            action_1st[i] = action[i+1, :] - action[i, :]
        for j in range(len(data.time)-2):
            action_2nd[j] = (action[j+2, :] - action[j+1, :]) - (action[j+1, :] - action[j, :])
        for k in range(len(data.time)-3):
            action_3rd[k] = (action[k+3, :] - action[k+2, :]) - (action[k+2, :] - action[k+1, :]) - \
                            (action[k+2, :] - action[k+1, :]) - (action[k+1, :] - action[k, :])

        lens = data.arm_dof
        cols = math.ceil(lens/2)
        fig, axs = plt.subplots(2, cols, figsize=(3.5, 3.0), dpi=300)
        
        TitileLabel = data.joint_name
        [axs[math.floor(i/cols)][i%cols].plot(data.time, action[:, i], label="action") for i in range(lens)]
        [axs[math.floor(i/2)][i%2].plot(data.time, ref[:, i], linestyle='--', label="Ref") for i in range(lens)]
        [axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)') for i in range(lens)]
        axs[0][0].set_ylabel('action')
        axs[1][0].set_ylabel('action')
        [axs[math.floor(i/cols)][i%cols].set_title(TitileLabel[i]) for i in range(lens)]
        [axs[math.floor(i/cols)][i%cols].legend() for i in range(lens)]

    # State Plot
    def StatePlot(self):
        data = self.Data
        obs = data.ob
        obs = np.asarray(obs)
        lens = len(obs[0])-4
        cols = math.ceil(lens/4)
        fig, axs = plt.subplots(4, cols, figsize=(3.5, 3.0), dpi=300)
        
        TitileLabel = ["Joint1", "Joint2", "Joint3", "Joint4",
                       "JointVel1", "JointVel2", "JointVel3", "JointVel4",
                       "x", "y", "z",
                       "vx", "vy", "vz"]
        [axs[math.floor(i/cols)][i%cols].plot(data.time, obs[:, i]) for i in range(lens)]
        [axs[math.floor(i/cols)][i%cols].set_xlabel('time (s)') for i in range(lens)]
        axs[0][0].set_ylabel('obs')
        axs[1][0].set_ylabel('obs')
        [axs[math.floor(i/cols)][i%cols].set_title(TitileLabel[i]) for i in range(lens)]


# ----------------------仿真-----------------------------------------
def do_sim(env,model,viewer,data, Dirpath, sim_num=1):
    
    flag = 0
    # 设置viewer视角
    viewer.cam.lookat = np.array([0, 0, 1.5])
    viewer.cam.elevation = 0
    viewer.cam.azimuth = 180
    viewer.cam.distance = 5.0

    while viewer.is_running():

        obs, reest_info = env.reset()
        done = False
        a,state = model.predict(obs,deterministic=True)
        a = np.zeros(env.model.nu)

        data.reset(env)
        while not done and viewer.is_running():
            a,state = model.predict(obs,deterministic=True)

            obs, reward, terminated, truncated, info = env.step(a)       
            done = terminated or truncated
            
            viewer.sync()
            data.add_frame(env,info,reward,obs,a)

        print("test terminated: ", terminated, truncated)
        print("i: ", flag)
        env.close()

        if flag == sim_num:
            print(env.data.time)
            break
        flag+=1
    viewer.close()

def main(load_dir, load_episode, stage):

    # 导入测试环境
    Dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(Dirpath+'/data/'+stage+'/'+ load_dir + '/env')
    from KirinEnv import KirinEnv
    from Custom import CustomCallback, CustomWrappedEnv
    xml_dir = Dirpath +'/resources/KirinApollo_scene.xml'
    yaml_dir = os.path.abspath(Dirpath+'/data/'+stage) + '/' + load_dir + '/config/badm_cfg.yaml'
    
    with open(yaml_dir,'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    loaded_model = PPO.load(Dirpath+'/data/' + stage +'/' + load_dir + '/' + str(load_episode))

    # create test env
    data = []
    env = CustomWrappedEnv(KirinEnv(xml_dir, yaml_dir, Dirpath, mode='test'))
    data = Data(env)

    print('\n\n*********************   Loading Successfully  *********************\n\n')

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    # test num
    sim_num = 50
    do_sim(env,loaded_model,viewer,data, Dirpath, sim_num)
    
    # data visualization
    Vis = Visualization(data)
    Vis.BallStatePlot()
    Vis.JointStatePlot()
    Vis.JointVelPlot()
    Vis.MotorExternalCurve()
    Vis.TorqueCurvePlot()
    plt.show()



if __name__ == "__main__":  

    load_dir = '2025.03.11-14.42.49'                # Imitation
    load_dir = '2025.03.12-08.57.29'                # Relaxation: 2025.03.11-14.42.49

    load_episode = 20000
    # stage = 'Imitation'
    stage = 'Relaxation'

    # 随机种子设置
    seed = int(time.time() * 1e6) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    main(load_dir, load_episode, stage)

    pass
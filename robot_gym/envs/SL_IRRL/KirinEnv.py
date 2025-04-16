import sys
import copy
import math
import yaml
import mujoco
import torch
import onnxruntime as ort
import numpy as np
from gymnasium import utils
from collections import deque
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler
from gymnasium.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation as R
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
sys.path.append('../../')

# robot parameters 
class Robot:
    def __init__(self,cfg,noise,model,np_random):
        noise_kinematics_val = 0
        self.Abad_Pitch = cfg['robot']['Abad_Pitch'] * (1 + noise_kinematics_val * np_random.uniform(-1,1))
        self.Pitch_Yaw = cfg['robot']['Pitch_Yaw'] * (1 + noise_kinematics_val * np_random.uniform(-1,1))
        self.Upper = cfg['robot']['Upper'] * (1 + noise_kinematics_val * np_random.uniform(-1,1))
        self.Lower = cfg['robot']['Lower'] * (1 + noise_kinematics_val * np_random.uniform(-1,1))
        self.Racket = cfg['robot']['Racket'] * (1 + noise_kinematics_val * np_random.uniform(-1,1))

        noise_dynamics_val = 0
        self.mass = model.body_mass * (1 + noise_dynamics_val * np_random.uniform(-1,1,size = model.nbody))
        self.inertia = model.body_inertia * (1 + 0.1*noise_dynamics_val * np_random.uniform(-1,1,size = [model.nbody,3]))


# Ball parameters setting
class Ball:
    def __init__(self, np_random):
        self.BallInitPos = np.zeros(3)         # initial pos of ball
        self.BallInitVel = np.zeros(3)         # initial vel of ball
        self.p = 1.2                           # air density                              
        self.CD = 0.67                         # air drag coef
        self.S = 2.827e-3                      # ball parameters
        self.random = np_random
        self.rand_flag = 1
        pass

    # 羽毛球发球机初始状态设置
    def set_ball_initial_pos(self):
        self.CD = 0.67
        vel_init_range_x = np.array([-0.8, 0.8])            # initial vel range in x
        vel_init_range_y = np.array([-10.0, -7.5])          # initial vel range in y
        vel_init_range_z = np.array([6.0, 7.0])             # initial vel range in z

        vel_init_x = vel_init_range_x[0] + self.rand_flag*self.random.uniform(0,1)*\
                     (vel_init_range_x[1]-vel_init_range_x[0])
        vel_init_y = vel_init_range_y[0] + self.rand_flag*self.random.uniform(0,1)*\
                     (vel_init_range_y[1]-vel_init_range_y[0])
        vel_init_z = vel_init_range_z[0] + self.rand_flag*self.random.uniform(0,1)*\
                     (vel_init_range_z[1]-vel_init_range_z[0])
        
        BallInitPos = np.array([-0.13, 4.0, 0.7, 1.0, 0.0, 0.0, 0.0])
        BallInitVel = np.array([vel_init_x, vel_init_y, vel_init_z, 0.0, 0.0, 0.0])

        return BallInitPos, BallInitVel

# sigmoid trajectory fitting parameters calculation
class Trajectory:
    def __init__(self, cfg):
        self.t_sim = 0           
        self.init_t = 0
        self.c = 0.0
        self.a = np.zeros(4)
        self.d = np.zeros(4)
        self.b = 6

    def update(self, reset_flag, dt):
        if reset_flag:
            self.t_sim = np.copy(self.init_t)
            pass
        else:
            self.t_sim = np.copy(self.t_sim + dt)
            pass

    def traj_params_update(self, reset_flag, t_predict, Arm_Init, arm_roll):
        if reset_flag:
            theta_tar = np.array([arm_roll, 2.6, 0.4, 0.0])
            self.c = t_predict - 0.12
            for i in range(len(Arm_Init)):
                if i == 0:
                    self.d[i] = Arm_Init[i]
                else:
                    self.d[i] = Arm_Init[i]

                if i == 1:
                    self.a[i] = 2*(theta_tar[i] - self.d[i])
                else:
                    self.a[i] = 1*(theta_tar[i] - self.d[i])
            pass


# Joint angle and joint angular velocity imitation term
class Mimic:
    def __init__(self,model,Motor,cfg):
        self.joint = np.zeros(model.nu)
        self.joint_last = np.zeros(model.nu)
        self.joint_dot = np.zeros(model.nu)


# Truncated
class Truncated:
    def __init__(self,cfg):
        self.max_time = cfg['environment']['truncated']['max_time']


# PID controller
class PID_Regulator:
    def __init__(self, kp=0, kd=0, ki=0, weigh_kp_max=0, weigh_kd_max=0, weigh_ki_max=0, output_max=0):
        self.ref = 0
        self.fdb = 0
        self.fdb_dot = 0
        self.err = [0,0]
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.kp_raw = kp
        self.kd_raw = kd
        self.ki_raw = ki
        self.weigh_kp = 0
        self.weigh_kd = 0
        self.weigh_ki = 0
        self.weigh_kp_max = weigh_kp_max
        self.weigh_kd_max = weigh_kd_max
        self.weigh_ki_max = weigh_ki_max
        self.output = 0
        self.output_max = output_max
    
    # clip function
    def _val_lim(self, val, max):
        if val>max:
            return max
        else:
            if val<-max:
                return -max
        return val

    def PID_clc(self):
        # torque calcultation based on PID
        self.err[0] = self.err[1]
        self.err[1] = self.ref - self.fdb
        self.weigh_kp = self.kp * self.err[1]
        self.weigh_kd = -self.kd * self.fdb_dot
        self.weigh_ki += self.ki * self.err[1]
        self.output = self.weigh_kp + self.weigh_kd + self.weigh_ki

    def set_noise(self,noise_val,np_random):
        self.output = self.output * (1 + noise_val * np_random.uniform(-1,1))


# motor parameters setting
class Motor:
    def __init__(self,model,cfg):
        self.MotorName = ['ShoulderRoll', 'ShoulderPitch', 'Elbow', 'WristPitch']
        self.PID = [PID_Regulator() for _ in range(model.nu)]
        self.torque = np.zeros(model.nu)
        self.torque_last = np.zeros(model.nu)                  
        self.torque_dot = np.zeros(model.nu)                  
        self.torque_clip = np.zeros(model.nu)                  # torque clip
        self.torque_lim = np.zeros(model.nu)                   # max torque of the motor
        self.low_bound = np.zeros(model.nu)                    # low bound of the torque
        self.up_bound = np.zeros(model.nu)                     # max bound of the torque
        self.max_speed = np.zeros(model.nu)                    # the max speed of the motor speed
        self.critical_speed = np.zeros(model.nu)               # the critical speed of the motor speed
        self.gear = np.zeros(model.nu)                         # gear ratio of the motor
        for name in self.MotorName:
            id = model.actuator(name).id
            self.PID[id] = PID_Regulator(
                    cfg['controller'][name]['kp'],
                    cfg['controller'][name]['kd'],
                    cfg['controller'][name]['ki'],
                    min(np.abs(model.actuator(name).ctrlrange)),
                    min(np.abs(model.actuator(name).ctrlrange)),
                    min(np.abs(model.actuator(name).ctrlrange)),
                    min(np.abs(model.actuator(name).ctrlrange)),
            )
            self.torque_lim[id] = min(np.abs(model.actuator(name).ctrlrange))
            self.gear[id] = model.actuator_gear[id][0]
            self.max_speed[id] = (
                model.actuator_user[id][0]
                / 60 * 2 * math.pi
                / self.gear[id]
            )
            self.critical_speed[id] = (
                model.actuator_user[id][1]
                / 60 * 2 * math.pi
                / self.gear[id]
            )

    # clip of the motor torque
    def clip(self):
        self.torque_clip = np.array(self.torque).copy()
        # Limiting the torque based on the motor force velocity boundary
        for i in range(len(self.torque)):
            low_bound = -self.torque_lim[i]
            up_bound = self.torque_lim[i]
            if self.PID[i].fdb_dot < -self.critical_speed[i]:
                low_bound = min((-self.max_speed[i] - self.PID[i].fdb_dot) / (-self.max_speed[i]+self.critical_speed[i]) * -self.torque_lim[i],0)
            if self.PID[i].fdb_dot > self.critical_speed[i]:
                up_bound = max((self.max_speed[i] - self.PID[i].fdb_dot) / (self.max_speed[i]-self.critical_speed[i]) * self.torque_lim[i],0)
            if self.torque_clip[i]<low_bound:
                self.torque_clip[i]=low_bound           
            if self.torque_clip[i]>up_bound:
                self.torque_clip[i]=up_bound
            self.low_bound[i] = low_bound
            self.up_bound[i] = up_bound
        
    def add_armature(self,model):
        # Calculate the reflection inertia generated by the motor rotor, 
        # which is the product of the rotational inertia of the motor rotor and the square of the reduction ratio
        # unit: kgm^2
        armature = np.zeros(model.nv)
        for name in self.MotorName:
            armature[model.joint(name).id] = self.gear[model.actuator(name).id] ** 2 * model.actuator_user[model.actuator(name).id][2]
        return armature


# 噪音
class Noise:
    def __init__(self,cfg,mode):
        self.enable = cfg['noise']['enable']
        self.delay_val = cfg['noise']['delay_val']
        self.delay_enable = cfg['noise']['delay_enable']
        self.delay = cfg['noise']['delay_val']
        self.angle = cfg['noise']['angle']
        self.ang_vel = cfg['noise']['ang_vel']
        self.ball_pos = cfg['noise']['ball_pos']
        self.ball_vel = cfg['noise']['ball_vel']
        self.torque_val = cfg['noise']['torque_val']
        self.pd_val = cfg['noise']['pd_val']
        self.pid_bias = 0
        self.joint_fric_val = cfg['noise']['joint_fric_val']
        self.joint_damp_val = cfg['noise']['joint_damp_val']

# Randomization of initial position
class InitRand:
    def __init__(self,cfg,mode):
        self.enable = cfg['randomization']['enable']
        self.ball_pos = cfg['randomization']['Ball']['ball_pos']
        self.ball_vel = cfg['randomization']['Ball']['ball_vel']
        self.Cd = cfg['randomization']['Ball']['Cd']
        self.angle = cfg['randomization']['Joint']['angle']


# state and action normalization
class Scale:
    def __init__(self,observation_shape,action_shape,model,Motor,cfg, history_frame):
        
        self.action_b = np.zeros(action_shape)
        self.action_k = np.zeros(action_shape)
        a_k = cfg['scale']['action_std']
        a_b = cfg['scale']['action_mean']

        # observation boundary value
        self.observation_min = np.zeros(observation_shape)
        self.observation_max = np.zeros(observation_shape)
        for i in range(action_shape):
            self.observation_min[i] = model.jnt_range[model.joint(Motor.MotorName[i]).id][0]
            self.observation_max[i] = model.jnt_range[model.joint(Motor.MotorName[i]).id][1]
            self.observation_min[i + action_shape] = -Motor.max_speed[i]* 1.2
            self.observation_max[i + action_shape] = -self.observation_min[i + action_shape] 
            
            # last action
            self.observation_min[i + action_shape*2 + 6*history_frame] = -1.0
            self.observation_max[i + action_shape*2 + 6*history_frame] = 1.0

        # the range of the ball pos and vel
        for i in range(history_frame):
            self.observation_min[(action_shape*2+6*i):(action_shape*2+6*(i+1))] = cfg['scale']['min']['ball_state']
            self.observation_max[(action_shape*2+6*i):(action_shape*2+6*(i+1))] = cfg['scale']['max']['ball_state']

        # observation normalization coef
        self.observation_k = (self.observation_max - self.observation_min) / 2
        self.observation_b = (self.observation_max + self.observation_min) / 2

        # action normalization coef
        for i in range(action_shape):
            self.action_b[i] = a_b[i]
            self.action_k[i] = a_k[i]


# robot env build
class KirinEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self, xml_dir, yaml_dir, Dirpath, mode, **kwargs):
        
        utils.EzPickle.__init__(self, **kwargs)
        
        # config file
        with open(yaml_dir,'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert mode=='train' or mode=='test', \
            "mode should be the string 'train' or 'test' "
        self.Dirpath = Dirpath
        self.mode = mode
        self.max_time = cfg['general']['max_time_env']
        self.net_freq = cfg['general']['net_freq']
        self.train_stage = cfg['general']['train_stage']
        self.Arm_Init = np.array(cfg['robot']['Arm_Init'])
        self.history_frame = cfg['general']['history_frame']            # frame of history obs
        self.arm_len = [cfg['robot']['Pitch_Yaw'], 
                        cfg['robot']['Upper'],
                        cfg['robot']['Lower'],
                        cfg['robot']['Racket']]
        
        self.predict_flag = True                  # Ball State Prediction Flag
        self.collision_flag = True                # Collision penalty Flag
        self.target_flag = False                  # Landing Target area Flag
        self.touch_flag = False                   # racket-ball Flag

        self.motion_time = 0.0                    # ball fly time
        self.arm_roll = 0.0                       # roll angel of arm
        self.ball_traj = []                       # trajectory of the ball
        self.ball_target_state = np.zeros(7)      # ball target state of the ball in plane
        
        # pridiciton network load
        self.session = ort.InferenceSession(self.Dirpath + '/config/' + 'SL_MLP.onnx')

        ## Observations defination: 18D
        # joint angle of robotic arm：4D
        # joint angle vel of robotic arm：4D
        # ball position：3D
        # ball velocity：3D
        # last action：4D
        self.observation_shape = 18
        observation_space = Box(low=-1, high=1, shape=(self.observation_shape,), dtype=np.float64)
        
        self.metadata['render_fps'] = np.round(1.0 / cfg['general']['ctrl_dt'])
        # Please note the relationship between frame_stkip and frame rate. In the XML file, 
        # the simulation time step * frame rate * frame_stkip==1
        MujocoEnv.__init__(
            self,
            model_path = xml_dir,
            frame_skip = round(cfg['general']['ctrl_dt']/cfg['general']['sim_dt']), 
            observation_space=observation_space,
            render_mode = 'rgb_array',
            width = 2400,
            height = 1600,
            camera_name = 'body_track',
            **kwargs,
        )

        ## action defination: 4D
        # joint angle of robot arm：4D
        self.action_shape = 4
        self.action_space = Box(low=-100, high=100, shape=(self.action_shape,), dtype=np.float32)

        # yaml文件和xml文件中的仿真时间步长务必一致
        assert cfg['general']['sim_dt']==self.model.opt.timestep, \
            'sim_dt in cfg.yaml file and timestep in xml file must consistent!'

        self.Noise = Noise(cfg, self.mode)

        self.InitRand = InitRand(cfg, self.mode)

        self.Robot = Robot(cfg,self.Noise, self.model, self.np_random)
        self.model.body_mass = self.Robot.mass
        self.model.body_inertia = self.Robot.inertia

        self.Ball = Ball(self.np_random)

        self.Trajectory = Trajectory(cfg)
              
        self.Truncated = Truncated(cfg)

        self.Motor = Motor(self.model,cfg)
        self.model.dof_armature = self.Motor.add_armature(self.model)

        self.Scale = Scale(self.observation_space.shape[0],self.action_space.shape[0],
                           self.model, self.Motor, cfg, self.history_frame)

        self.Mimic = Mimic(self.model,self.Motor,cfg)

        self.reward_name = cfg['environment']['reward_name']
        self.reward_weigh = cfg['environment']['reward_weigh']
        self.mean_reward = {}
        for name in self.reward_name:
            self.mean_reward[name] = 0

        # the info is reward
        self.info_keywords = self.reward_name

        self.obs_last = np.zeros(self.observation_shape)
        self.a_last = np.zeros(self.action_shape)
        self.a_last_2 = np.zeros(self.action_shape)
        self.a_last_3 = np.zeros(self.action_shape)
        
    # step function defination
    def step(self, a):
        
        # time<0.02 s, ball dynamic simulation, robot arm keep stationary
        while self.data.time <= 0.02:
            self.motion_time = self.data.time
            ball_pos = self.data.qpos[4:7]
            ball_vel = self.data.qvel[4:7]
            ball_state = np.array([ball_pos[0], ball_pos[1], ball_pos[2],
                                   ball_vel[0], ball_vel[1], ball_vel[2], self.motion_time])
            self.ball_traj.append(ball_state.copy())

            # robotic arm remains stationary
            for name in self.Motor.MotorName:
                id = self.model.actuator(name).id
                self.Motor.PID[id].ref = self.Arm_Init[id]        
                self.Motor.PID[id].fdb = self.data.joint(name).qpos[0]
                self.Motor.PID[id].fdb_dot = self.data.joint(name).qvel[0]
                self.Motor.PID[id].PID_clc()

                self.Motor.torque_last[id] = self.Motor.torque[id]
                self.Motor.torque[id] = self.Motor.PID[id].output
            self.Motor.torque_dot = (self.Motor.torque - self.Motor.torque_last) / self.dt

            # torque clip
            self.Motor.clip()

            # add torque noise
            if self.Noise.enable:
                self.Motor.torque_clip += self.Motor.torque_clip*self.np_random.uniform(-self.Noise.torque_val,self.Noise.torque_val,size=4)
                
            # set drag force to the ball
            ball_vel = self.data.qvel[self.action_shape:(self.action_shape+3)]
            self.set_drag_force(ball_vel)

            # do simulation
            self.do_simulation(self.Motor.torque_clip, self.frame_skip)
        if self.predict_flag:
            # Predicting the state of ball based in hitting plane (y=0.25) on historical trajectory
            self.ball_target_state = self._ball_state_prediction()

            # Calculate the joint angles of the robotic arm through IK based on the position of the ball
            self.arm_roll = self._inverse_kinematics_roll(self.ball_target_state[0:3])
            
            # update the parameters of the sigmoid trajectory
            self.Trajectory.traj_params_update(True, self.ball_target_state[-1], 
                                            self.TrajInit, self.arm_roll)
            # generate ref joint angel using sigmoid trajectory
            self._generate_joint_ref(True)

            self.predict_flag = False

        # Action filter and limititation
        a = np.clip(a, -100, 100)
        actons_filtered = self.a_last * 0.2 + a * 0.8
        ref = actons_filtered * self.Scale.action_k + self.Scale.action_b

        # generate ref joint angel using sigmoid trajectory
        self._generate_joint_ref(False)

        # PID control of robot arm
        for name in self.Motor.MotorName:
            id = self.model.actuator(name).id
            self.Motor.PID[id].ref = ref[id]            
            self.Motor.PID[id].fdb = self.data.joint(name).qpos[0]
            self.Motor.PID[id].fdb_dot = self.data.joint(name).qvel[0]
            self.Motor.PID[id].PID_clc()

            self.Motor.torque_last[id] = self.Motor.torque[id]
            self.Motor.torque[id] = self.Motor.PID[id].output
        self.Motor.torque_dot = (self.Motor.torque - self.Motor.torque_last) / self.dt

        # torque clip
        self.Motor.clip()
        # add torque noise
        if self.Noise.enable:
            self.Motor.torque_clip += self.Motor.torque_clip*self.np_random.uniform(-self.Noise.torque_val,self.Noise.torque_val,size=4)
        
        # set drag force to the ball
        ball_vel = self.data.qvel[self.action_shape:(self.action_shape+3)]
        self.set_drag_force(ball_vel)

        # do simulation
        self.do_simulation(self.Motor.torque_clip, self.frame_skip)
        
        # calculate reward
        reward = self._clc_reward(a)

        # get observations
        ob = self._get_obs(False, a)

        # calculate total reward
        total_reward = 0
        for name in self.reward_name:
            total_reward += reward[name] * self.reward_weigh[name]

        # Determine whether the end condition has been met
        Terminated, Truncated, done_reward = self._Is_Terminated()
        reward['Terminated'] = done_reward
        total_reward += reward['Terminated']

        # calculate mean reward
        for name in self.reward_name:
            self.mean_reward[name] = self.reward_weigh[name]*reward[name]
        self.mean_reward['Terminated'] = reward['Terminated']

        info = copy.deepcopy(self.mean_reward)

        self.a_last_2 = self.a_last.copy()
        self.a_last = a

        return ob, total_reward, Terminated, Truncated, info

    # reset model function definition
    def reset_model(self):
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data
        )
        
        # noise reset
        if self.Noise.enable:
            # Delay setting
            self.Noise.delay = self.np_random.integers(low=2,high=self.Noise.delay_val)
            # pd noise
            self.Noise.pid_bias = self.np_random.uniform(
                low = -self.Noise.pd_val,
                high = self.Noise.pd_val,
                size = (3,self.action_shape)
            )
            for i in range(self.action_shape):
                self.Motor.PID[i].kp = self.Motor.PID[i].kp_raw * (1+self.Noise.pid_bias[0,i])
                self.Motor.PID[i].ki = self.Motor.PID[i].ki_raw * (1+self.Noise.pid_bias[1,i])
                self.Motor.PID[i].kd = self.Motor.PID[i].kd_raw * (1+self.Noise.pid_bias[2,i])

            # Joint friction and damping
            for i,name in enumerate(self.Motor.MotorName):
                id = self.model.joint(name).id
                self.model.dof_frictionloss[id] = self.Motor.torque_lim[i]*self.Motor.gear[i]*self.Noise.joint_fric_val*self.np_random.uniform(0,1)
                self.model.dof_damping[id] = self.Noise.joint_damp_val*self.np_random.uniform(0,1)

        self.ball_traj = []
        self.ball_target_state = np.zeros(7)
        if self.Noise.delay_enable:
            self.ball_deque = deque()
            self.ball_delay_index = math.ceil(1/180/self.dt)+1        # 180 is the frequency of motion capture
            for i in range(math.ceil(self.Noise.delay/1000/self.dt)):
                self.ball_deque.append(np.array([10, 10, 10, 0.0, 0.0, 0.0]))
            self.ball_current = np.array([10, 10, 10, 0.0, 0.0, 0.0])

        # Randomization of initial position
        ball_pos, ball_vel = self.Ball.set_ball_initial_pos()
        qpos_init = np.zeros(self.model.nq)
        if self.InitRand.enable:
            qpos_init[0:self.action_shape] = self.Arm_Init + self.InitRand.angle*np.random.uniform(-1,1, size=4)
            qpos_init[self.action_shape:(self.action_shape+7)] = ball_pos + self.InitRand.ball_pos*np.random.uniform(-1,1, size=7)
            self.Ball.CD = self.Ball.CD + self.InitRand.Cd*np.random.uniform(-1, 1)
        else:
            qpos_init[0:self.action_shape] = self.Arm_Init.copy()
            qpos_init[self.action_shape:(self.action_shape+7)] = ball_pos.copy()
        
        qvel_init = np.zeros(self.model.nv)
        qvel_init[self.action_shape:(self.action_shape+6)] = ball_vel.copy()

        qpos_init[(self.action_shape+3):(self.action_shape+7)] = np.array([1, 0, 0, 0])
        self.TrajInit = qpos_init[0:self.action_shape].copy()
        
        self.touch_flag = False
        self.collision_flag = True
        self.target_flag = False
        self.motion_time = 0.0
        self.predict_flag = True

        self.set_state(qpos_init, qvel_init)

        self.a_last = np.zeros(self.action_shape)
        self.a_last_2 = np.zeros(self.action_shape)

        return self._get_obs(True, np.zeros(self.action_shape))
    
    # Obtain observations
    def _get_obs(self, reset_flag, a):
        ob = []
        
        ob_qpos = []
        ob_qvel = []
        for name in self.Motor.MotorName:
            ob_qpos = np.append(ob_qpos,self.data.joint(name).qpos[0])
            ob_qvel = np.append(ob_qvel,self.data.joint(name).qvel[0])
        
        ball_pos= self.data.qpos[self.action_shape:(self.action_shape+3)].copy()
        ball_vel= self.data.qvel[self.action_shape:(self.action_shape+3)].copy()
        ball_state = np.array([ball_pos[0], ball_pos[1], ball_pos[2],
                               ball_vel[0], ball_vel[1], ball_vel[2]])

        if self.Noise.delay_enable:
            # The update frequency of badminton is 180Hz, 
            # which means that the state of the ball is updated every three steps under the control frequency of 500Hz
            if self.ball_delay_index<math.ceil(1/180/self.dt):
                self.ball_delay_index += 1
            else:
                self.ball_current = ball_state.copy()
                self.ball_delay_index = 1

            ball_delay = self.ball_deque.popleft()
            self.ball_deque.append(self.ball_current.copy())
        else:
            ball_delay = ball_state.copy()

        # Joint angles: 4D, index 0-4
        ob = np.append(ob,ob_qpos)
        # Joint vel: 4D, index 4-8
        ob = np.append(ob,ob_qvel)
        # ball pos and vel: 6D, index 8-14
        ob = np.append(ob,ball_delay)
        # last action: dD, index 14-18
        ob = np.append(ob,a)
        
        ob = np.reshape(ob,-1)
        
        # Add noise to the observations
        if self.Noise.enable:
            # robot arm angle noise
            ob[0:self.action_shape] = ob[0:self.action_shape] + self.Noise.angle*self.np_random.uniform(-1,1,size=self.action_shape)
            # robot arm angle vel noise
            ob[self.action_shape:self.action_shape*2] = ob[self.action_shape:self.action_shape*2] + \
                        self.Noise.ang_vel * self.np_random.uniform(-1,1,size=self.action_shape)
            # ball pos noise
            ob[self.action_shape*2:(self.action_shape*2+3)] = ob[self.action_shape*2:(self.action_shape*2+3)] + \
                        self.Noise.ball_pos * self.np_random.uniform(-1,1,size=3)      
            # ball vel noise
            ob[(self.action_shape*2+3):(self.action_shape*2+6)] = ob[(self.action_shape*2+3):(self.action_shape*2+6)] + \
                        self.Noise.ball_vel * self.np_random.uniform(-1,1,size=3)
        
        self.obs_last = ob

        # Limit the amplitude of ob
        ob[ob<self.Scale.observation_min] = self.Scale.observation_min[ob<self.Scale.observation_min]
        ob[ob>self.Scale.observation_max] = self.Scale.observation_max[ob>self.Scale.observation_max]
        # observations normalization
        ob = (ob - self.Scale.observation_b) / self.Scale.observation_k

        return ob
    

    def _clc_reward(self, a):
        # Calculate reward based on current observations and action
        # Reward includes:
        # 1. joint angle mimic
        # 2. joint angle vel mimic
        # 3. racket-ball touch term
        # 4. ball landing targer area term
        # 5. collision penalty
        # 6. power penalty
        # 7. first-order action smooth
        # 8. second-order action smooth

        reward =  dict.fromkeys(self.reward_name,0)

        ############# ToDo #############
        joint_qpos = np.zeros(self.model.nu)
        joint_qvel = np.zeros(self.model.nu)
        for name in self.Motor.MotorName:
            joint_qpos[self.model.actuator(name).id] = self.data.joint(name).qpos[0]
            joint_qvel[self.model.actuator(name).id] = self.data.joint(name).qvel[0]
        
        ball_pos = self.data.qpos[self.action_shape:(self.action_shape+3)].copy()
        ball_vel = self.data.qvel[self.action_shape:(self.action_shape+3)].copy()

        JointLim = np.array([np.pi, 4.8, 2.5, 1.2])
        JointVelLim = np.array([10.6, 10.6, 16.9, 41])
        
        # joint angle mimic and joint angle vel mimic
        reward['JointRef'] = 1.0 * math.exp(-1.0 * np.sum(((joint_qpos - self.Mimic.joint)/JointLim) ** 2))
        reward['JointDotRef'] = 1.0 * math.exp(-1.0 * np.sum(((joint_qvel - self.Mimic.joint_dot)/JointVelLim) ** 2))

        # racket-ball touch term 
        if self.touch_flag==False: 
            if self.data.qvel[self.action_shape+1] > 0.0:
                reward['Touch'] = 1.0
                self.touch_flag = True

        # ball landing targer area term
        if ball_pos[2]<=0.02 and np.abs(ball_vel[2]) < 0.1 and self.target_flag == False:
            if ball_pos[1] <= 5.0 and ball_pos[1] >= 2.0:
                if ball_pos[0] <= 2.0 and ball_pos[0] >= -2.0:
                    reward['TargetPoint'] = 1.0
                    self.target_flag = True
                else:
                    reward['TargetPoint'] = 0.0
            else:
                reward['TargetPoint'] = 0.0
        else:
            reward['TargetPoint'] = 0.0

        # collision penalty
        ball_id = mujoco.mj_name2id(self.model, 5, 'shutt')
        racket_id = mujoco.mj_name2id(self.model, 5, 'racket2')
        ground_id = mujoco.mj_name2id(self.model, 5, 'floor')
        check_collision = [[ground_id, ball_id], [ball_id, ground_id], 
                           [ball_id, racket_id], [racket_id, ball_id]]
        
        collision_pair = self.data.contact.geom
        collision_pair = collision_pair.tolist()

        if self.data.time > 0.001:
            if len(collision_pair) > 1:
                self.collision_flag = True

            elif len(collision_pair) == 1:
                self.collision_flag = True
                for check_pair in check_collision:
                    if check_pair in collision_pair:
                        self.collision_flag = False
            else:
                self.collision_flag = False

            if self.collision_flag:
                reward['Collision'] = -1.0
            else:
                reward['Collision'] = 0.0
        else:
            reward['Collision'] = 0.0
            
        # power
        reward['Power'] = 1.0 * math.exp(-1 * (np.array((self.Motor.torque_clip*joint_qvel)/
                            (self.Motor.torque_lim*JointVelLim))**2).sum())
        
        # first-order action smooth
        reward['Smooth'] = 1.0 * math.exp(-1 * (np.array(a-self.a_last)**2).sum())

        # second-order action smooth
        reward['Smooth_vel'] = 1.0 * math.exp(-0.5 * np.sum((np.array(a-2*self.a_last+self.a_last_2))**2))
        
        return reward
    
    # Predicting the state of ball in hitting plane based on historical trajectory
    def _ball_state_prediction(self):
        Input = np.array(self.ball_traj)
        Input = Input.reshape(1,70)
        Input = torch.tensor(Input, dtype=torch.float32)

        # 运行模型
        onnx_inputs = {'input': Input.numpy()}
        outputs = self.session.run(None, onnx_inputs)

        res = outputs[0][0]
        
        return res
    
    # add air drag to ball
    def set_drag_force(self, ball_vel):
        # drag function
        Fd = -0.5*self.Ball.p*self.Ball.S*self.Ball.CD*np.sqrt(ball_vel[0]**2+ball_vel[1]**2+ball_vel[2]**2)*ball_vel[0:3]
        
        self.data.qfrc_applied[self.action_shape:(self.action_shape+3)] = Fd
    
    # generate ref trajectory
    def _generate_joint_ref(self, reset_flag):
        # Generate joint references based on current state
        self.Trajectory.update(reset_flag, self.dt)
        
        # ref anlge init
        joint_ref = np.zeros(self.model.nu)

        # generate ref angle
        joint_ref =self._generate_joint_trajectory()

        if reset_flag:
            self.Mimic.joint = joint_ref.copy()
            self.Mimic.joint_last = joint_ref.copy()
            self.Mimic.joint_dot = (joint_ref - self.Mimic.joint_last) / self.dt
        else:
            self.Mimic.joint_last = self.Mimic.joint.copy()
            self.Mimic.joint = joint_ref.copy()
            self.Mimic.joint_dot = (self.Mimic.joint  - self.Mimic.joint_last) / self.dt

    # Calculate the angle of the roll
    def _inverse_kinematics_roll(self, ball_pos):
        Pos_tar = ball_pos - np.array([0.0, 0.061, 0.81])
        maxlen = np.sqrt(self.arm_len[0]**2+
                (self.arm_len[1]+self.arm_len[2]+self.arm_len[3])**2)
        l_tar = np.sqrt(np.sum(Pos_tar**2))

        if l_tar>maxlen:
            Pos_tar[0] = Pos_tar[0]*(maxlen - 1e-4)/l_tar
            Pos_tar[1] = Pos_tar[1]*(maxlen - 1e-4)/l_tar
            Pos_tar[2] = Pos_tar[2]*(maxlen - 1e-4)/l_tar

        l_s = self.arm_len[0]
        pos = Pos_tar[0:3]

        tmp = l_s /np.sqrt(pos[0]**2 + pos[2]**2)
        theta_y = -(np.arccos(tmp) - np.arcsin(pos[0]/np.sqrt(pos[0]**2 + pos[2]**2)) - np.pi / 2)

        return theta_y

    # Generate reference trajectory
    def _generate_joint_trajectory(self):

        t = self.Trajectory.t_sim
        a = self.Trajectory.a
        b = self.Trajectory.b
        c = self.Trajectory.c
        d = self.Trajectory.d

        joint_ref = np.zeros(4)
        for i in range(len(self.Arm_Init)):
            joint_ref[i] = a[i]/(1+np.exp(-b*(t-c))) + d[i]
        
        return joint_ref

    # Termination
    def _Is_Terminated(self):
        Terminated = False
        Truncated = False
        Termination_reward = 0

        if self.train_stage == "Relaxation":
            ball_vel = self.data.qvel[self.action_shape:(self.action_shape+3)].copy()
            ball_pos= self.data.qpos[self.action_shape:(self.action_shape+3)].copy()
            if self.touch_flag:
                if ball_vel[1] > 0.0:
                    Termination_reward = 0.0

            # Failed to hit the ball, terminated early
            if ball_pos[1]< -0.5:
                Terminated = True

            # x, y, z of the ball exceeds the threshold, terminate prematurely
            if ball_pos[2] > 3.0 or ball_pos[1] > 6.0 or \
            ball_pos[0] > 2.0 or ball_pos[0] < -2.0:
                Terminated = True

            # Hit the ball and land, but the landing point is not within the designated area, terminated early
            if ball_pos[2] <= 0.02 and np.abs(ball_vel[2]) < 0.1:
                if ball_pos[1] > 6.0 or ball_pos[1] < 3.5:
                    Terminated = True
                if ball_pos[0] > 0.8 or ball_pos[0] < -0.8:
                    Terminated = True

        if Terminated:
            Termination_reward = -1
        else:
            Termination_reward = 0.0

        # reward will be given 0 if the truncation condition is reached due to timeout
        if (self.data.time >= self.Truncated.max_time):
            Truncated = True

        return Terminated, Truncated, Termination_reward    
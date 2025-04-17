# Real-Time Robotic Badminton Striking via Hybrid Supervised and Reinforcement Learning
Codebase for the "Real-Time Robotic Badminton Striking via Hybrid Supervised and Reinforcement Learning" project.This repository contains the code necessary to train a 4DOf robotic arm (named Kirinarm) to learn bandminton striking. 

Authors: Yanyan Yuan, Yucheng Tao, Shaowen Cheng, Yanhong Liang, Yongbin Jin, Hongtao Wang
Website: https://stylite-y.github.io/SL-IRRL-For-Badminton/
Paper: subumitted to Ral

## Installation
1. create a new python virtual env with the python 3.8.10 (recommended):
    ```shell
    mkvirtualenv --python=python3.8.10 myenv
    ```
2. Install mujoco env
    ```shell
    pip install mujoco
    ```
3. Install gymnasium
    ```shell
    pip install gymnasium
    ```
4. Install stable-baselines3, torch
    ```shell
    pip install stable-baselines3==2.3.2
    pip install tensorboard==2.13.0
    pip install torch==2.4.0
    ```
5. Install others
    ```shell
    pip install setuptools==63.2.0
    pip install wheel==0.38.4
    pip install ruamel.yaml==0.18.6
    ```

## CODE STRUCTURE
1. Each environment is defined by an env file (`robot_gym/envs/SL_IRRL/KirinEnv.py`), a config file (`robot_gym/envs/SL_IRRL/config/badm_cfg.py`) and callback file (`robot_gym/envs/SL_IRRL/Custom.py`). The config file contains environment parameters, training parameters and so on.
2. The robotic arm model file is contained in (`resources`) folder.
3. The supervised learning prediction network model is in `config` folder.
4. Training and Test file: `robot_gym/envs/scripts`.

## Usage 
1. Train
```shell
python robot_gym/envs/scripts/train.py
```
- The training and env parameters is in `robot_gym/envs/SL_IRRL/config/badm_cfg.py` file.
- Imitation stage: 
```python
# config.yaml
...
train_stage: 'Imitation'          # train stage: Imitation or Relaxation
n_episodes: 16000                 # maximum number of iterations for training
learning_rate: 0.0001
...
```
- Relaxation stage: 
```python
# config.yaml
...
train_stage: 'Relaxation'          # train stage: Imitation or Relaxation
n_episodes: 20000                 # maximum number of iterations for training
...
```
To train the policy in relaxation stage, the imitation results file shoud also be determined in `train.py`. The training data is saved in `data/Imitation` or `data/Relaxation` folder
```python
# train.py
...
elif train_stage == 'Relaxation':
    print('\n\n*********************   Continue Learning  *********************\n\n')
    
    # load imitation taining results
    load_dir = '/2025.04.08-16.46.11/'
    load_episode = 16000
...
```

## Citation
If you found any part of this code useful, please consider citing:
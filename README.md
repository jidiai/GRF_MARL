

# Google Research Football Multi-Agent Reinforcement Learning Library (GRF MARL Lib)

[![license](https://img.shields.io/badge/license-Apache_v2.0-blue.svg?style=flat)](./LICENSE)
[![Release Version](https://img.shields.io/badge/release-1.1-red.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

This repo provides a simple, distributed and asynchronous multi-agent reinforcement learning framework for the [Google Research Football](https://github.com/google-research/football) environment, along with research tools and results for benchmarking. In particular, it includes:
- A distributed and asynchronous MARL framework 
- Implementation of algorithm IPPO, MAPPO, HAPPO, A2PO, MAT
- Ready-to-run experiment configuration
- Population-based training pipline, such as PSRO and League Training
- Pre-trained GRF policies in both 5-vs-5 and 11-vs-11 full-game scenarios
- Single-step match replay debugger
- [Tutorial for GRF online ranking](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)

Documentation: [grf-marl.readthedocs.io/](https://grf-marl.readthedocs.io/)

Implementation for CDS_QMIX and CDS_QPLEX: [MyCDS benchmark](https://github.com/DiligentPanda/MyCDS/tree/benchmark)

Check out the paper at [Boosting Studies of Multi-Agent Reinforcement Learning on Google Research Football Environment: the Past, Present, and Future
](https://arxiv.org/abs/2309.12951)


----

## Contents
- [Install](#install)
- [Execution](#execution)
- [Cooperative MARL benchmark](#cooperative-marl-benchmark)
- [Population-based self-play training](#population-based-self-play-training)
- [Framework architecture](#framework-architecture)
- [Google Reseach Football Toolkit](#google-reseach-football-toolkit)
- [Pre-trained policies](#pre-trained-policies)
- [Online Ranking](#online-ranking)
- [Contribution](#contribution)
- [Tensorboard tags explained](#tensorboard-tags-explained)
- [Contact](#contact)

----

## Install
You can use any tool to manage your python environment. Here, we use conda as an example.
1. install conda/minconda.
2. `conda create -n light-malib python==3.9` to create a new conda env.
3. activate the env by `conda activate light-malib` when you want to use it or you can add this line to your `.bashrc` file to enable it everytime you login into the bash.
4. Clone the repository and install the required dependencies:
```bash
git clone https://github.com/jidiai/GRF_MARL.git
cd GRF_MARL
pip install -r requirements.txt
```
5. Follow the instructions in the official website https://pytorch.org/get-started/locally/ to install PyTorch (for example, version 1.13.0+cu116).
6. Follow the instructions in the official repo https://github.com/google-research/football and install the Google Research Football environment.

[Return to Contents](#contents)

----

## Execution
After installation, run an example experiment by executing the following command from the home folder:
```python
python3 light_malib/main_pbt.py --config PATH_TO_CONFIG
```
where `PATH_TO_CONFIG` is the relative path of the experiment configuration file.

To run experiments on a small cluster, please follow [ray](https://docs.ray.io/en/latest/ray-core/starting-ray.html)'s official instructions to start a cluster. For example, use `ray start --head` on the master, then connect other machines to the master following the hints from command line output.

[Return to Contents](#contents)

----

## Cooperative MARL Benchmark
We support multiple algorithms on benchmark scenarios.

### [Scenarios](light_malib/envs/gr_football/scenarios/)

- **Pass and shot with keeper (2v1)**: A 3 vs 2 academy game. Two left-team players start at the right half, competing against one right-team defense player and the goalkeeper.
The episode terminates when: a. reaches maximum duration (400 steps); b. ball is out of bounds; c. one team scores; d. ball ownership changes.
- **3 vs 1 with keeper (3v1)**: A 4 vs 2 academy game. Three left-team players start at the right half, competing against one right-team defense player and the goalkeeper.
The same termination condition applies as the pass and shoot with keeper scenario.
- **corner**: An 11 vs 11 academy game. The left team starts the ball at the right team’s corner. The same termination condition applies as the pass and shoot with keeper scenario.
- **counterattack (CT)**: An 11 vs 11 academy game. Four left team players start the ball at the mid-field in the right team’s half and only two right team players defend in their own half. 
The rest of the players are at the left team’s half. The same termination condition applies as the pass and shoot with keeper scenario.
- **5-vs-5 full-game (5v5)**: A 5 vs 5 full-game. Four players from each team gather at the center of the field. The left-team
starts the kick-off. The game terminates when the episode reaches the maximum duration (3,000steps). The second half begins at the 1501st step and two teams will swap sides.
- **11-vs-11 full-game (11v11)**: An 11 vs 11 full-game. The left-team starts the kick-off. The game terminates when the episode
reaches the maximum duration (3,000 steps). The
second half begins at the 1501st step and two
teams will swap sides.

### [Supported algorithms](light_malib/algorithm/)

- **Independent PPO (IPPO)**
- **Multi-Agent PPO (MAPPO)**
- **Heterogeneous-Agent PPO (HAPPO)**
- **Agent-by-agent Policy Optimization (A2PO)**
- **Multi-Agent Transformer (MAT)**

### Experiment configurations
The experiment configurations are listed under [this folder](expr_configs/cooperative_MARL_benchmark).
Your can run an experiment, for example,  by
```python
python3 light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/pass_and_shoot_with_keeper/ippo.yaml
```

[Return to Contents](#contents)

----

## Population-based self-play training 
### [Scenarios](light_malib/envs/gr_football/scenarios/)
- **5-vs-5 full-game (5v5)**
- **11-vs-11 full-game (11v11)**


### [Supported algorithms](light_malib/framework/scheduler/)

- **Policy Space Response Oracle (PSRO)**
- **League Training** 


### Experiment configurations
The experiment configurations are listed under [this folder](expr_configs/population_based_self_play).
Your can run an experiment, for example,  by
```python
python3 light_malib/main_pbt.py --config expr_configs/population_based_self_play/ippo_5v5_hard_psro.yaml
```

### [Pretrained policies](light_malib/trained_models/gr_football/)

We offer some pre-trained policies for study in both 5-vs-5 and 11-vs-11 full-game scenarios. You probably want use them as opponents or for initalization. Please refer to [this section](#pre-trained-policies).

<!-- #### 5-vs-5 full-game
<img src='docs/source/images/radar_5v5.svg' width='400px'>

#### 11-vs-11 full-game
<img src='docs/source/images/radar_11v11.svg' width='400px'> -->


[Return to Contents](#contents)

----

## Framework architecture
<div style="text-align:center">
<img src="docs/source/images/framework.png" width="500" >
</div>

Our framework design draws great inspiration from MALib and RLlib. It has five major components, each serving a specific role:

- **Rollout Manager**: The Rollout Manager establishes multiple parallel rollout workers and delegates
rollout tasks to each worker. Each rollout task includes environment settings, policy distributions for
simulation, and information pertaining to the Episode Server.
- **Training Manager**: The Training Manager sets up multiple distributed trainers and assigns training
tasks to each trainer. Training task descriptions consist of training configurations and details regarding
the Policy and Episode buffers.
- **Data Buffer**: The Data Buffer serves as a repository for episodes and policies. The Episode Server
saves new episodes submitted by the rollout workers, while trainers retrieve sampled episodes from the
Episode Server for training. The Policy Server, on the other hand, stores updated policies submitted
by the Training Manager. Rollout workers subsequently fetch these updated policies from the Policy
Server for simulation.
- **Agent Manager**: The Agent Manager manages a population of policies and their associated data,
which includes pairwise match results and individual rankings.
- **Task Scheduler**: The Task Scheduler is responsible for scheduling and assigning tasks to the Training
Manager and Rollout Manager. In each training generation, it selects an opponent distribution based
on computed statistics retrieved from the Agent Manager.

Beside training against a fixed opponent, Light-MALib also supports population-based training, such as Policy-Space Response Oracle (PSRO). An illustration of a PSRO trial is given as below:
<div style="text-align:center">
<img src="docs/source/images/psro.svg" width="500" >
</div>

[Return to Contents](#contents)

----

## Google Reseach Football Toolkit
Currently, we provide the following tools for better study in the field of Football AI.
### [Google Football Game Graph](light_malib/envs/gr_football/game_graph/)

A data structure representing a game as a tree structure with branching indicating important events like goals or intercepts. See its usage in [README](light_malib/envs/gr_football/game_graph/README.md).

<img src='docs/source/images/grf_data_structure.svg'>

### [Google Football Game Debugger](light_malib/envs/gr_football/debugger/):

A single-step graphical debugger illustrating both 3D and 2D frames with detailed frame data, such as the movements of players and the ball. See its usage in [README](light_malib/envs/gr_football/debugger/README.md).

<img src='docs/source/images/debugger_panels.png' width='600px'>

[Return to Contents](#contents)

----

## [Pre-trained policies](light_malib/trained_models/gr_football/)
At this stage, we release some of our trained model for use as initializations or opponents. 

### 5-vs-5 full-game
<img src='docs/source/images/radar_5v5.svg' width='400px'>

### 11-vs-11 full-game
<img src='docs/source/images/radar_11v11.svg' width='400px'>

[Return to Contents](#contents)

----

## Online Ranking

See [documentation](https://grf-marl.readthedocs.io/).

----

## Contribution

Thanks for your interests! The project is open for contribution. You can either add new environment or algorithm 
to be tested under the framework. 

For new **environment**, feel free to check out this [example](https://github.com/jidiai/ai_lib/tree/V2_mpe/envs).

For new **algorithm**, it needs to be put in the directory `\light_malib\algorithm\` and should include the following components:
1. `loss.py`: given samples, how to compute loss function and performs gradient update;
2. `policy.py`: policy instance for action generation mainly;
3. `trainer.py`: trainer class for data preprocessing;

For each **policy setting** (actor/critic network, feature settings, etc), please check out this [doc](/mappo.md).

----

## Tensorboard tags explained

DataServer:
1. `alive_usage_mean/std`: mean/std usage of data samples in buffer;
2. `mean_wait_time`: total reading waiting time divided reading counts;
3. `sample_per_minute_read`: number of samples read per minute;
4. `sample_per_minute_write`: number of samples written per minute;

PSRO: 
1. `Elo`: Elo-rate during PBT; 
2. `Payoff Table`: plot of payoff table;

Rollout: 
1. `bad_pass,bad_shot,get_intercepted,get_tackled,good_pass,good_shot,interception,num_pass,num_shot,tackle, total_move,total_pass,total_possession,total_shot`: detailed football statistics;
2. `goal_diff`: goal difference of the training agent (positive indicates more goals); 
3. `lose/win`: expected lose/win rate during rollout;
4. `score`: expected scores durig rollout, score for a single game has value 0 if lose, 1 if win and 0.5 if draw;


RolloutTimer
1. `batch`: timer for getting a rollout batch;
2. `env_core_step`: timer for simulator stepping time;
3. `env_step`: total timer for an enviroment step;
4. `feature`: timer for feature encoding;
5. `inference`: timer for policy inference;
6. `policy_update`: timer for pulling policies from remote;
7. `reward`: timer for reward calculation;
8. `rollout`: total timer for one rollout;
9. `sample`: timer for policy sampling;
10. `stats`: timer for collecting statistics;

Training:
1. `Old_V_max/min/mean/std`: value estimate at rollout;
2. `V_max/min/mean/std`: current value estimate;
3. `advantage_max/min/mean/std`: Advantage value;
4. `approx_kl`: KL divergence between old and new action distributions;
5. `clip_ratio`: proportion of clipped entries;
6. `delta_max/min/mean/std`: TD error;
7. `entropy`: entropy value;
8. `imp_weights_max/min/mean/std`: importance weights;
9. `kl_diff`: variation of `approx_kl`;
10. `lower_clip_ratio`: proportion of up-clipping entries;
11. `upper_clip_ratio`: proportion of down-clipping entries;
12. `policy_loss`: policy loss;
14. `training_epoch`: number of training epoch at each iteration;
15. `value_loss`: value loss

TrainingTimer:
1. `compute_return`: timer for GAE compute;
2. `data_copy`: timer for data copy when processing data;
3. `data_generator`: timer for generating data;
4. `loss`: total timer for loss computing;
5. `move_to_gpu`: timer for sending data to GPU;
6. `optimize`: total timer for an optimization step; 
7. `push_policy`: timer for pushing trained policies to the remote;
8. `train_step`: total timer for a training step; 
9. `trainer_data`: timer for get data from `local_queue`;
10. `trainer_optimize`: timer for a optimization step in the trainer;

[Return to Contents](#contents)

----

## Contact
If you have any questions about this repo, feel free to leave an issue. You can also contact current maintainers, [YanSong97](https://github.com/YanSong97) and [DiligentPanda](https://github.com/DiligentPanda), by email.



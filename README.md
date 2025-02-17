```diff
- If you want high sample efficiency, please use qmix_high_sample_efficiency.yaml
- which uses 4 processes for training, slower but higher sample efficiency.
- Performance is *not* comparable of models trained with different number of processes. 
```

# PyMARL2

## Framework


## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

```shell
# For Difficulty-Enhanced Predator-Prey
python3 src/main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

```shell
# For Communication tasks
python3 src/main.py --config=qmix_att --env-config=sc2 with env_args.map_name=1o_10b_vs_1r
```

```shell
# For Google Football (Insufficient testing)
# map_name: academy_counterattack_easy, academy_counterattack_hard, five_vs_five...
python3 src/main.py --config=vdn_gfootball --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

**Run n parallel experiments**

```shell
# bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)
bash run.sh qmix sc2 6h_vs_8z epsilon_anneal_time=500000,td_lambda=0.3 2 0 5
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

# Citation

```
@article{hu2021rethinking,
      title={Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning}, 
      author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
      year={2021},
      eprint={2102.03479},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

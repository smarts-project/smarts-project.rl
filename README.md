# SMARTS Zoo
This repository contains reinforcement learning (RL) environments and models built using [SMARTS](https://github.com/huawei-noah/SMARTS).

## NeurIPS2022
+ The directory [neurips2022](./neurips2022/) contains additional example solutions for Track2 of the Driving SMARTS [competition](https://codalab.lisn.upsaclay.fr/competitions/6618).

## Environment zoo
+ [ULTRA](./ultra/) provides a gym-based environment built upon SMARTS to tackle intersection navigation, specifically the unprotected left turn.

## Agent Zoo
+ [intersection-v0](./intersection-v0)
    + used at: [SMARTS/examples/rl/intersection](https://github.com/huawei-noah/SMARTS/tree/master/examples/rl/intersection)
    + RL library: [StableBaselines3](https://github.com/DLR-RM/stable-baselines3)
    + algorithm: PPO

+ [interaction-aware motion prediction](./interaction_aware_motion_prediction)
    + used at: [SMARTS/zoo/policies/interaction_aware_motion_prediction](https://github.com/huawei-noah/SMARTS/tree/master/zoo/policies/interaction_aware_motion_prediction)

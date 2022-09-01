# Examples : Imitation Learning
1. An example solution for Track-2 offline learning based model development is presented here. This example uses Imitation Learning method.
    + **The policy here has not yet been trained to fully solve the task environments.** 
    + **This example is only meant to demonstrate one potential method of developing an offline learning using Waymo dataset. Here, any offline learning method may be used to develop the policy.**

## Setup
+ Use `python3.8` to develop your model.
    ```bash
    $ cd <path>/track2
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e ./train
    ```

## Notes on the used Observation, Action, and Reward
+ Observations: We use a 3-channel rgb birds eye view image of the form (3, 256, 256) plus oridented and normalized dx & dy between the position of the ego vehicle and the goal location at each time step. dx & dy are calculated by first orienting both the current position and the goal location with respect to the current heading then substracting the oriented current position from the oriented goal location. dx & dy are then normalized using MinMaxScaler whose bound is (-0.1, 0.1).
+ Actions: The action space (output of the policy) is using dx, dy and dh, which are the value change per step in x, y direction and heading for the ego vehicle in its birds eye view image coordinate. dh is normalized by multiplying the values by 100. Since dx and dy can not be directly obtained from smarts observation, we have to get displacement change in global coordinate first and use a rotation matrix w.r.t the heading to get dx, dy. In evaluation, the values of predicted dh need to be divided by 100.
+ Rewards: The reward use the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is not a "goal" concept in the training set, we use the last point of each trajectory as the goal position for training. 

## Train locally
1. Train
    ```bash
    $ python3.8 train.py --dataset_path <path_to_data> \
                        --output_path <path_to_saved_model> \
                        [--cache] False \
                        [--learning_rate] 0.001 \
                        [--save_steps] 10 \
                        [--batch_size] 32 \
                        [--num_epochs] 100 \
    ```
1. First time running `train.py`, please set `cache=False`, the processed data will be saved to `./output/dataset.npy`. For later use, set `cache=True` and it will use the cached dataset.

## Evaluate and visualize a trained model
1. A trained model is assumed to be available in the `track2/submission` folder. 
    + This would be true if the training steps above had been executed previously, as a model would be saved to the `track2/submission` folder at the end of the training. 
1. Execute the following to evaluate a trained model.
    ```bash
    $ cd <path>/track2
    $ python3.8 train/evaluate.py
    ```
    A SUMO GUI will automatically pop up to visualize the evaluation.
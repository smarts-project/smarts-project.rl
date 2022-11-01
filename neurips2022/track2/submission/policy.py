import math
import numpy as np
import os
import pickle
import sys
import torch
from typing import Any, Dict
from pathlib import Path
from torchvision import transforms

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[0]))

from model_IL import MainNet


data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


def to_tensor(bev):

    bev_tensor = torch.from_numpy(bev)
    bev_tensor = torch.squeeze(bev_tensor)
    bev_input = data_transform(bev_tensor)
    bev_input = torch.unsqueeze(bev_input, axis=0)

    return bev_input


def rotate_points_for_decode(x, y, heading):
    if heading < 0:
        angle = 2 * math.pi - heading
    elif heading > 0:
        angle = heading
    else:
        angle = 0

    new_x = (x * math.cos(angle)) - (y * math.sin(angle))
    new_y = (y * math.cos(angle)) + (x * math.sin(angle))

    return -1 * new_x, new_y


def rotate_points_for_encode(x, y, heading):
    if heading < 0:
        angle = abs(heading)
    elif heading > 0:
        angle = 2 * math.pi - heading
    else:
        angle = 0

    new_x = (x * math.cos(angle)) - (y * math.sin(angle))
    new_y = (y * math.cos(angle)) + (x * math.sin(angle))

    return new_x, new_y


class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError


def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].
    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    # Insert wrappers here, if any.
    wrappers = []

    return wrappers


class Policy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        # Load saved model and instantiate any needed objects.
        current_path = str(Path(__file__).parent.resolve())
        self.model = MainNet()
        checkpoint = torch.load(os.path.join(current_path, "model_IL.ckpt"), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.scaler = pickle.load(
            open(os.path.join(current_path, "scaler_IL.pkl"), "rb")
        )

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """

        # Use saved model to predict multi-agent action output given multi-agent SMARTS observation input.
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            bev_obs = np.array([np.moveaxis(agent_obs.top_down_rgb.data, -1, 0)])
            bev_input = to_tensor(bev_obs)

            # current_heading = agent_obs["ego"]["heading"]
            current_heading = float(agent_obs.ego_vehicle_state.heading)
            # goal_x = agent_obs["mission"]["goal_pos"][0]
            goal_x = agent_obs.ego_vehicle_state.mission.goal.position.x
            # goal_y = agent_obs["mission"]["goal_pos"][1]
            goal_y = agent_obs.ego_vehicle_state.mission.goal.position.y
            # current_x = agent_obs["ego"]["pos"][0]
            current_x = agent_obs.ego_vehicle_state.position[0]
            # current_y = agent_obs["ego"]["pos"][1]
            current_y = agent_obs.ego_vehicle_state.position[1]

            goal_x, goal_y = rotate_points_for_encode(goal_x, goal_y, current_heading)
            current_x, current_y = rotate_points_for_encode(
                current_x, current_y, current_heading
            )

            dx_goal = goal_x - current_x
            dy_goal = goal_y - current_y

            dxdy_goal = self.scaler.transform([[dx_goal, dy_goal]])[0]

            goal_input = torch.tensor(np.array(dxdy_goal.tolist()))
            goal_input = torch.unsqueeze(goal_input, axis=0)

            input_x = [bev_input, goal_input]

            actions = self.model(input_x)
            predicted_dx, predicted_dy = rotate_points_for_decode(
                actions["dx"].item(), actions["dy"].item(), current_heading
            )
            predicted_dh = actions["d_heading"].item() / 100

            action = (predicted_dx, predicted_dy, predicted_dh)
            target_pose = np.array(
                [
                    action[0] + agent_obs.ego_vehicle_state.position[0],
                    action[1] + agent_obs.ego_vehicle_state.position[1],
                    action[2] + float(agent_obs.ego_vehicle_state.heading),
                    0.1,
                ]
            )
            wrapped_act.update({agent_id: target_pose})

        return wrapped_act

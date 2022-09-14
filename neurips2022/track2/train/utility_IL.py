import math
import numpy as np
import os
import pickle
import re
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

torch.cuda.empty_cache()


def dheading(heading_l, heading_2):
    return (heading_l - heading_2 + math.pi) % (2 * math.pi) - math.pi


def rotate_points(x, y, heading):
    if heading < 0:
        angle = abs(heading)
    elif heading > 0:
        angle = 2 * math.pi - heading
    else:
        angle = 0

    new_x = (x * math.cos(angle)) - (y * math.sin(angle))
    new_y = (y * math.cos(angle)) + (x * math.sin(angle))

    return new_x, new_y


def load_from_npy(data_path):
    with open(os.path.join(data_path, "dataset.npy"), "rb") as f:
        x = np.load(f, allow_pickle=True)
        y = np.load(f, allow_pickle=True)

    return x, y


def load_data_scratch(dataset_path, save_path):
    obs = []
    actions = []
    regexp_agentid = re.compile(r".*_(.*).png")
    regexp_time = re.compile(r"(.*)_.*")

    scenarios = os.listdir(dataset_path)
    for scenario in scenarios:
        vehicles = set()
        scen_dir = os.path.join(dataset_path, scenario)
        for filename in os.listdir(scen_dir):
            if filename.endswith(".png"):
                match = regexp_agentid.search(filename)
                vehicles.add(match.group(1))

        for vehicle in vehicles:
            with open(os.path.join(scen_dir, f"{vehicle}.pkl"), "rb") as f:
                vehicle_data = pickle.load(f)

            image_names = []
            for filename in os.listdir(scen_dir):
                if filename.endswith("_" + vehicle + ".png"):
                    image_names.append(filename)

            image_names = sorted(image_names)
            prev_dheading = 0
            match = regexp_time.search(image_names[0])
            goal_location = vehicle_data[float(match.group(1))].ego_vehicle_state.mission.goal.position.as_np_array
            for i in range(len(image_names) - 1):
                image = Image.open(os.path.join(scen_dir, image_names[i]))
                obs.append([np.moveaxis(np.asarray(image), -1, 0)])
                match = regexp_time.search(image_names[i])
                sim_time = match.group(1)
                match = regexp_time.search(image_names[i + 1])
                sim_time_next = match.group(1)

                current_position = vehicle_data[float(sim_time)].ego_vehicle_state.position
                next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                current_heading = vehicle_data[float(sim_time)].ego_vehicle_state.heading
                next_heading = vehicle_data[float(sim_time_next)].ego_vehicle_state.heading

                new_current_x, new_current_y = rotate_points(
                    current_position[0], current_position[1], current_heading
                )
                new_next_x, new_next_y = rotate_points(
                    next_position[0], next_position[1], current_heading
                )
                dx = new_next_x - new_current_x
                dy = new_next_y - new_current_y
                d_heading = dheading(current_heading, next_heading) * 100

                if d_heading > 100:
                    d_heading = prev_dheading
                elif d_heading < -100:
                    d_heading = prev_dheading

                prev_dheading = d_heading

                rotated_goal_location_x, rotated_goal_location_y = rotate_points(
                    goal_location[0], goal_location[1], current_heading
                )
                dx_goal = rotated_goal_location_x - new_current_x
                dy_goal = rotated_goal_location_y - new_current_y

                obs[-1].append(np.array([dx_goal, dy_goal]))
                actions.append([dx, dy, d_heading])

    obs = np.array(obs, dtype=object)
    actions = np.array(actions)

    # Normalizing dx and dy.
    moved_obs = np.moveaxis(obs, -1, 0)
    dxdy = []
    for i in range(len(moved_obs[1])):
        dxdy_i = moved_obs[1][i]
        dxdy.append(dxdy_i)
    scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
    scaler.fit(dxdy)
    for i in range(len(obs)):
        obs[i][1] = scaler.transform([dxdy[i].tolist()])[0]

    # Save scaler.
    pickle.dump(scaler, open(os.path.join(save_path, "scaler_IL.pkl"), "wb"))

    print("== Finished loading data ==")
    print("State dimension = ", obs.shape)
    print("Action dimension = ", actions.shape)

    return obs, actions


def load_data(dataset_path, save_path, cache=False):
    if cache is False:
        obs, actions = load_data_scratch(dataset_path, save_path)
        return obs, actions

    else:
        obs, actions = load_from_npy(save_path)
        return obs, actions

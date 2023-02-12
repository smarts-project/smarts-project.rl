import math
from collections import namedtuple

import numpy as np

from .format_obs import formatter


class ObsWrapper:
    """
    Transform the default dict obs to numpy obs.

    Neighbors(relative to ego)(7-dim each):
     - Position
     - Velocityee
     - Bounding Box
     - Heading

    Env(relative to ego)(15-dim each):
     - Position, Heading
     - speed limit
     - width
     - is target lane
     - is goal lane
     - is turnable
     - relative lane index
    """

    def __init__(self, use_fake_goal_lane=False):
        self.preserved_info_single_agent = namedtuple(
            "PreservedInfoSingleAgent",
            [
                "raw_obs",
                "np_obs",
                "lane_index",
                "all_lane_indeces",
                "target_wps",
                "speed_limit",
                "speed",
                "is_on_goal_lane",
                "humaness",
                "classifier_input",
                "is_turnable",
                "goal_lane",
                "wrapped_obs",
            ],
        )
        self.preserved_info = None
        self.target_lane_index = None

        self.neighbor_info_dim = 5
        self.env_info_dim = 25

        self._FAKE_GOAL_LANE_CHG_FREQ = 200
        self._FAKE_GOAL_LANE_COUNTER = 0
        self.fake_goal_lane = 0
        self.use_fake_goal_lane = use_fake_goal_lane

    def reset(self):
        self.target_lane_index = None
        self.fake_goal_lane = -1
        self._FAKE_GOAL_LANE_COUNTER = 0

    def step(self):
        self._FAKE_GOAL_LANE_COUNTER += 1
        self._update_fake_goal_lane()

    def _update_fake_goal_lane(self):
        if self._FAKE_GOAL_LANE_COUNTER == self._FAKE_GOAL_LANE_CHG_FREQ:
            self._FAKE_GOAL_LANE_COUNTER = 0
            self.fake_goal_lane = np.random.randint(0, 4) - 1

    def cal_rel_vel(
        self, v1: float, theta1: float, v2: float, theta2: float
    ) -> np.ndarray:
        """Calculate v1 relative to v2."""
        return np.array(
            [
                -np.sin(theta1) * v1 + np.sin(theta2) * v2,
                np.cos(theta1) * v1 - np.cos(theta2) * v2,
            ]
        )

    def cal_rel_heading(self, heading1: float, heading2: float) -> float:
        """Calculate heading1 relative to heading2."""
        h = heading1 - heading2
        if h > np.pi:
            h -= np.pi
        if h < -np.pi:
            h += np.pi
        return h

    def cal_goal_lane(self, np_obs, raw_obs, lane_index, all_lane_indeces):
        goal_lane = np.zeros((3, 3))
        if self.use_fake_goal_lane:
            if self.fake_goal_lane == -1:
                return goal_lane
            goal_lane[:, 0] = (
                self.fake_goal_lane < lane_index + np.array([-1, 0, 1])
            ).astype(np.float32)
            goal_lane[:, 1] = (
                self.fake_goal_lane == lane_index + np.array([-1, 0, 1])
            ).astype(np.float32)
            goal_lane[:, 2] = (
                self.fake_goal_lane > lane_index + np.array([-1, 0, 1])
            ).astype(np.float32)
            return goal_lane
        if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
            return goal_lane
        cos_thetas = np.zeros(4)
        for i in range(4):
            y1 = (
                np_obs["waypoint_paths"]["pos"][i, 1][:2]
                - np_obs["waypoint_paths"]["pos"][i, 0][:2]
            )
            y2 = (
                np_obs["mission"]["goal_pos"][:2]
                - np_obs["waypoint_paths"]["pos"][i, 0][:2]
            )
            if np.linalg.norm(y2) <= 0.2 or np.linalg.norm(y1) <= 0.2:
                continue
            cos_thetas[i] = abs(y1 @ y2 / np.sqrt(y1 @ y1 * y2 @ y2))
        if cos_thetas.max() > 1 - 0.0001:
            l = all_lane_indeces[cos_thetas.argmax()]
            goal_lane[:, 0] = (l < lane_index + np.array([-1, 0, 1])).astype(np.float32)
            goal_lane[:, 1] = (l == lane_index + np.array([-1, 0, 1])).astype(
                np.float32
            )
            goal_lane[:, 2] = (l > lane_index + np.array([-1, 0, 1])).astype(np.float32)
        return goal_lane

    def observation(self, obs):
        raw_obs, np_obs = obs, formatter(obs)

        # ego_vehicle_state
        pos = np_obs["ego_vehicle_state"]["pos"][:2]
        heading = np_obs["ego_vehicle_state"]["heading"]
        speed = np_obs["ego_vehicle_state"]["speed"]
        lane_index = np_obs["ego_vehicle_state"]["lane_index"]
        jerk_linear = np.linalg.norm(np_obs["ego_vehicle_state"]["linear_jerk"])
        jerk_angular = np.linalg.norm(np_obs["ego_vehicle_state"]["angular_jerk"])
        humaness = jerk_linear + jerk_angular
        rotate_M = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )

        all_lane_indeces = np_obs["waypoint_paths"]["lane_index"][:, 0]
        all_lane_speed_limit = np_obs["waypoint_paths"]["speed_limit"][:, 0].reshape(
            4, 1
        )
        all_lane_width = np_obs["waypoint_paths"]["lane_width"][:, 0].reshape(4, 1)
        all_lane_position = np_obs["waypoint_paths"]["pos"][:, :, :2].reshape(4, 20, 2)
        all_lane_heading = np_obs["waypoint_paths"]["heading"][:, :].reshape(4, 20)

        all_lane_rel_position = (
            (all_lane_position.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T
        ).reshape(4, 20, 2)
        all_lane_rel_heading = all_lane_heading - heading
        all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi
        all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

        if lane_index not in all_lane_indeces:
            lane_index = all_lane_indeces[0]
        if (self.target_lane_index == None) or (
            self.target_lane_index not in all_lane_indeces
        ):
            self.target_lane_index = lane_index

        # Env Info
        st = [0] * 3

        EnvInfo_is_turnable = np.zeros((3, 1))
        if lane_index - 1 in all_lane_indeces:
            EnvInfo_is_turnable[0] = 1.0
            st[0] = np.where(all_lane_indeces == lane_index - 1)[0][0].item()
        if lane_index + 1 in all_lane_indeces:
            EnvInfo_is_turnable[2] = 1.0
            st[2] = np.where(all_lane_indeces == lane_index + 1)[0][0].item()
        EnvInfo_is_turnable[1] = 1.0
        st[1] = np.where(all_lane_indeces == lane_index)[0][0].item()
        speed_limit = all_lane_speed_limit[st[1]]

        EnvInfo_rel_pos_heading = np.zeros((3, 15))
        EnvInfo_rel_pos_heading_classifier = np.zeros((3, 60))
        EnvInfo_speed_limit = np.zeros((3, 1))
        EnvInfo_width = np.zeros((3, 1))
        for i in range(3):
            if (i == 0 and EnvInfo_is_turnable[0] == 0) or (
                i == 2 and EnvInfo_is_turnable[1] == 0
            ):
                continue
            EnvInfo_rel_pos_heading[i, :10] = all_lane_rel_position[st[i]][
                :5, :
            ].reshape(
                10,
            )
            EnvInfo_rel_pos_heading[i, 10:] = all_lane_rel_heading[st[i]][:5].reshape(
                5,
            )
            EnvInfo_rel_pos_heading_classifier[i, :40] = all_lane_rel_position[
                st[i]
            ].reshape(
                40,
            )
            EnvInfo_rel_pos_heading_classifier[i, 40:] = all_lane_rel_heading[
                st[i]
            ].reshape(
                20,
            )
            EnvInfo_speed_limit[i] = all_lane_speed_limit[st[i]]
            EnvInfo_width[i] = all_lane_width[st[i]]

        EnvInfo_is_target = np.zeros((3, 1))
        if self.target_lane_index < lane_index:
            EnvInfo_is_target[0] = 1.0
        elif self.target_lane_index > lane_index:
            EnvInfo_is_target[2] = 1.0
        else:
            EnvInfo_is_target[1] = 1.0

        EnvInfo_is_goal = self.cal_goal_lane(
            np_obs, raw_obs, lane_index, all_lane_indeces
        ).reshape(3, 3)
        is_on_goal_lane = EnvInfo_is_goal[1, 1]

        EnvInfo_index = np.eye(3).reshape(3, 3)

        EnvInfo = np.concatenate(
            [
                EnvInfo_rel_pos_heading,  # 15
                EnvInfo_speed_limit,  # 1
                EnvInfo_width,  # 1
                EnvInfo_is_target,  # 1
                EnvInfo_is_goal,  # 3
                EnvInfo_is_turnable,  # 1
                EnvInfo_index,  # 3
            ],
            -1,
        ).astype(np.float32)

        EnvInfo_classifier = np.concatenate(
            [
                EnvInfo_rel_pos_heading_classifier,  # 60
                EnvInfo_speed_limit,  # 1
                EnvInfo_width,  # 1
                EnvInfo_is_target,  # 1
                EnvInfo_is_goal,  # 3
                EnvInfo_is_turnable,  # 1
                EnvInfo_index,  # 3
            ],
            -1,
        ).astype(np.float32)

        # Neighbor Info

        neighbors_pos = np_obs["neighborhood_vehicle_states"]["pos"][:, :2]
        neighbors_speed = np_obs["neighborhood_vehicle_states"]["speed"]
        neighbors_heading = np_obs["neighborhood_vehicle_states"]["heading"]

        neighbors_rel_vel = np.empty((10, 2))
        neighbors_rel_vel[:, 0] = (
            -np.sin(neighbors_heading) * neighbors_speed + np.sin(heading) * speed
        )
        neighbors_rel_vel[:, 1] = (
            np.cos(neighbors_heading) * neighbors_speed - np.cos(heading) * speed
        )

        nb_mask = np.all(neighbors_pos == 0, -1).reshape(10, 1).astype(np.float32)
        neighbors_pos += nb_mask * 200.0

        neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1))
        st = np.argsort(neighbors_dist)[:5]

        NeighborInfo_rel_pos = (neighbors_pos[st] - pos) @ rotate_M.T
        NeighborInfo_rel_vel = (neighbors_rel_vel[st]) @ rotate_M.T
        NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
        # NeighborInfo_boundingbox = np_obs["neighbors"]["box"][st, :2]

        NeighborInfo = np.concatenate(
            [
                NeighborInfo_rel_pos,  # 2
                NeighborInfo_rel_vel,  # 2
                NeighborInfo_rel_heading,  # 1
                # NeighborInfo_boundingbox,   # 2
            ],
            -1,
        ).astype(np.float32)

        # preserved_info
        wp_mask = (
            np.abs(np.all(np_obs["waypoint_paths"]["pos"][:, :4] == 0, -1) - 1)
            .sum(1)
            .reshape(4, 1)
        )
        target_wps = np_obs["waypoint_paths"]["pos"][:, :4, :].sum(1) / (wp_mask + 1e-8)

        self.preserved_info = self.preserved_info_single_agent(
            raw_obs=raw_obs,
            lane_index=lane_index,
            all_lane_indeces=all_lane_indeces,
            target_wps=target_wps,
            speed_limit=speed_limit,
            np_obs=np_obs,
            is_on_goal_lane=is_on_goal_lane,
            speed=speed,
            humaness=humaness,
            classifier_input=np.concatenate(
                [
                    NeighborInfo.reshape(
                        -1,
                    ),  # (5, 5)
                    EnvInfo_classifier.reshape(
                        -1,
                    ),  # (3, 70)
                ]
            ),  # dim: 235
            is_turnable=EnvInfo_is_turnable,
            goal_lane=EnvInfo_is_goal[1, :],
            wrapped_obs=np.concatenate(
                [
                    NeighborInfo.reshape(
                        -1,
                    ),  # (5, 5)
                    EnvInfo.reshape(
                        -1,
                    ),  # (3, 19)
                ]
            ),
        )

        wrapped_obs = np.concatenate(
            [self.preserved_info.wrapped_obs, self.preserved_info.classifier_input], -1
        )

        return wrapped_obs


class EnvWrapper:
    def __init__(self, obs_wrapper):
        self.obs_wrapper = obs_wrapper
        self.time_cost = -1.0
        self.crash_times = 0
        self._rule_stop_cnt = 0
        self._is_on_goal_lane_last = -1

    def is_in_safe_box(self, turn_left: bool):
        o = self.obs_wrapper.preserved_info.wrapped_obs
        neighbor_x = o[np.arange(5) * 3]
        neighbor_y = o[np.arange(5) * 3 + 1]
        neighbor_h = o[np.arange(5) * 3 + 4]

        for i in range(5):
            if turn_left:
                if (
                    (neighbor_x[i] < -1.6 and neighbor_x[i] > -4.0)
                    and (abs(neighbor_y)[i] < 3.8)
                    and (neighbor_h[i] < 0.05)
                ):
                    return False
            else:
                if (
                    (neighbor_x[i] > 1.6 and neighbor_x[i] < 4.0)
                    and (abs(neighbor_y)[i] < 3.8)
                    and (neighbor_h[i] < 0.05)
                ):
                    return False
        else:
            return True

    def collision_forecast(
        self,
        vehicle_state1,
        vehicle_state2,
        l_front=5,
        l_back=0,
        w_left=1.25,
        w_right=1.25,
        steps=5,
    ):
        v1, v2 = vehicle_state1.speed, vehicle_state2.speed
        theta1, theta2 = (
            vehicle_state1.heading + math.pi / 2,
            vehicle_state2.heading + math.pi / 2,
        )
        v1_vec, v2_vec = v1 * np.array(
            [math.cos(theta1), math.sin(theta1)]
        ), v2 * np.array([math.cos(theta2), math.sin(theta2)])
        init_pos1, init_pos2 = vehicle_state1.position[:2], vehicle_state2.position[:2]
        bound1, bound2 = vehicle_state1.bounding_box, vehicle_state2.bounding_box
        # l1, w1, l2, w2 = bound1.length, bound1.width, bound2.length, bound2.width
        l1, w1, l2, w2 = bound1.length, bound1.width, bound2.length, bound2.width
        l2_vec = l2 / 2 * np.array([math.cos(theta2), math.sin(theta2)])
        w2_vec = w2 / 2 * np.array([math.sin(theta2), -1 * math.cos(theta2)])

        l1_front_vec, l1_back_vec = (l1 / 2 + l_front) * np.array(
            [math.cos(theta1), math.sin(theta1)]
        ), (l1 / 2 + l_back) * np.array([math.cos(theta1), math.sin(theta1)])
        w1_left_vec = (w1 / 2 + w_left) * np.array(
            [math.sin(theta1), -1 * math.cos(theta1)]
        )
        w1_right_vec = (w1 / 2 + w_right) * np.array(
            [math.sin(theta1), -1 * math.cos(theta1)]
        )

        for step in range(0, steps + 1, 1):
            t = 0.1 * step
            pos1, pos2 = init_pos1 + v1_vec * t, init_pos2 + v2_vec * t
            # calculate bounding points
            bps_1 = [
                pos1 + l1_front_vec - w1_left_vec,
                pos1 + l1_front_vec + w1_right_vec,
                pos1 - l1_back_vec - w1_left_vec,
                pos1 - l1_back_vec + w1_right_vec,
            ]
            bps_2 = [
                pos2 + l2_vec + w2_vec,
                pos2 + l2_vec - w2_vec,
                pos2 - l2_vec + w2_vec,
                pos2 - l2_vec - w2_vec,
            ]
            bps_1_front, bps1_right = bps_1[:2], [bps_1[0], bps_1[2]]

            for bp in bps_2:
                if (
                    np.dot(bp - bps_1_front[0], bps_1_front[0] - bps_1_front[1])
                    * np.dot(bp - bps_1_front[1], bps_1_front[0] - bps_1_front[1])
                    <= 0
                    and np.dot(bp - bps1_right[0], bps1_right[0] - bps1_right[1])
                    * np.dot(bp - bps1_right[1], bps1_right[0] - bps1_right[1])
                    <= 0
                ):
                    return True
        return False

    def reset(self):
        self.crash_times = 0
        self._rule_stop_cnt = 0
        self._is_on_goal_lane_last = -1

    def step(self, raw_act):
        wrapped_act = self.pack_action(raw_act)
        return wrapped_act

    def cal_collision_r(self, bounding_box):
        return np.sqrt(bounding_box.length**2 + bounding_box.width**2)

    def cal_distance(self, pos1, pos2):
        return np.sqrt(((pos1 - pos2) ** 2).sum())

    def pack_action(self, action):
        raw_obs = self.obs_wrapper.preserved_info.raw_obs
        all_lane_indeces = self.obs_wrapper.preserved_info.all_lane_indeces

        target_wps = self.obs_wrapper.preserved_info.target_wps
        exp_speed = min(self.obs_wrapper.preserved_info.speed_limit, 13.88)
        speed = self.obs_wrapper.preserved_info.speed
        pos = raw_obs.ego_vehicle_state.position[:2]
        heading = raw_obs.ego_vehicle_state.heading - 0.0
        acc = 0

        # ? Rule-based stopping condition 1
        dist_min = 2
        collision_r = self.cal_collision_r(raw_obs.ego_vehicle_state.bounding_box)
        lane_id = raw_obs.ego_vehicle_state.lane_id
        for neighbor in raw_obs.neighborhood_vehicle_states:
            if (
                neighbor.lane_id == lane_id
                and neighbor.lane_position.s > raw_obs.ego_vehicle_state.lane_position.s
            ):
                neighbor_dists = np.clip(
                    self.cal_distance(pos, neighbor.position[:2])
                    - self.cal_collision_r(neighbor.bounding_box)
                    - collision_r,
                    0,
                    None,
                )
                if neighbor_dists < dist_min:
                    dist_min = neighbor_dists
                    exp_speed = min(neighbor.speed, exp_speed)
                    acc = 1.2
        if dist_min < 0.1:
            action = 9

        # ? Rule-based stopping condition 2
        ego = self.obs_wrapper.preserved_info.raw_obs.ego_vehicle_state
        for (
            neighbor
        ) in self.obs_wrapper.preserved_info.raw_obs.neighborhood_vehicle_states:
            if self.collision_forecast(ego, neighbor):
                self._rule_stop_cnt += 1
                if self._rule_stop_cnt >= 5:
                    self._rule_stop_cnt = 0
                    break
                if action not in [10]:
                    exp_speed = 0.0
                    acc = 1.2
                break
        # ? Rule-based lane changing condition 3
        else:
            goal_lane = self.obs_wrapper.preserved_info.goal_lane
            if (
                self.obs_wrapper.preserved_info.lane_index
                == self.obs_wrapper.target_lane_index
                and action in [7, 8, 9]
            ):
                if goal_lane[2] and self.is_in_safe_box(True):
                    action = 2
                elif goal_lane[0] and self.is_in_safe_box(False):
                    action = 5

        # keep_lane
        if action in [6, 7, 8, 9, 10]:
            exp_speed *= [0.0, 0.4, 0.7, 1.0, -0.2][action - 6]

        # change_lane_left
        elif action in [0, 1, 2]:
            exp_speed *= [0.4, 0.7, 1.0][action - 0]
            if self.obs_wrapper.target_lane_index < all_lane_indeces.max():
                self.obs_wrapper.target_lane_index += 1

        # change_lane_right
        elif action in [3, 4, 5]:
            exp_speed *= [0.4, 0.7, 1.0][action - 3]
            if self.obs_wrapper.target_lane_index > 0:
                self.obs_wrapper.target_lane_index -= 1

        st = np.where(all_lane_indeces == self.obs_wrapper.target_lane_index)[0][
            0
        ].item()
        target_wp = target_wps[st][:2]

        delta_pos = target_wp - pos
        delta_pos_dist = self.cal_distance(target_wp, pos)

        acc = 0.5 if exp_speed > speed else max(0.8, acc)
        # smooth expected speed
        if exp_speed > speed:
            exp_speed = np.clip(exp_speed, None, speed + acc)
        if exp_speed < speed:
            exp_speed = np.clip(exp_speed, speed - acc, None)

        exp_heading = np.arctan2(-delta_pos[0], delta_pos[1])
        hc_id = np.abs(2 * np.pi * (np.arange(3) - 1) + exp_heading - heading).argmin()
        exp_heading += (hc_id - 1) * 2 * np.pi
        heading = np.clip(exp_heading, heading - 0.15, heading + 0.15)
        new_pos = pos + delta_pos / delta_pos_dist * min(exp_speed, 13.88) * 0.1
        wrapped_act = np.concatenate([new_pos, [heading, 0.1]])

        return wrapped_act

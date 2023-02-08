import copy
import logging
import math

import gym
import numpy as np

logger = logging.getLogger(__name__)

"""
1. Add s_OGM, s_AGM, s_rel_goal_lane
2. Add rule-based stop trigger when being too close to other cars.
3. Add lane-changing penalty to decrease meaningless lane changing.
4. Delete s_rel_goal_pos
"""
# TODO: decrease perception range
# TODO: complex structure: mix & dueling


def collision_forecast(
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
    v1_vec, v2_vec = v1 * np.array([math.cos(theta1), math.sin(theta1)]), v2 * np.array(
        [math.cos(theta2), math.sin(theta2)]
    )
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
                # if np.linalg.norm(pos1 - pos2) < 5:
        #     logger.debug('collision prediction...')
        #     logger.debug(f"current position:{init_pos1.tolist()}, {init_pos2.tolist()}")
    return False


class EnvWrapper:
    def __init__(self, model, agent_id=None):

        self._HEADING_ERROR_INDEX = [1, 3, 6, 10, 15]
        self._PERCEIVE_RANGE = 25  # 100
        self._N_NEIGHBORS = 4

        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(
            -1e10, 1e10, (19 + 5 * self._N_NEIGHBORS,)
        )

        self.raw_obs = None
        self.collision_r = None

        self.all_lanes = [""]
        self.lane_index = None
        self.target_lane_index = None
        self.changing_lane = False
        self.lane_id = ""

        self.r_lane_change = 0
        self.static_steps = 0
        self.alert_area_r = 5
        self.left_width = self.right_width = self.init_width = 1.25
        self.ignore_collision_steps = 0
        self.agent_ignore_collision_steps = 0
        self.agent_static_cnt = 0
        self.holding_cnt = 0
        self.moving_act_cnt = 0
        self.try_to_change_lane = False
        self.straight_change_lane = False
        self.find_goal = False
        self.blocking_car_ids = []
        self.neighbor_ids = []
        self.novice_steps = 15
        self.distance_travelled = 0
        self.merge_tag = False
        self.straight_tag = False
        self.final_straight_cnt = 0
        self.changing_lane_while_merging = False
        self.near_goal = False
        self.find_merge = False
        self.find_merge_new = False
        self.straight_tag_cnt = 0
        self.other_agent_nearby = False
        self.find_all_wrong_lane_tag = False
        self.find_all_wrong_lane_tag_init = False
        self.collision_from_behind = False
        self.ego_last_pos = None
        self.ego_current_pos = None
        self.static_at_end_cnt = 0
        self.last_good_wps = None
        self.agent_tag = "unknown"
        self.first_in_bad_lane = False

        # TODO: Initialize the agent
        # encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750, 750])
        # self.model = d3rlpy.algos.DiscreteBC(
        #     use_gpu=False, batch_size=64, encoder_factory=encoder
        # )
        # observations = np.random.random((100, 53))  # Check whether the state dim can be passed from other places
        # actions = np.random.random((100, 2))
        # rewards = np.random.random(100)
        # terminals = np.random.randint(2, size=100)
        # demo_dataset_for_initialization = MDPDataset(observations, actions, rewards, terminals)
        # self.model.build_with_dataset(demo_dataset_for_initialization)
        # self.model.load_model(
        #     './model_50000.pt')  # path to the trained model
        self.model = model

        if agent_id:
            # 保存agent index,供协商时使用
            self.agent_id = agent_id
            self.agent_index = int(agent_id.split("_")[1])

    @property
    def n_lanes(self):
        return len(self.all_lanes)

    @property
    def speed(self):
        return self.raw_obs.ego_vehicle_state.speed

    @property
    def goal_position(self):
        if hasattr(self.raw_obs.ego_vehicle_state.mission.goal, "position"):
            return self.raw_obs.ego_vehicle_state.mission.goal.position[:2]
        else:
            position = self.raw_obs.ego_vehicle_state.position[:2]
            return np.ones((2,), dtype=np.float32) * self._PERCEIVE_RANGE + position

    @property
    def is_endless_mission(self):
        if hasattr(self.raw_obs.ego_vehicle_state.mission.goal, "position"):
            return False
        else:
            return True

    def cal_goal_lane(self, raw_obs):
        goal_lane = np.zeros(5)
        if len(self.all_lanes) == 0 or not hasattr(
            raw_obs.ego_vehicle_state.mission.goal, "position"
        ):
            return goal_lane
        cos_thetas = np.zeros(len(self.all_lanes))
        for i in range(len(cos_thetas)):
            if len(raw_obs.waypoint_paths[i]) <= 1:
                continue
            y1 = raw_obs.waypoint_paths[i][1].pos - raw_obs.waypoint_paths[i][0].pos
            y2 = (
                np.array(raw_obs.ego_vehicle_state.mission.goal.position[:2])
                - raw_obs.waypoint_paths[i][0].pos
            )
            cos_thetas[i] = abs(y1 @ y2 / np.sqrt(y1 @ y1 * y2 @ y2))
        if cos_thetas.max() > 1 - 0.0001:
            goal_lane[np.clip(2 + cos_thetas.argmax() - self.lane_index, 0, 4)] = 1.0
        return goal_lane

    def cal_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def cal_collision_r(self, bounding_box):
        return np.linalg.norm([bounding_box.length, bounding_box.width])

    def pi_clip(self, array):
        array[np.where(array < -np.pi)] += 2 * np.pi
        array[np.where(array > +np.pi)] -= 2 * np.pi
        return array / np.pi

    def step(self, next_raw_obs):
        # self.lane_index = lane_index
        self.update_lane_info(next_raw_obs)
        # designed_rew = self.pack_reward(next_raw_obs, r_forward, reached_goal)
        # r_goal = 0
        # designed_obs = self.pack_observation(next_raw_obs)

        # # if next_raw_obs.events.on_shoulder:
        # #     done = True
        # if done:
        #     position = next_raw_obs.ego_vehicle_state.position[:2]
        #     goal_distance = self.cal_distance(position, self.goal_position)
        #     if goal_distance < 20:
        #         r_goal += min(5 / goal_distance, 10) * 2
        #
        # designed_rew += r_goal
        # # info.update({
        # #     'r_list': r_list,
        # #     'r_goal': r_goal
        # # })
        # # info['is_success'] = (info["score"] or self.is_endless_mission)
        #
        self.raw_obs = next_raw_obs
        # return designed_obs, designed_rew
        return 1
        # return designed_obs

    def update_lane_info(self, raw_obs):
        new_all_lanes = []
        for i, each_lane in enumerate(raw_obs.waypoint_paths):
            each_lane_id = each_lane[0].lane_id
            new_all_lanes.append(each_lane_id)
            if each_lane_id == raw_obs.ego_vehicle_state.lane_id:
                self.lane_index = i

        if new_all_lanes != self.all_lanes:
            if self.lane_index >= len(new_all_lanes):
                self.lane_index = len(new_all_lanes) - 1
            self.target_lane_index = self.lane_index
        self.all_lanes = new_all_lanes

        new_lane_id = raw_obs.waypoint_paths[self.lane_index][0].lane_id
        if new_lane_id != self.lane_id:
            self.changing_lane = False
            self.lane_id = new_lane_id

        assert self.n_lanes > 0, "No waypoints are found!"

    def cal_wps_bias(self):
        raw_obs = self.raw_obs
        headings = []
        if self.lane_index != self.target_lane_index:
            headings.append(float(raw_obs.ego_vehicle_state.heading))
            target_wps = raw_obs.waypoint_paths[self.target_lane_index][:4]
        else:
            target_wps = raw_obs.waypoint_paths[self.target_lane_index][:5]
        for wp in target_wps:
            headings.append(wp.heading)
        if len(headings) < 2:
            return 0
        abs_bias = 0
        for i, heading in enumerate(headings[:-1]):
            next_heading = headings[i + 1]
            abs_bias += abs(next_heading - heading)
        return abs_bias

    def changing_lane_forecast(
        self,
        vehicle_state1,
        neighborhood_vehicle_states,
        target_direction,
        lane_widths,
        ego_lane_heading,
        l_front=1.5,
        l_back=6,
    ):
        direction_index = {"right": 0, "left": 1}[target_direction]
        v1 = vehicle_state1.speed
        theta1 = ego_lane_heading + math.pi / 2  # 逆行时会有问题
        v1_vec = v1 * np.array([math.cos(theta1), math.sin(theta1)])

        init_pos1 = vehicle_state1.position[:2]
        bound1 = vehicle_state1.bounding_box
        l1, w1 = bound1.length, bound1.width
        l1_front_vec, l1_back_vec = (l1 / 2 + l_front) * np.array(
            [math.cos(theta1), math.sin(theta1)]
        ), (l1 / 2 + l_back) * np.array([math.cos(theta1), math.sin(theta1)])
        # todo 获取车辆到lane边缘的距离
        if direction_index == 0:  # 向右并道
            w_left = 0.01
            w_right = lane_widths[0]
        elif direction_index == 1:
            w_right = 0.01
            w_left = lane_widths[2]
        else:
            # logger.debug(f'wrong direction: {direction_index} and {target_direction}')
            aaa = 1
        w1_left_vec = (w1 / 2 + w_left) * np.array(
            [math.sin(theta1), -1 * math.cos(theta1)]
        )
        w1_right_vec = (w1 / 2 + w_right) * np.array(
            [math.sin(theta1), -1 * math.cos(theta1)]
        )
        ego_lane_index = vehicle_state1.lane_index

        for nb in neighborhood_vehicle_states:
            if (
                nb.lane_index
                == ego_lane_index + {"left": 1, "right": -1}[target_direction]
            ):
                # 理论上讲，lane_index不具有唯一性，可能检测到其他方向的车辆，但我们认为，只要在此范围内的车辆，都在换道时有碰撞风险
                # ma 协商
                v2 = nb.speed
                theta2 = nb.heading + math.pi / 2
                v2_vec = v2 * np.array([math.cos(theta2), math.sin(theta2)])
                init_pos2 = nb.position[:2]
                bound2 = nb.bounding_box
                l2, w2 = bound2.length, bound2.width
                l2_vec = l2 / 2 * np.array([math.cos(theta2), math.sin(theta2)])
                w2_vec = w2 / 2 * np.array([math.sin(theta2), -1 * math.cos(theta2)])

                pos1, pos2 = init_pos1, init_pos2
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

    def rule_based_action(self, wps_bias=None):

        # 预处理
        self.ego_last_pos = self.ego_current_pos
        self.ego_current_pos = self.raw_obs.ego_vehicle_state.position[:2]
        if self.ego_last_pos is not None:
            if np.linalg.norm(self.ego_last_pos - self.ego_current_pos) < 0.01:
                self.static_at_end_cnt += 1
            else:
                self.static_at_end_cnt = 0
        in_bad_lane = False

        self.novice_steps = max(0, self.novice_steps - 1)
        self.distance_travelled += self.raw_obs.distance_travelled

        raw_obs = self.raw_obs

        if self.agent_tag == "unknown":
            try:
                lane_id = self.raw_obs.waypoint_paths[self.lane_index][0].lane_id
                if hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                    goal_pos = raw_obs.ego_vehicle_state.mission.goal.position
                    goal_str = (
                        "x_"
                        + str(round(goal_pos[0], 1))
                        + "_y_"
                        + str(round(goal_pos[1], 1))
                    )
                else:
                    goal_str = "no_goal"
                self.agent_tag = lane_id + "+" + goal_str
            except:
                self.agent_tag = "wrong_agent_tag"

        events = raw_obs.events

        self.other_agent_nearby = False
        for nb in raw_obs.neighborhood_vehicle_states:
            if nb.id[:5] == "Agent":
                self.other_agent_nearby = True
                break
        position = raw_obs.ego_vehicle_state.position[:2].copy()
        heading = raw_obs.ego_vehicle_state.heading
        lane_id = raw_obs.ego_vehicle_state.lane_id

        # wps检查
        try:
            if hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                dist_travel_sum = 0
                unavi_path_cnt = 0
                goal_pos = raw_obs.ego_vehicle_state.mission.goal.position[:2]
                goal_pos = np.array(goal_pos)

                for path in raw_obs.waypoint_paths:
                    if len(path) > 3:
                        last_dist_to_goal = np.linalg.norm(path[0].pos - goal_pos)
                        last_wp = path[0]
                        for wp_ind, wp in enumerate(path):
                            if (
                                wp_ind % 3 == 0
                                and wp_ind > 0
                                and np.linalg.norm(wp.pos - last_wp.pos)
                            ):
                                dist_to_goal = np.linalg.norm(
                                    path[wp_ind].pos - goal_pos
                                )
                                if dist_to_goal > last_dist_to_goal + 0.01:
                                    # logger.debug(f'find wrong lane:{path[0].lane_id}, {path[0].lane_index}')
                                    unavi_path_cnt += 1
                                    break
                                last_dist_to_goal = dist_to_goal
                                last_wp = wp
                if len(raw_obs.waypoint_paths) == unavi_path_cnt:
                    # logger.debug(f'all wrong lane at {raw_obs.elapsed_sim_time}')
                    if not self.find_all_wrong_lane_tag_init:
                        # logger.debug(f'{self.agent_tag} first find wrong lanes at {raw_obs.elapsed_sim_time}')
                        self.find_all_wrong_lane_tag_init = True
                        try:
                            # logger.debug(f'neighbor info')
                            ego_pos = raw_obs.ego_vehicle_state.position[:2]
                            logger.debug(
                                f"meet all wrong lane, current lane:{raw_obs.ego_vehicle_state.lane_index}"
                            )
                            logger.debug("  ")
                            logger.debug(
                                f"ego pos x:{round(ego_pos[0], 4)}, y:{round(ego_pos[1], 4)}"
                            )
                            if hasattr(
                                raw_obs.ego_vehicle_state.mission.goal, "position"
                            ):
                                logger.debug(
                                    f"goal pos x:{round(raw_obs.ego_vehicle_state.mission.goal.position.x, 4)}, y:{round(raw_obs.ego_vehicle_state.mission.goal.position.y, 4)}"
                                )
                            exist_lane_id_set = set()
                            for rd_ind, rd_key in enumerate(
                                raw_obs.road_waypoints.lanes
                            ):
                                rd = raw_obs.road_waypoints.lanes[rd_key]
                                for ln_ind, ln in enumerate(rd):
                                    if (
                                        len(ln) > 0
                                        and ln[0].lane_id not in exist_lane_id_set
                                    ):
                                        logger.debug(
                                            f"len of {ln[0].lane_id}:{len(ln)}"
                                        )
                                        logger.debug(
                                            f"ln first wp x:{round(ln[0].pos[0], 4)}, y:{round(ln[0].pos[1], 4)}, h:{round(float(ln[0].heading), 4)}"
                                        )
                                        if len(ln) > 1:
                                            logger.debug(
                                                f"ln last wp x:{round(ln[-1].pos[0], 4)}, y:{round(ln[-1].pos[1], 4)}, h:{round(float(ln[-1].heading), 4)}"
                                            )
                                        exist_lane_id_set.add(ln[0].lane_id)
                                logger.debug("  ")
                            try:
                                for wps_ind, wps in enumerate(raw_obs.waypoint_paths):
                                    if len(wps) > 0:
                                        logger.debug(
                                            f"wronglane len of {wps[0].lane_id}:{len(wps)}"
                                        )
                                        logger.debug(
                                            f"ln first wp x:{round(wps[0].pos[0], 4)}, y:{round(wps[0].pos[1], 4)}, h:{round(float(wps[0].heading), 4)}"
                                        )
                                        if len(wps) > 1:
                                            logger.debug(
                                                f"ln last wp x:{round(wps[-1].pos[0], 4)}, y:{round(wps[-1].pos[1], 4)}, h:{round(float(wps[-1].heading), 4)}"
                                            )
                                        exist_lane_id_set.add(wps[0].lane_id)
                                logger.debug("  ")
                            except:
                                logger.debug("wrong wps info")

                        except:
                            logger.debug("wrong lanes info")
        except:
            logger.debug("wrong wps check")

        # 转弯减速
        is_turning = False
        ego_wp_path = self.raw_obs.waypoint_paths[self.lane_index]
        if len(ego_wp_path) >= 2:
            for i, wp in enumerate(ego_wp_path[:-1][:10]):
                next_wp = ego_wp_path[i + 1]
                if abs(next_wp.heading - wp.heading) > 5 * math.pi / 180:  # 5
                    is_turning = True
                    break

        # 拥挤减速
        vel_atten_rate = 1.0
        detection_radius = 20
        for neighbor in raw_obs.neighborhood_vehicle_states:
            if np.linalg.norm(neighbor.position[:2] - position) < detection_radius:
                vel_atten_rate -= 0.02
        vel_atten_rate = max(vel_atten_rate, 0.5)
        # logger.debug('atten rate:', vel_atten_rate)

        # merge识别/直道识别
        self.merge_tag = False
        self.straight_tag = True

        path_dheadings = []  # 弧度制
        path_headings = []
        path_lane_indexs = []
        for lane_idx_in_list, path_wps in enumerate(raw_obs.waypoint_paths):
            abs_dheadings = []
            for i in range(len(path_wps) - 1):
                current_h, next_h = path_wps[i].heading, path_wps[i + 1].heading
                raw_dh = abs(next_h - current_h)
                abs_dh = min(raw_dh, 2 * math.pi - raw_dh)
                abs_dheadings.append(abs_dh)

            if len(abs_dheadings) > 10:
                path_dheadings.append((lane_idx_in_list, abs_dheadings))

            if len(abs_dheadings) > 0 and max(abs_dheadings[:10]) > 1e-3:
                self.straight_tag = False

        if self.straight_tag:
            self.straight_tag_cnt = self.straight_tag_cnt + 1
        else:
            self.straight_tag_cnt = 0
        try:
            if len(raw_obs.waypoint_paths) > 1:
                lane_end_pos = []
                max_dh_end, max_dist_to_ego = -1, -1
                ego_pos = raw_obs.ego_vehicle_state.position[:2]
                if len(path_dheadings) > 1:
                    for lane_idx_in_list, abs_dheadings in path_dheadings:
                        path = raw_obs.waypoint_paths[lane_idx_in_list]
                        max_dh_end = max(max_dh_end, max(abs_dheadings[-10:]))
                        max_dist_to_ego = max(
                            max_dist_to_ego, np.linalg.norm(ego_pos - path[0].pos)
                        )
                        lane_end_pos.append(path[-1].pos)

                    if (
                        max_dh_end < 1e-4 and max_dist_to_ego < 1
                    ):  # todo 统计50m内且位于不可达车道上的车辆
                        for lane_idx_in_list, path_wps in enumerate(
                            raw_obs.waypoint_paths
                        ):
                            tmp_lane_index = path_wps[0].lane_index
                            try:
                                abs_dheadings = path_dheadings[lane_idx_in_list]
                            except:
                                continue
                            tmp_headings = [wp.heading for wp in path_wps]
                            max_dheading = -1
                            for i in range(len(path_wps) - 1):
                                current_h = path_wps[i].heading
                                for j in range(i + 1, len(path_wps)):
                                    next_h = path_wps[j].heading
                                    raw_dh = abs(next_h - current_h)
                                    abs_dh = min(raw_dh, 2 * math.pi - raw_dh)
                                    max_dheading = max(max_dheading, abs_dh)
                            if (
                                max_dheading < 45 / 180 * math.pi
                            ):  # todo O(n2)寻找小于pi的最大角度
                                path_dheadings.append((lane_idx_in_list, abs_dheadings))
                                path_headings.append((lane_idx_in_list, tmp_headings))
                                path_lane_indexs.append(tmp_lane_index)

                        reachable_lane_ids = set()  # 记录附近可达车道的所有lane_id
                        for path in raw_obs.waypoint_paths:
                            for wp in path[:3]:
                                reachable_lane_ids.add(wp.lane_id)
                        unreachable_cars = []
                        curr_pos = raw_obs.ego_vehicle_state.position[:2]
                        for nb in raw_obs.neighborhood_vehicle_states:
                            if (
                                np.linalg.norm(curr_pos - nb.position[:2])
                                and nb.lane_id not in reachable_lane_ids
                            ):
                                if nb.id[:5] != "Agent":
                                    unreachable_cars.append(nb)
                        unreachable_car_headings = [
                            float(nb.heading) for nb in unreachable_cars
                        ]
                        if len(unreachable_car_headings) >= 3:
                            # todo 这里可以做一个clip，筛选出最近的n个social car
                            avg_nb_heading = sum(unreachable_car_headings) / len(
                                unreachable_car_headings
                            )
                            best_lane_idx_in_list, min_dheading = 0, 100
                            for index_in_list, headings in path_headings:
                                avg_end_h = np.mean(np.array(headings[-10:]))
                                raw_dh = abs(avg_end_h - avg_nb_heading)
                                abs_dh = min(raw_dh, 2 * math.pi - raw_dh)
                                if abs_dh < min_dheading:
                                    best_lane_idx_in_list = index_in_list
                                    min_dheading = abs_dh
                            if len(path_lane_indexs) <= best_lane_idx_in_list:
                                logger.debug(
                                    f"unexpected path lane index:{best_lane_idx_in_list}, {len(path_lane_indexs)}"
                                )
                            else:
                                best_lane_index = path_lane_indexs[
                                    best_lane_idx_in_list
                                ]
                                if not self.find_merge:
                                    logger.debug(
                                        f"merge info: avg heading:{avg_nb_heading}, best lane:{best_lane_index} at {raw_obs.elapsed_sim_time}"
                                    )
                                    self.find_merge = True
                                if (
                                    best_lane_index
                                    != raw_obs.ego_vehicle_state.lane_index
                                ):
                                    self.merge_tag = True

        except:
            logger.debug("wrong merge detection")

        exp_speed = raw_obs.waypoint_paths[self.lane_index][0].speed_limit

        collision_from_behind_tag = False
        if self.straight_tag and not self.collision_from_behind:
            try:
                ego = raw_obs.ego_vehicle_state
                for nb in raw_obs.neighborhood_vehicle_states:
                    tmp_dh = min(
                        math.pi * 2 - abs(nb.heading - ego.heading),
                        abs(nb.heading - ego.heading),
                    )
                    tmp_dist = np.linalg.norm(nb.position[:2] - ego.position[:2])
                    if (
                        nb.lane_index == ego.lane_index
                        and tmp_dist < 10
                        and tmp_dist
                        < 0.8 + nb.bounding_box.length / 2 + ego.bounding_box.length / 2
                        and tmp_dh < math.pi / 180 * 1
                        and abs(ego.lane_position.t) < 1e-5
                        and abs(nb.lane_position.t) < 1e-5
                    ):
                        collision_from_behind_tag = True

                        self.collision_from_behind = True
            except:
                pass

        left_exp_speed, left_dist_min = 0, 2 + 8
        if self.lane_index != len(raw_obs.waypoint_paths) - 1:
            left_exp_speed = raw_obs.waypoint_paths[self.lane_index + 1][0].speed_limit

        right_exp_speed, right_dist_min = 0, 2 + 8
        if self.lane_index != 0:
            right_exp_speed = raw_obs.waypoint_paths[self.lane_index - 1][0].speed_limit

        dist_min_init = dist_min = 5.5
        reachable_lane_ids = set()  # 记录附近可达车道的所有lane_id
        curr_pos = raw_obs.ego_vehicle_state.position[:2]
        for path in raw_obs.waypoint_paths:
            wp_lane_index = path[0].lane_index
            if self.lane_index - 1 <= wp_lane_index <= self.lane_index + 1:
                for wp in path:
                    if abs(wp.lane_index - self.lane_index) > 1:
                        # logger.debug(f'wrong lane index {wp.lane_index} and {self.lane_index}')
                        continue
                    if (
                        np.linalg.norm(wp.pos - curr_pos)
                        >= [2 + 2, 10 + 2][abs(wp.lane_index - self.lane_index)]
                    ):
                        reachable_lane_ids.add(wp.lane_id)

        s_dist_from_center = abs(raw_obs.ego_vehicle_state.lane_position.t)
        changing_lane = False
        if self.lane_index != self.target_lane_index:
            changing_lane = True
        elif (
            s_dist_from_center
            >= raw_obs.waypoint_paths[self.lane_index][0].lane_width / 10
        ):
            changing_lane = True
            if self.left_width > self.right_width:
                self.left_width = (
                    self.raw_obs.waypoint_paths[self.lane_index][0].lane_width / 2
                )
            elif self.left_width < self.right_width:
                self.right_width = (
                    self.raw_obs.waypoint_paths[self.lane_index][0].lane_width / 2
                )
            else:
                # logger.debug('error: unchanged widths', self.left_width, self.right_width)
                # logger.debug(s_dist_from_center, raw_obs.waypoint_paths[self.lane_index][0].lane_width / 10)
                # exit(0)
                pass
        else:
            self.left_width = self.right_width = self.init_width
        raw_dheading_from_lane = abs(
            float(
                raw_obs.ego_vehicle_state.heading
                - raw_obs.waypoint_paths[self.lane_index][0].heading
            )
        )
        dheading_from_lane = min(
            raw_dheading_from_lane, 2 * math.pi - raw_dheading_from_lane
        )
        if (
            self.near_goal
            and self.straight_tag
            and self.lane_index == self.target_lane_index
            and s_dist_from_center
            < raw_obs.waypoint_paths[self.lane_index][0].lane_width / 10
            and dheading_from_lane > 10 * math.pi / 180
        ):
            if self.changing_lane_while_merging or not self.other_agent_nearby:
                changing_lane = True
        if (
            self.near_goal
            and self.straight_tag
            and changing_lane
            and (self.changing_lane_while_merging or not self.other_agent_nearby)
        ):
            l_front = 0.5
            forecast_steps = 0
        elif changing_lane:
            l_front = 2.5
            forecast_steps = 5
        elif is_turning:
            l_front = 7
            forecast_steps = 5
        else:
            l_front = 5
            forecast_steps = 5

        # agent
        packed_obs = self.pack_observation_v1(raw_obs)
        # This version doesn't need to proprocess the observations
        # TODO: Call the agent
        if (
            self.agent_ignore_collision_steps > 0
            or self.novice_steps > 0
            or self.distance_travelled < 5
            or collision_from_behind_tag
        ):
            model_action = 1
            if self.agent_ignore_collision_steps > 0:
                self.agent_ignore_collision_steps -= 1
        elif packed_obs is not None:
            model_action = self.model.predict([packed_obs])[0]
            # logger.debug('action', model_action)
        else:
            model_action = 1  # 信息不足，放弃决策
        if model_action == 1:
            self.agent_static_cnt = 0
        elif self.agent_static_cnt >= 10:
            self.agent_static_cnt = 0
            self.agent_ignore_collision_steps = 10  # 10
        elif model_action == 0 and self.agent_ignore_collision_steps == 0:
            self.agent_static_cnt += 1
            # if self.static_steps > 0:
            #     self.static_steps -= 1
            #     self.holding_cnt += 1

        if self.ignore_collision_steps > 0:
            self.ignore_collision_steps -= 1
        elif self.static_steps > 0:
            self.holding_cnt += 1
            if model_action == 1:  # 直行
                self.moving_act_cnt += 1
            # self.static_steps = 0
            # self.ignore_collision_steps = 10
            elif model_action == 0:  # 保持静止
                pass
            else:
                assert 0, f"wrong action index:{model_action}"
            is_clear = True

            # 1028 rule 1
            # 如果引发制动的车辆从碰撞预警区域消失，则停止对该车辆的监视
            last_blocking_car_ids = copy.deepcopy(self.blocking_car_ids)
            current_nb_ids = [n.id for n in raw_obs.neighborhood_vehicle_states]
            for key_nb_id in self.blocking_car_ids:
                if key_nb_id not in current_nb_ids:
                    self.blocking_car_ids.remove(key_nb_id)
                    continue
                for nb in raw_obs.neighborhood_vehicle_states:
                    if nb.id == key_nb_id:
                        if (
                            self.near_goal
                            and self.straight_tag
                            and changing_lane
                            and nb.lane_index != self.target_lane_index
                        ):
                            if (
                                self.changing_lane_while_merging
                                or not self.other_agent_nearby
                            ):
                                continue
                        ego_to_nb_pos_vec = (
                            nb.position[:2] - raw_obs.ego_vehicle_state.position[:2]
                        )
                        ego_heading_x = (
                            float(raw_obs.ego_vehicle_state.heading) + math.pi / 2
                        )
                        heading_vec = np.array(
                            [math.cos(ego_heading_x), math.sin(ego_heading_x)]
                        )
                        cos_theta = np.sum(heading_vec * ego_to_nb_pos_vec) / (
                            np.linalg.norm(ego_to_nb_pos_vec) + 1e-5
                        )
                        if cos_theta < math.cos(math.pi * 5 / 6):  # 侧后方碰撞不能停车
                            continue
                        if collision_forecast(
                            raw_obs.ego_vehicle_state,
                            nb,
                            w_left=self.left_width,
                            l_front=l_front,
                            w_right=self.right_width,
                            steps=forecast_steps,
                        ):
                            is_clear = False  # 引发阻塞的key neighbor仍位于碰撞检测区域内
                        else:
                            self.blocking_car_ids.remove(
                                key_nb_id
                            )  # 引发阻塞的key neighbor已经离开了碰撞检测区域
                        break  # 已经找到了key neighbor

            if not is_clear:  # 如果引发阻塞的key neighbor都没有碰撞风险，以较宽松的规则检测是否会与其他车辆碰撞
                for (
                    neighbor
                ) in raw_obs.neighborhood_vehicle_states:  # 圆形范围检测会导致车辆卡住后面的车 所以暂时使用矩形
                    if neighbor.id in last_blocking_car_ids:
                        continue  # 不再重复检测key neighbor
                    ego_to_nb_pos_vec = (
                        neighbor.position[:2] - raw_obs.ego_vehicle_state.position[:2]
                    )
                    ego_heading_x = (
                        float(raw_obs.ego_vehicle_state.heading) + math.pi / 2
                    )
                    heading_vec = np.array(
                        [math.cos(ego_heading_x), math.sin(ego_heading_x)]
                    )
                    cos_theta = np.sum(heading_vec * ego_to_nb_pos_vec) / (
                        np.linalg.norm(ego_to_nb_pos_vec) + 1e-5
                    )
                    if cos_theta < math.cos(math.pi * 5 / 6):  # 侧后方碰撞不能停车
                        continue
                    if wps_bias != None and neighbor.id[:5] == "Agent":
                        neighbor_agent_id = neighbor.id.split("-")[0]
                        neighbor_agent_index = int(neighbor_agent_id.split("_")[1])
                        if wps_bias[self.agent_id] < wps_bias[neighbor_agent_id]:
                            continue  # 转弯让直行，直行车辆进行一个宽松的碰撞检测
                        elif (
                            wps_bias[self.agent_id] == wps_bias[neighbor_agent_id]
                            and self.agent_index < neighbor_agent_index
                        ):
                            continue  # 车辆序号大的让小的
                    if (
                        self.near_goal
                        and self.straight_tag
                        and changing_lane
                        and neighbor.lane_index != self.target_lane_index
                    ):
                        if (
                            self.changing_lane_while_merging
                            or not self.other_agent_nearby
                        ):
                            continue
                    if collision_forecast(
                        raw_obs.ego_vehicle_state,
                        neighbor,
                        l_front=l_front,
                        w_left=0.001,
                        w_right=0.001,
                        steps=forecast_steps,
                    ):
                        is_clear = False

            if is_clear:
                self.static_steps = 0
                if self.holding_cnt > 1:
                    # logger.debug(f'holding for {self.holding_cnt} steps')
                    # logger.debug(f'the model has maken {self.moving_act_cnt} moving actions')
                    aaa = 1
            else:
                self.static_steps -= 1
                if self.static_steps == 0:
                    self.ignore_collision_steps = 10
                    # logger.debug()
                    if self.holding_cnt > 1:
                        # logger.debug(f'holding for {self.holding_cnt} steps')
                        # logger.debug(f'the model has maken {self.moving_act_cnt} moving actions')
                        aaa = 1
                return np.concatenate([curr_pos, [heading, 0.1]])

        else:
            for neighbor in raw_obs.neighborhood_vehicle_states:
                ego_to_nb_pos_vec = (
                    neighbor.position[:2] - raw_obs.ego_vehicle_state.position[:2]
                )
                ego_heading_x = float(raw_obs.ego_vehicle_state.heading) + math.pi / 2
                heading_vec = np.array(
                    [math.cos(ego_heading_x), math.sin(ego_heading_x)]
                )
                cos_theta = np.sum(heading_vec * ego_to_nb_pos_vec) / (
                    np.linalg.norm(ego_to_nb_pos_vec) + 1e-5
                )
                if cos_theta < math.cos(math.pi * 5 / 6):  # 侧后方碰撞不能停车
                    continue
                if wps_bias != None and neighbor.id[:5] == "Agent":
                    neighbor_agent_id = neighbor.id.split("-")[0]
                    neighbor_agent_index = int(neighbor_agent_id.split("_")[1])
                    if wps_bias[self.agent_id] < wps_bias[neighbor_agent_id]:
                        continue  # 转弯让直行
                    elif (
                        wps_bias[self.agent_id] == wps_bias[neighbor_agent_id]
                        and self.agent_index < neighbor_agent_index
                    ):
                        continue  # 车辆序号大的让小的
                if (
                    self.near_goal
                    and self.straight_tag
                    and changing_lane
                    and neighbor.lane_index != self.target_lane_index
                ):
                    if self.changing_lane_while_merging or not self.other_agent_nearby:
                        continue
                if collision_forecast(
                    raw_obs.ego_vehicle_state,
                    neighbor,
                    l_front=l_front,
                    w_left=self.left_width,
                    w_right=self.right_width,
                    steps=forecast_steps,
                ):
                    self.blocking_car_ids.append(neighbor.id)  #
                    # 'car-flow-route-gneE3_2_0-gneE4_2_max-1160313594726501813--7266489842092764092--1-0.0'
                    self.static_steps = 50  # 静止一段时间
                    self.static_steps -= 1
                    self.holding_cnt = 0
                    self.moving_act_cnt = 0
                    return np.concatenate([curr_pos, [heading, 0.1]])

        # 规则选择了前行
        for neighbor in raw_obs.neighborhood_vehicle_states:  # 计算附近3车道的预期速度
            neighbor_pos, curr_wp_pos = (
                neighbor.position[:2],
                raw_obs.ego_vehicle_state.position[:2],
            )
            lane_heading = (
                raw_obs.waypoint_paths[self.lane_index][0].heading + math.pi / 2
            )  # 将极轴由y正半轴调整为x正半轴
            lane_vec = np.array([math.cos(lane_heading), math.sin(lane_heading)])
            rela_pos_vec = neighbor_pos - curr_wp_pos
            cos_rela_pos_angle = np.dot(rela_pos_vec, lane_vec) / np.linalg.norm(
                rela_pos_vec
            )  # 相邻车辆与车道方向的夹角余弦值
            if (
                self.lane_index - 1 <= neighbor.lane_index <= self.lane_index + 1
                and abs(
                    neighbor.heading
                    - raw_obs.waypoint_paths[self.lane_index][0].heading
                )
                <= math.pi / 2
                and cos_rela_pos_angle >= 0
                and neighbor.lane_id in reachable_lane_ids
            ):  # todo 用relative pos和curr road heading判定车辆相对位置
                neighbor_dists = np.clip(
                    self.cal_distance(
                        raw_obs.ego_vehicle_state.position, neighbor.position
                    )
                    - self.cal_collision_r(neighbor.bounding_box)
                    - self.collision_r,
                    0,
                    None,
                )
                if (
                    neighbor_dists < dist_min_init
                    and neighbor.lane_index == self.lane_index
                ):
                    dist_min = neighbor_dists
                    exp_speed = max(0, min(neighbor.speed, exp_speed))
                    # exp_speed = max(0, min(neighbor.speed - (dist_min_init - neighbor_dists) / 0.1, exp_speed))
                elif (
                    not changing_lane
                    and neighbor_dists < right_dist_min
                    and neighbor.lane_index == self.lane_index - 1
                    and self.lane_index > 0
                ):
                    right_dist_min = neighbor_dists
                    right_exp_speed = min(neighbor.speed, right_exp_speed)
                elif (
                    not changing_lane
                    and neighbor_dists < left_dist_min
                    and neighbor.lane_index == self.lane_index + 1
                    and self.lane_index < self.n_lanes - 1
                ):
                    left_dist_min = neighbor_dists  # todo 给相邻车道dist计算投影距离
                    left_exp_speed = min(neighbor.speed, left_exp_speed)

        near_goal = False
        if not changing_lane:
            if model_action == 0:  # 模型要求停车
                return np.concatenate([curr_pos, [heading, 0.1]])

            # 计算换道速度增益阈值

            change_lane2right_ver_threshold = (
                change_lane2left_ver_threshold
            ) = change_lane_ver_threshold = 1

            change_to_goal_lane_dist_threshold = 20
            if self.straight_tag and self.straight_tag_cnt >= 10:
                if self.changing_lane_while_merging or not self.other_agent_nearby:
                    change_to_goal_lane_dist_threshold = 50

            changeable_good_lane = [-1, -1]
            ego_path = self.raw_obs.waypoint_paths[self.lane_index]
            if (
                len(ego_path) == 1
                and self.ego_last_pos is not None
                and np.linalg.norm(
                    self.raw_obs.ego_vehicle_state.position[:2] - self.ego_last_pos
                )
                < 0.1
            ):
                logger.debug(
                    f"in bad lane:{self.lane_index} at {raw_obs.elapsed_sim_time}"
                )
                in_bad_lane = True  # 可能要加远离终点约束
                if not self.first_in_bad_lane:
                    self.first_in_bad_lane = True

            curr_pos = raw_obs.ego_vehicle_state.position[:2]
            if hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                if not self.find_goal:
                    self.find_goal = True
                    logger.debug(f"looking for target at {raw_obs.elapsed_sim_time}")
                goal_pos = raw_obs.ego_vehicle_state.mission.goal.position[:2]
                goal_pos = np.array(goal_pos)
                dist_to_goal = np.linalg.norm(goal_pos - curr_pos)
                lane_index_of_goal, lane_id_of_goal = None, None
                if dist_to_goal < change_to_goal_lane_dist_threshold:
                    self.near_goal = True
                    if dist_to_goal < 10 and not self.straight_change_lane:
                        logger.debug(
                            f"keep moving on straight at {raw_obs.elapsed_sim_time}"
                        )
                        self.straight_change_lane = True
                    if not self.try_to_change_lane:
                        self.try_to_change_lane = True
                        logger.debug(f"changing lane at {raw_obs.elapsed_sim_time}")
                    # todo 找出goal所在车道
                    for (
                        lane
                    ) in raw_obs.waypoint_paths:  # todo 不确定是否会在waypoint_paths出现只有1个点的车道
                        if lane_index_of_goal is not None:
                            break
                        if len(lane) < 2:
                            continue
                        lane_id, lane_index = lane[0].lane_id, lane[0].lane_index
                        lane_width = lane[0].lane_width
                        for wp_idx in range(
                            len(lane) - 1
                        ):  # goal在车道上的投影位于2点之间，且到车道中心的距离小于width/2
                            wp1, wp2 = lane[wp_idx].pos, lane[wp_idx + 1].pos
                            vec1 = wp1 - goal_pos
                            vec2 = wp2 - goal_pos
                            lane_vec = wp1 - wp2
                            distance_to_lane = np.abs(
                                np.cross(vec1, vec2)
                            ) / np.linalg.norm(
                                wp1 - wp2
                            )  # 近似地计算goal到车道中心线的距离
                            is_goal_between_2wps = (
                                np.dot(vec1, lane_vec) * np.dot(vec2, lane_vec) <= 0
                            )  # 判断投影是否位于2个waypoint之间
                            min_dist_to_wps = min(
                                np.linalg.norm(vec1), np.linalg.norm(vec2)
                            )
                            if (
                                distance_to_lane <= lane_width / 2
                                and is_goal_between_2wps
                                or min_dist_to_wps <= lane_width / 2
                            ):
                                lane_index_of_goal, lane_id_of_goal = (
                                    lane_index,
                                    lane_id,
                                )
                    if lane_index_of_goal is None:
                        try:
                            goal_lane_vec = self.cal_goal_lane(raw_obs)
                            if np.max(goal_lane_vec) > 0.5:
                                lane_index_of_goal = max(
                                    0,
                                    int(np.argmax(goal_lane_vec)) - 2 + self.lane_index,
                                )
                        except:
                            pass
                    if (
                        lane_index_of_goal is not None
                        and lane_index_of_goal != self.lane_index
                    ):
                        decrease_v_threshold = 100 / (
                            dist_to_goal + 1e-5
                        )  # 还有10m时克服10m/s的速度差
                        if self.straight_tag:
                            if (
                                self.changing_lane_while_merging
                                or not self.other_agent_nearby
                            ):
                                decrease_v_threshold *= 2.5
                        # todo 开始检测周围车辆，若目标车道有车辆，则抑制换道并减速，为并道做准备
                        lane_widths = [
                            0,
                            raw_obs.waypoint_paths[self.lane_index][0].lane_width,
                            0,
                        ]
                        if self.lane_index > 0:
                            lane_widths[0] = raw_obs.waypoint_paths[
                                self.lane_index - 1
                            ][0].lane_width
                        if self.lane_index < len(self.all_lanes) - 1:
                            lane_widths[2] = raw_obs.waypoint_paths[
                                self.lane_index + 1
                            ][0].lane_width
                        ego_lane_heading = float(
                            raw_obs.waypoint_paths[self.lane_index][0].heading
                        )
                        if lane_index_of_goal < self.lane_index:  # 序号由右至左排列，goal在当前车道右侧
                            if (
                                self.changing_lane_forecast(
                                    raw_obs.ego_vehicle_state,
                                    raw_obs.neighborhood_vehicle_states,
                                    "right",
                                    lane_widths,
                                    ego_lane_heading,
                                )
                                and self.near_goal
                                and self.straight_tag
                            ):
                                if (
                                    self.changing_lane_while_merging
                                    or not self.other_agent_nearby
                                ):
                                    self.final_straight_cnt += 1
                                    change_lane2right_ver_threshold += 1000
                            else:
                                change_lane2right_ver_threshold -= decrease_v_threshold
                        elif lane_index_of_goal > self.lane_index:
                            if (
                                self.changing_lane_forecast(
                                    raw_obs.ego_vehicle_state,
                                    raw_obs.neighborhood_vehicle_states,
                                    "left",
                                    lane_widths,
                                    ego_lane_heading,
                                )
                                and self.near_goal
                                and self.straight_tag
                            ):
                                if (
                                    self.changing_lane_while_merging
                                    or not self.other_agent_nearby
                                ):
                                    self.final_straight_cnt += 1
                                    change_lane2left_ver_threshold += 1000
                            else:
                                change_lane2left_ver_threshold -= decrease_v_threshold

                    if (
                        lane_index_of_goal is not None
                        and lane_index_of_goal <= self.lane_index
                    ):  # 序号由右至左排列，goal在当前车道右侧
                        change_lane2left_ver_threshold += (
                            1000  # 2 * decrease_v_threshold
                        )
                    if (
                        lane_index_of_goal is not None
                        and lane_index_of_goal >= self.lane_index
                    ):
                        change_lane2right_ver_threshold += (
                            1000  # 2 * decrease_v_threshold
                        )
            current_lane_speed_limit = raw_obs.waypoint_paths[self.lane_index][
                0
            ].speed_limit
            if self.novice_steps > 0 or self.distance_travelled < 5:
                change_lane2left_ver_threshold += current_lane_speed_limit * 0.75
                change_lane2right_ver_threshold += current_lane_speed_limit * 0.75
            else:
                change_lane2left_ver_threshold += current_lane_speed_limit * 0.2
                change_lane2right_ver_threshold += current_lane_speed_limit * 0.2
            if self.merge_tag:
                if best_lane_index > self.lane_index:  # 最佳车道在左侧
                    change_lane2left_ver_threshold -= current_lane_speed_limit
                    change_lane2right_ver_threshold += 1000
                elif best_lane_index < self.lane_index:
                    change_lane2right_ver_threshold -= current_lane_speed_limit
                    change_lane2left_ver_threshold += 1000
            if (
                len(self.raw_obs.waypoint_paths) > 1
                and in_bad_lane
                and not self.find_all_wrong_lane_tag_init
            ):
                current_lane_punishment = 2000
            else:
                current_lane_punishment = 0
            action = np.argmax(
                np.array(
                    [
                        right_exp_speed - change_lane2right_ver_threshold,
                        exp_speed - current_lane_punishment,
                        left_exp_speed - change_lane2left_ver_threshold,
                    ]
                )
            )
            if current_lane_punishment > 0:
                logger.debug(
                    f"{self.agent_tag} find punishment, action:{action} at {raw_obs.elapsed_sim_time}"
                )
            if action != 1:
                self.final_straight_cnt = 0  # 换道后停止减速，快速换道
            if self.merge_tag and best_lane_index > self.lane_index and action == 2:
                self.changing_lane_while_merging = True
                logger.debug("changing lane to left while merging")
            if self.merge_tag and best_lane_index < self.lane_index and action == 0:
                self.changing_lane_while_merging = True
                logger.debug("changing lane to right while merging")
        else:
            action = 1  # 换道中
        # exp_speed = [right_exp_speed, exp_speed, left_exp_speed][action]  # todo 车辆一般无法在1个step内换道，所以仍采用当前车道的最大速度

        # * keep_lane
        # logger.debug('sim time', raw_obs.elapsed_sim_time)
        # logger.debug('lane index', self.lane_index)

        # 拥堵速度衰减
        exp_speed *= vel_atten_rate

        target_path = raw_obs.waypoint_paths[self.target_lane_index]

        if action != 1:
            checkpoint = 1
        if action == 1:
            self.r_lane_change = 0
            target_wp = target_path[:3][-1]  # 第2个wp
            if self.merge_tag and self.changing_lane_while_merging:
                if (
                    np.linalg.norm(
                        target_wp.pos - raw_obs.ego_vehicle_state.position[:2]
                    )
                    < 0.5
                ):  # 避免agent原地不动
                    for wp_ind, wp in enumerate(target_path):
                        if (
                            np.linalg.norm(
                                wp.pos - raw_obs.ego_vehicle_state.position[:2]
                            )
                            >= 0.5
                        ):
                            target_wp = wp
                            break

        # * change_lane_left
        elif action == 2:
            self.r_lane_change = -2
            if self.target_lane_index < self.n_lanes - 1:
                self.target_lane_index += 1
            self.left_width += target_path[0].lane_width
            target_wp = target_path[:4][-1]  # 第3个wp

        # * change_lane_right
        elif action == 0:
            self.r_lane_change = -2
            # exp_speed *= [0.4, 0.7, 1.0][action - 3]
            if self.target_lane_index > 0:
                self.target_lane_index -= 1
            self.right_width += target_path[0].lane_width
            target_wp = target_path[:4][-1]  # 第3个wp

        if action != 1:
            if (
                self.near_goal
                and self.straight_tag
                and (self.changing_lane_while_merging or not self.other_agent_nearby)
            ):
                forecast_steps = 0
                l_front = 0.5
                if self.target_lane_index != self.lane_index:
                    target_lane_index = self.target_lane_index
                else:
                    target_lane_index = None
            else:
                forecast_steps = 5
                target_lane_index = None
            for neighbor in raw_obs.neighborhood_vehicle_states:
                ego_to_nb_pos_vec = (
                    neighbor.position[:2] - raw_obs.ego_vehicle_state.position[:2]
                )
                ego_heading_x = float(raw_obs.ego_vehicle_state.heading) + math.pi / 2
                heading_vec = np.array(
                    [math.cos(ego_heading_x), math.sin(ego_heading_x)]
                )
                cos_theta = np.sum(heading_vec * ego_to_nb_pos_vec) / (
                    np.linalg.norm(ego_to_nb_pos_vec) + 1e-5
                )
                if cos_theta < math.cos(math.pi * 5 / 6):  # 侧后方碰撞不能停车
                    continue
                if (
                    self.near_goal
                    and self.straight_tag
                    and target_lane_index is not None
                    and target_lane_index != neighbor.lane_index
                ):
                    if self.changing_lane_while_merging or not self.other_agent_nearby:
                        continue
                if collision_forecast(
                    raw_obs.ego_vehicle_state,
                    neighbor,
                    l_front=l_front,
                    w_left=self.left_width,
                    w_right=self.right_width,
                    steps=forecast_steps,
                ):
                    self.blocking_car_ids.append(neighbor.id)  #
                    self.static_steps = 50  # 静止一段时间
                    self.static_steps -= 1
                    return np.concatenate([curr_pos, [heading, 0.1]])

        # ?
        # if np.any(raw_obs.occupancy_grid_map.data[108:118, 128]):
        #     exp_speed = min(0, exp_speed)

        self.exp_speed = exp_speed

        if is_turning:
            target_wp = target_path[:2][-1]
            exp_speed *= 0.5
        if self.final_straight_cnt > 0:
            exp_speed *= max(0.2, 0.1 * (10 - self.final_straight_cnt))

        target_wp_pos = target_wp.pos
        if (
            in_bad_lane
            and self.static_at_end_cnt >= 3
            and len(raw_obs.waypoint_paths) == 1
            and not self.find_all_wrong_lane_tag_init
        ):
            if hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                target_wp_pos = np.array(
                    raw_obs.ego_vehicle_state.mission.goal.position[:2]
                )

        delta_pos = target_wp_pos - position
        delta_pos_dist = self.cal_distance(target_wp.pos, position)

        max_dheading = 10 * math.pi / 180
        heading = np.clip(
            np.arctan2(-delta_pos[0], delta_pos[1]),
            heading - max_dheading,
            heading + max_dheading,
        )  # todo 感觉这个clip约束很难触发，3*180/pi>120'

        if delta_pos_dist <= self.speed * 0.1:
            position = target_wp_pos.copy()
        else:
            position += delta_pos / delta_pos_dist * exp_speed * 0.1
        if (
            in_bad_lane
            and self.static_at_end_cnt >= 3
            and len(raw_obs.waypoint_paths) == 1
        ):
            if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                ego_heading_x = float(raw_obs.ego_vehicle_state.heading) + math.pi / 2
                heading_vec = np.array(
                    [math.cos(ego_heading_x), math.sin(ego_heading_x)]
                )
                position = position + heading_vec * exp_speed * 0.1
                heading = raw_obs.ego_vehicle_state.heading

        return np.concatenate([position, [heading, 0.1]])

    def reset(self, init_obs):
        self.raw_obs = init_obs  # 直接把time=0.1的obs传进来

        # todo 直接使用ego vehicle的lane_index作为self.lane_index
        self.lane_index = init_obs.ego_vehicle_state.lane_index
        # self.lane_index = lane_index

        self.update_lane_info(self.raw_obs)
        self.target_lane_index = self.lane_index
        self.collision_r = self.cal_collision_r(
            self.raw_obs.ego_vehicle_state.bounding_box
        )
        return 1
        # return self.pack_observation_v1(self.raw_obs)

    def pack_observation_v1(self, raw_obs):
        # todo 确定observation，包括waypoint_obs和neighbor_obs
        fixed_waypoints = raw_obs.waypoint_paths[self.target_lane_index][:5]
        if len(fixed_waypoints) < 5:
            return None  # 如果无法取得5个有效的wp，则放弃决策
        # ego_vehicle_state
        current_raw_obs = raw_obs
        state = current_raw_obs.ego_vehicle_state
        pos = state.position[:2]
        heading = float(state.heading)
        speed = state.speed
        lane_index = state.lane_index
        rotate_M = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )  # heading是与y正半轴的夹角，这样转化是否会出现问题？

        ego_lane_positions = np.array([wp.pos for wp in fixed_waypoints])
        ego_lane_headings = np.array([float(wp.heading) for wp in fixed_waypoints])

        all_lane_rel_position = (
            (ego_lane_positions.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T
        ).reshape(5, 2)
        all_lane_rel_heading = ego_lane_headings - heading
        all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi  # 这是在干嘛
        all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

        EnvInfo_rel_pos_heading = np.zeros((1, 15))
        EnvInfo_speed_limit = np.zeros((1, 1))
        EnvInfo_bounding_box = np.zeros((1, 2))
        EnvInfo_lane_width = np.zeros((1, 1))
        EnvInfo_rel_pos_heading[0, :10] = all_lane_rel_position.reshape(
            10,
        )  # todo 看一下测试环境的wp[0].heading是否始终是0
        EnvInfo_rel_pos_heading[0, 10:] = all_lane_rel_heading.reshape(
            5,
        )
        try:
            speed_limit = current_raw_obs.waypoint_paths[lane_index][0].speed_limit
            # 有时waymo数据集的lane_index会out of range
        except:
            speed_limit = current_raw_obs.waypoint_paths[0][0].speed_limit
        EnvInfo_speed_limit[0, 0] = speed_limit
        EnvInfo_bounding_box[0, 0] = state.bounding_box.length
        EnvInfo_bounding_box[0, 1] = state.bounding_box.width

        EnvInfo = np.concatenate(
            [
                EnvInfo_rel_pos_heading,  # 15
                EnvInfo_speed_limit,  # 1
                EnvInfo_bounding_box,  # 2
            ],
            -1,
        ).astype(np.float32)

        # Neighbor Info
        on_road_neighbors = []
        for neighbor in current_raw_obs.neighborhood_vehicle_states:
            if neighbor.lane_id != "off_lane":
                on_road_neighbors.append(neighbor)
        on_road_neighbors = on_road_neighbors[:50]
        neighbors_pos = np.zeros([50, 2])
        neighbors_bounding_box = np.zeros([50, 2])
        neighbors_speed = np.zeros(50)
        neighbors_heading = np.zeros([50])
        for nb_ind, neighbor in enumerate(on_road_neighbors):
            neighbors_pos[nb_ind] = neighbor.position[:2]
            neighbors_speed[nb_ind] = neighbor.speed
            neighbors_heading[nb_ind] = neighbor.heading
            neighbors_bounding_box[nb_ind, 0] = neighbor.bounding_box.length
            neighbors_bounding_box[nb_ind, 1] = neighbor.bounding_box.width
        nb_num = len(on_road_neighbors)
        neighbors_rel_vel = np.zeros((50, 2))
        neighbors_rel_vel[:nb_num, 0] = (
            -np.sin(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num]
            + np.sin(heading) * speed
        )
        neighbors_rel_vel[:nb_num, 1] = (
            np.cos(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num]
            - np.cos(heading) * speed
        )

        nb_mask = np.all(neighbors_pos == 0, -1).astype(np.float32)

        neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1)) + nb_mask * 1e5
        st = np.argsort(neighbors_dist)[:5]

        NeighborInfo_rel_pos = (neighbors_pos[st] - pos) @ rotate_M.T
        NeighborInfo_rel_vel = (neighbors_rel_vel[st]) @ rotate_M.T
        NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
        NeighborInfo_boundingbox = neighbors_bounding_box[st]

        NeighborInfo = np.concatenate(
            [
                NeighborInfo_rel_pos,  # 2
                NeighborInfo_rel_vel,  # 2
                NeighborInfo_rel_heading,  # 1
                NeighborInfo_boundingbox,  # 2
            ],
            -1,
        ).astype(np.float32)

        if len(on_road_neighbors) < 5:  # padding
            NeighborInfo[len(on_road_neighbors) :] = 0

        wrapped_obs = np.concatenate(
            [
                NeighborInfo.reshape(
                    -1,
                ),  # (25)
                EnvInfo.reshape(
                    -1,
                ),  # (15)
            ]
        )
        return wrapped_obs

from smarts.env.wrappers.format_obs import FormatObs
from collections import namedtuple
import gym
import numpy as np
import math

class ObsWrapper(gym.ObservationWrapper):
    '''
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
    '''
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.np_wrapper = FormatObs(env)
        
        self.preserved_info_single_agent = namedtuple("PreservedInfoSingleAgent", [
            'raw_obs',
            'np_obs',
            
            # lane index info
            'lane_index',
            'all_lane_indeces',
            'masked_all_lane_indeces',
            
            # road index info
            'all_road_indeces',
            
            'target_wps',
            'speed_limit',
            'speed',
            'is_on_goal_lane',
            'humaness',
            'classifier_input',
            'is_turnable',
            'goal_lane',
            'wrapped_obs',
            'road_all_wrong',
            'lane_on_heading',
            'distance_to_goal',
        ])
        self.preserved_info: dict[str, self.preserved_info_single_agent] = {}
        self.target_lane_index = {}
        self.target_road_index = {}
                
        self.last_correct_wp_pos = {}
        self.last_correct_st = {}  
        self.neighbor_info_dim = 5
        self.env_info_dim = 25

    def reset(self):
        self.target_lane_index = {}
        self.target_road_index = {}
        self.last_correct_wp_pos = {}
        self.last_correct_st = {}
        return super().reset()
    
    def step(self, action):
        return super().step(action)

    def cal_rel_vel(self, v1: float, theta1: float, v2: float, theta2: float) -> np.ndarray:
        ''' Calculate v1 relative to v2. '''
        return np.array([
            -np.sin(theta1) * v1 + np.sin(theta2) * v2,
            np.cos(theta1) * v1 - np.cos(theta2) * v2
        ])
        
    def cal_rel_heading(self, heading1: float, heading2: float) -> float:
        ''' Calculate heading1 relative to heading2. '''
        h = heading1 - heading2
        if h >  np.pi: h -= np.pi
        if h < -np.pi: h += np.pi
        return h
        
    def cal_goal_lane(self, np_obs, raw_obs, lane_index, all_lane_indeces):
        goal_lane = np.zeros((3, 3))
        if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"): return goal_lane
        cos_thetas = np.zeros(4)
        for i in range(4):
            y1 = np_obs["waypoints"]["pos"][i, 1][:2] - np_obs["waypoints"]["pos"][i, 0][:2]
            y2 = np_obs["mission"]["goal_pos"][:2] - np_obs["waypoints"]["pos"][i, 0][:2]
            if np.linalg.norm(y2) <= 0.2 or np.linalg.norm(y1) <= 0.2: continue 
            cos_thetas[i] = abs(y1@y2/np.sqrt(y1@y1*y2@y2))
        if cos_thetas.max() > 1 - 0.0001:
            l = all_lane_indeces[cos_thetas.argmax()]
            if l == -1: return goal_lane
            goal_lane[:, 0] = (l < lane_index+np.array([-1,0,1])).astype(np.float32)
            goal_lane[:, 1] = (l == lane_index+np.array([-1,0,1])).astype(np.float32)
            goal_lane[:, 2] = (l > lane_index+np.array([-1,0,1])).astype(np.float32)
        return goal_lane
        
    def get_which_road_the_lane_belongs_to(self, np_obs, mse_threshold=40, horizon=20) -> np.ndarray:
        ''' lane_indeces: [0,0,0,0] -> road_indeces: [0,1,2,-1] '''
        road_indeces = np.zeros(4)
        wp_pos = np_obs["waypoints"]["pos"][:, :horizon, :2]
        rel_wp_pos = (wp_pos - wp_pos[0, :, :])
        delta_rel_wp_pos = rel_wp_pos - rel_wp_pos[:, 0:1, :]
        mask = np.any(wp_pos, -1).reshape((4,20,1))
        delta_rel_wp_pos *= mask
        
        similarity = (delta_rel_wp_pos ** 2).sum((1,2))     
        for i in range(1, 4):
            if np.all(wp_pos[i, ...] == 0):
                road_indeces[i] = -1 # padding
                continue
            for j in range(i):
                if abs(similarity[i] - similarity[j]) < mse_threshold:
                    road_indeces[i] = road_indeces[j] # belongs to the same road
                    break
            else:
                road_indeces[i] = road_indeces[i-1] + 1 # belongs to another road
        return road_indeces
    
    def choose_target_road_index(self, np_obs, all_road_indeces) -> int:
        ''' Choose the road that is the closest to goal. '''
        # if it is an endless mission and there is more than one road, we just need to BAI LAN
        wp_end_points = np_obs["waypoints"]["pos"][:, -1, :2]
        goal_point = np_obs["mission"]["goal_pos"][:2].reshape(1, -1)
        dist = ((goal_point-wp_end_points)**2).sum(-1)
        dist[np.where(all_road_indeces == -1)] = dist.max() + 1
        return int(all_road_indeces[dist.argmin()])
    
    def check_wrong_road(self, np_obs, raw_obs):
        wrong_road_indeces = np.ones(4)
        if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"): return wrong_road_indeces
        goal_point = np_obs["mission"]["goal_pos"][:2].reshape(1, -1)
        wp_pos = np_obs["waypoints"]["pos"][:, :, :2]
        
        for i in range(4):
            last_zero_index = 0
            for j in range(-1, -20, -1):
                if np.any(wp_pos[i, j-1]):
                    last_zero_index = j
                    break
            else:
                last_zero_index = j-1
            if last_zero_index == -20:
                wrong_road_indeces[i] = 0
                continue
            wp_end_points = wp_pos[i, last_zero_index-3:last_zero_index]
            if wp_end_points.shape[0] < 3:
                wrong_road_indeces[i] = 1
                continue
            dist_to_goal = ((goal_point-wp_end_points)**2).sum(-1)
            if not (dist_to_goal[0] > dist_to_goal[1] and 
                    dist_to_goal[1] > dist_to_goal[2]):
                wrong_road_indeces[i] = 0
        return wrong_road_indeces
        
    def get_np_neighbor_info(self, raw_obs, pos, speed, heading, rotate_M):
        neighbor_pos, neighbor_speed, neighbor_heading = [],[],[]
        for neighbor in raw_obs.neighborhood_vehicle_states:
            neighbor_pos.append(neighbor.position[:2])
            neighbor_speed.append(neighbor.speed)
            neighbor_heading.append(neighbor.heading)
            
        if len(neighbor_pos) != 0:
            neighbor_pos = np.concatenate(neighbor_pos).reshape(-1, 2)
            neighbor_speed = np.array(neighbor_speed)
            neighbor_heading = np.array(neighbor_heading)
            
            if neighbor_pos.shape[0] < 5:
                sp = 5 - neighbor_pos.shape[0]
                neighbor_pos = np.concatenate([neighbor_pos, np.ones((sp, 2))*200])
                neighbor_speed = np.concatenate([neighbor_speed, np.ones(sp)*speed])
                neighbor_heading = np.concatenate([neighbor_heading, np.ones(sp)*heading])
        else:
            neighbor_pos = np.ones((5, 2))*200
            neighbor_speed = np.ones(5)*speed
            neighbor_heading = np.ones(5)*heading
        assert neighbor_pos.shape[0] >= 5
        
        NeighborInfo_rel_pos = ((neighbor_pos - pos.reshape(1, 2)) @ rotate_M.T)
        rel_dist = (NeighborInfo_rel_pos**2).sum(-1)
        st = rel_dist.argsort()[:5]
        
        NeighborInfo_rel_pos = NeighborInfo_rel_pos[st]
        neighbor_heading = neighbor_heading[st]
        neighbor_speed = neighbor_speed[st]
        
        neighbors_rel_vel = np.empty((5, 2))
        neighbors_rel_vel[:, 0] = -np.sin(neighbor_heading) * neighbor_speed + np.sin(heading) * speed
        neighbors_rel_vel[:, 1] = np.cos(neighbor_heading) * neighbor_speed - np.cos(heading) * speed
        NeighborInfo_rel_vel = ((neighbors_rel_vel) @ rotate_M.T)
    
        NeighborInfo_rel_heading = (neighbor_heading - heading).reshape(5, 1)
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading >  np.pi)] -= np.pi
        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
        
        return (NeighborInfo_rel_pos, NeighborInfo_rel_vel, NeighborInfo_rel_heading)
        
    def observation(self, all_raw_obs):
        all_np_obs = self.np_wrapper.observation(all_raw_obs)
        
        self.preserved_info = dict.fromkeys(all_raw_obs.keys())
        wrapped_obs = dict.fromkeys(all_raw_obs.keys())
        
        # ! LOG
        success_flag = True
        collision_flag = False
        
        for agent_id in all_raw_obs.keys():
            raw_obs, np_obs = all_raw_obs[agent_id], all_np_obs[agent_id]
            
            # ! LOG
            success_flag = (success_flag and raw_obs.events.reached_goal)
            collision_flag = (collision_flag or len(raw_obs.events.collisions))

            # ego_vehicle_state
            pos = np_obs["ego"]["pos"][:2]
            heading = np_obs["ego"]["heading"]
            speed = np_obs["ego"]["speed"]
            lane_index = np_obs["ego"]["lane_index"]
            jerk_linear = np.linalg.norm(np_obs["ego"]["linear_jerk"])
            jerk_angular = np.linalg.norm(np_obs["ego"]["angular_jerk"])
            humaness = jerk_linear + jerk_angular
            rotate_M = np.array([
                [ np.cos(heading), np.sin(heading)], 
                [-np.sin(heading), np.cos(heading)]]
            )
            if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"):
                distance_to_goal = -1 
            else:
                distance_to_goal = np.sqrt(((raw_obs.ego_vehicle_state.mission.goal.position[:2] - pos)**2).sum())

            all_lane_indeces = np_obs["waypoints"]["lane_index"][:, 0]
            all_road_indeces = self.get_which_road_the_lane_belongs_to(np_obs)
            self.target_road_index[agent_id] = self.choose_target_road_index(np_obs, all_road_indeces)
            all_lane_speed_limit = np_obs["waypoints"]["speed_limit"][:, 0].reshape(4, 1)
            all_lane_width = np_obs["waypoints"]["lane_width"][:, 0].reshape(4, 1)
            all_lane_position = np_obs["waypoints"]["pos"][:, :, :2].reshape(4, 20, 2)
            all_lane_heading = np_obs["waypoints"]["heading"][:, :].reshape(4, 20)
            
            all_lane_rel_position = ((all_lane_position.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(4, 20, 2)
            all_lane_rel_heading = (all_lane_heading - heading)
            all_lane_rel_heading[np.where(all_lane_rel_heading >  np.pi)] -= np.pi
            all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

            # check wrong road
            wrong_road_indeces = self.check_wrong_road(np_obs, raw_obs)
            road_all_wrong = True if not np.any(wrong_road_indeces) else False
            if not road_all_wrong:
                self.last_correct_wp_pos[agent_id] = np_obs["waypoints"]["pos"][:, :, :2]
            else:
                # ! LOG
                print("[INFO] all roads are wrong.")

            # Env Info
            st = [0]*3
            # ! Only consider lanes belong to the target road
            masked_all_lane_indeces = all_lane_indeces.copy()
            masked_all_lane_indeces[np.where(all_road_indeces!=self.target_road_index[agent_id])] = -1
            if lane_index not in masked_all_lane_indeces:
                lane_index = masked_all_lane_indeces[np.where(masked_all_lane_indeces!=-1)[0][0]].item()
            if agent_id not in self.target_lane_index.keys() or self.target_lane_index[agent_id] not in masked_all_lane_indeces:
                self.target_lane_index[agent_id] = lane_index   
                         
            EnvInfo_is_turnable = np.zeros((3, 1))
            if lane_index - 1 in masked_all_lane_indeces and lane_index > 0: 
                EnvInfo_is_turnable[0] = 1.0
                st[0] = np.where(masked_all_lane_indeces == lane_index-1)[0][0].item()
            if lane_index + 1 in masked_all_lane_indeces: 
                EnvInfo_is_turnable[2] = 1.0
                st[2] = np.where(masked_all_lane_indeces == lane_index+1)[0][0].item()
            EnvInfo_is_turnable[1] = 1.0
            st[1] = np.where(masked_all_lane_indeces == lane_index)[0][0].item()
                        
            if not road_all_wrong:
                self.last_correct_st[agent_id] = st[1]
            
            speed_limit = all_lane_speed_limit[st[1]]
            lane_on_heading = all_lane_heading[st[1], 0]

            EnvInfo_rel_pos_heading = np.zeros((3, 15))
            EnvInfo_rel_pos_heading_classifier = np.zeros((3, 60))
            EnvInfo_speed_limit = np.zeros((3, 1))
            EnvInfo_width = np.zeros((3, 1))
            for i in range(3):
                if (i==0 and EnvInfo_is_turnable[0] == 0) or (i==2 and EnvInfo_is_turnable[1]==0): continue
                EnvInfo_rel_pos_heading[i, :10] = all_lane_rel_position[st[i]][:5,:].reshape(10,)
                EnvInfo_rel_pos_heading[i, 10:] = all_lane_rel_heading[st[i]][:5].reshape(5,)
                EnvInfo_rel_pos_heading_classifier[i, :40] = all_lane_rel_position[st[i]].reshape(40,)
                EnvInfo_rel_pos_heading_classifier[i, 40:] = all_lane_rel_heading[st[i]].reshape(20,)
                EnvInfo_speed_limit[i] = all_lane_speed_limit[st[i]]
                EnvInfo_width[i] = all_lane_width[st[i]]
                
            EnvInfo_is_target = np.zeros((3, 1))
            if   self.target_lane_index[agent_id] < lane_index: EnvInfo_is_target[0] = 1.0
            elif self.target_lane_index[agent_id] > lane_index: EnvInfo_is_target[2] = 1.0
            else : EnvInfo_is_target[1] = 1.0
            
            EnvInfo_is_goal = self.cal_goal_lane(np_obs, raw_obs, lane_index, masked_all_lane_indeces).reshape(3, 3)
            is_on_goal_lane = EnvInfo_is_goal[1, 1]
            
            EnvInfo_index = np.eye(3).reshape(3, 3)
            
            EnvInfo = np.concatenate([
                EnvInfo_rel_pos_heading, # 15
                EnvInfo_speed_limit,     # 1
                EnvInfo_width,           # 1
                EnvInfo_is_target,       # 1
                EnvInfo_is_goal,         # 3
                EnvInfo_is_turnable,     # 1
                EnvInfo_index,           # 3
            ], -1).astype(np.float32)
            
            EnvInfo_classifier = np.concatenate([
                EnvInfo_rel_pos_heading_classifier, # 60
                EnvInfo_speed_limit,                # 1
                EnvInfo_width,                      # 1
                EnvInfo_is_target,                  # 1
                EnvInfo_is_goal,                    # 3
                EnvInfo_is_turnable,                # 1
                EnvInfo_index,                      # 3
            ], -1).astype(np.float32)
            
            # Neighbor Info
            
            (NeighborInfo_rel_pos, NeighborInfo_rel_vel, NeighborInfo_rel_heading) = \
                self.get_np_neighbor_info(raw_obs, pos, speed, heading, rotate_M)
            NeighborInfo = np.concatenate([
                NeighborInfo_rel_pos,       # 2
                NeighborInfo_rel_vel,       # 2
                NeighborInfo_rel_heading,   # 1
                # NeighborInfo_boundingbox,   # 2
            ], -1).astype(np.float32)
            

            # preserved_info
            wp_mask = np.abs(np.all(np_obs["waypoints"]["pos"][:, :4] == 0, -1) - 1).sum(1).reshape(4, 1)
            target_wps = np_obs["waypoints"]["pos"][:, :4, :].sum(1) / (wp_mask + 1e-8)
            wrapped_obs[agent_id] = np.concatenate([
                NeighborInfo.reshape(-1,), # (5, 5)
                EnvInfo.reshape(-1,),      # (3, 19)
            ])
            
            self.preserved_info[agent_id] = self.preserved_info_single_agent(
                raw_obs = raw_obs,
                lane_index = lane_index,
                all_lane_indeces = all_lane_indeces,
                target_wps = target_wps,
                speed_limit = speed_limit,
                np_obs = np_obs,
                is_on_goal_lane = is_on_goal_lane,
                speed = speed,
                humaness = humaness,
                classifier_input =  np.concatenate([
                    NeighborInfo.reshape(-1,),            # (5, 5)
                    EnvInfo_classifier.reshape(-1,),      # (3, 70)
                ]), # dim: 235
                is_turnable = EnvInfo_is_turnable,
                goal_lane = EnvInfo_is_goal[1, :],
                wrapped_obs = wrapped_obs[agent_id],
                all_road_indeces=all_road_indeces,
                masked_all_lane_indeces=masked_all_lane_indeces,
                road_all_wrong = road_all_wrong,
                lane_on_heading = lane_on_heading,
                distance_to_goal = distance_to_goal,
            )
            
            wrapped_obs[agent_id] = np.concatenate([
                wrapped_obs[agent_id],
                self.preserved_info[agent_id].classifier_input
            ], -1)
           
        # ! LOG
        if success_flag: print("[INFO] Success")
        if collision_flag: print("[INFO] Collision")
        return wrapped_obs

class EnvWrapper(gym.Wrapper):
    def __init__(self, env: ObsWrapper):
        super().__init__(env)
        self.env = env
        self.time_cost = -1.0
        self.crash_times = 0
        
        self.action_space_n = 11
        self.observation_space_shape = (
            self.env.neighbor_info_dim*5+self.env.env_info_dim*3,
        )
        
        self._rule_stop_cnt = 0
        self._is_on_goal_lane_last = -1
        
    def is_in_safe_box(self, agent_id: str, turn_left: bool):
        o = self.env.preserved_info[agent_id].wrapped_obs
        neighbor_x = o[np.arange(5)*3]
        neighbor_y = o[np.arange(5)*3+1]
        neighbor_h = o[np.arange(5)*3+4]
        
        neighbor_vx = o[np.arange(5)*3+2]
        neighbor_vy = o[np.arange(5)*3+3]
        neighbor_v = np.sqrt(neighbor_vx**2+neighbor_vy**2)
        
        for i in range(5):
            if turn_left:
                if  (neighbor_x[i] < -1.6 and neighbor_x[i] > -4.0) and \
                    ((neighbor_y[i] <  0.0 and neighbor_y[i] > min(-9.0, -1.5*neighbor_v[i].item())) or \
                    (neighbor_y[i] >  0.0 and neighbor_y[i] <  6.0)) and \
                    (neighbor_h[i] < 0.05):
                        return False
            else:
                if  (neighbor_x[i] >  1.6 and neighbor_x[i] <  4.0) and \
                    ((neighbor_y[i] <  0.0 and neighbor_y[i] > min(-9.0, -1.5*neighbor_v[i].item())) or \
                    (neighbor_y[i] >  0.0 and neighbor_y[i] <  6.0)) and \
                    (neighbor_h[i] < 0.05):
                        return False
        else:
            return True
        
    def is_parallel(self, agent_id: str):
        o = self.env.preserved_info[agent_id].wrapped_obs
        neighbor_x = o[np.arange(5)*3]
        neighbor_y = o[np.arange(5)*3+1]
        neighbor_h = o[np.arange(5)*3+4]

        for i in range(5):
            if  (neighbor_x[i] < -1.6 and neighbor_x[i] > -4.0) and \
                ((neighbor_y[i] < 0.0 and neighbor_y[i] > -2.0) or \
                (neighbor_y[i] > 0.0 and neighbor_y[i] < 2.0)) and \
                (neighbor_h[i] < 0.05):
                    return True
        else:
            return False
        
    def collision_forecast(self, vehicle_state1, vehicle_state2, l_front=5, l_back=0, w_left=1.25, w_right=1.25, steps=5):
        v1, v2 = vehicle_state1.speed, vehicle_state2.speed
        theta1, theta2 = vehicle_state1.heading + math.pi / 2, vehicle_state2.heading + math.pi / 2
        v1_vec, v2_vec = v1 * np.array([math.cos(theta1), math.sin(theta1)]), \
                        v2 * np.array([math.cos(theta2), math.sin(theta2)])
        init_pos1, init_pos2 = vehicle_state1.position[:2], vehicle_state2.position[:2]
        bound1, bound2 = vehicle_state1.bounding_box, vehicle_state2.bounding_box
        # l1, w1, l2, w2 = bound1.length, bound1.width, bound2.length, bound2.width
        l1, w1, l2, w2 = bound1.length, bound1.width, bound2.length, bound2.width
        l2_vec = l2 / 2 * np.array([math.cos(theta2), math.sin(theta2)])
        w2_vec = w2 / 2 * np.array([math.sin(theta2), -1 * math.cos(theta2)])

        l1_front_vec, l1_back_vec = (l1 / 2 + l_front) * np.array([math.cos(theta1), math.sin(theta1)]), \
                                    (l1 / 2 + l_back) * np.array([math.cos(theta1), math.sin(theta1)])
        w1_left_vec = (w1 / 2 + w_left) * np.array([math.sin(theta1), -1 * math.cos(theta1)])
        w1_right_vec = (w1 / 2 + w_right) * np.array([math.sin(theta1), -1 * math.cos(theta1)])

        for step in range(0, steps + 1, 1):
            t = 0.1 * step
            pos1, pos2 = init_pos1 + v1_vec * t, init_pos2 + v2_vec * t
            # calculate bounding points
            bps_1 = [
                pos1 + l1_front_vec - w1_left_vec,
                pos1 + l1_front_vec + w1_right_vec,
                pos1 - l1_back_vec - w1_left_vec,
                pos1 - l1_back_vec + w1_right_vec
            ]
            bps_2 = [
                pos2 + l2_vec + w2_vec,
                pos2 + l2_vec - w2_vec,
                pos2 - l2_vec + w2_vec,
                pos2 - l2_vec - w2_vec
            ]
            bps_1_front, bps1_right = bps_1[:2], [bps_1[0], bps_1[2]]

            for bp in bps_2:
                if np.dot(bp - bps_1_front[0], bps_1_front[0] - bps_1_front[1]) * \
                        np.dot(bp - bps_1_front[1], bps_1_front[0] - bps_1_front[1]) <= 0 \
                        and np.dot(bp - bps1_right[0], bps1_right[0] - bps1_right[1]) * \
                        np.dot(bp - bps1_right[1], bps1_right[0] - bps1_right[1]) <= 0:
                    return True
        return False
        
    def reset(self):
        self.crash_times = 0
        self._rule_stop_cnt = 0
        self._is_on_goal_lane_last = -1
        raw_obs = self.env.reset()
        return raw_obs
    
    def step(self, raw_act):
        if self.agents_id == {}: wrapped_act = {}
        else: wrapped_act = self.pack_action(raw_act)
        obs, reward, done, info = self.env.step(wrapped_act)
        return obs, reward, done, info
        
    @property
    def agents_id(self):
        if self.env.preserved_info is not None:
            return self.env.preserved_info.keys()
        else: 
            return {}
        
    def cal_collision_r(self, bounding_box):
        return np.sqrt(bounding_box.length ** 2 + bounding_box.width ** 2)
    
    def cal_distance(self, pos1, pos2):
        return np.sqrt(((pos1 - pos2)**2).sum())

    def pack_action(self, action):
        wrapped_act = dict.fromkeys(self.agents_id)
        for agent_id in self.agents_id:

            raw_obs = self.env.preserved_info[agent_id].raw_obs
            all_lane_indeces = self.env.preserved_info[agent_id].masked_all_lane_indeces
            
            target_wps = self.env.preserved_info[agent_id].target_wps
            exp_speed = min(self.env.preserved_info[agent_id].speed_limit, 13.88)
            speed = self.env.preserved_info[agent_id].speed
            pos = raw_obs.ego_vehicle_state.position[:2]
            heading = raw_obs.ego_vehicle_state.heading - 0.0
            acc = 0
        
        # ? Rule-based keep correct lane 0
            use_preserved_target_wp = False
            if self.env.preserved_info[agent_id].road_all_wrong and agent_id in self.env.last_correct_wp_pos.keys():
                last_correct_wp_pos = self.env.last_correct_wp_pos[agent_id]
                pos_now = self.env.preserved_info[agent_id].np_obs["ego"]["pos"][:2].reshape(1,1,-1)
                st = [0, 0]
                dist_to_pos_now = ((pos_now - last_correct_wp_pos)**2).sum(-1)
                st[0] = self.env.last_correct_st[agent_id]
                st[1] = dist_to_pos_now[st[0]].argmin()
                target_wp = last_correct_wp_pos[st[0]][st[1]+1 : st[1]+4]
                if target_wp.shape[0] == 0:
                    # ! LOG
                    print("[INFO] No useful preserved waypoints exist.")
                    pass # ! BAI LAN
                else:
                    target_wp = target_wp.mean(0)
                    use_preserved_target_wp = True
                    
            # ? Rule-based stopping condition 1
            dist_min = 2
            collision_r = self.cal_collision_r(raw_obs.ego_vehicle_state.bounding_box)
            lane_id = raw_obs.ego_vehicle_state.lane_id
            for neighbor in raw_obs.neighborhood_vehicle_states:
                if neighbor.lane_id == lane_id and neighbor.lane_position.s > raw_obs.ego_vehicle_state.lane_position.s:
                    neighbor_dists = np.clip(
                        self.cal_distance(
                            pos,
                            neighbor.position[:2]) -
                        self.cal_collision_r(neighbor.bounding_box) - 
                        collision_r, 0, None
                    )   
                    if neighbor_dists < dist_min:
                        dist_min = neighbor_dists
                        exp_speed = min(neighbor.speed, exp_speed)
                        acc = 1.2
            if dist_min < 0.1:
                action[agent_id] = 9
                
            # ? Rule-based stopping condition 2
            ego = self.env.preserved_info[agent_id].raw_obs.ego_vehicle_state
            for neighbor in self.env.preserved_info[agent_id].raw_obs.neighborhood_vehicle_states:
                if self.collision_forecast(ego, neighbor):
                    self._rule_stop_cnt += 1
                    if self._rule_stop_cnt >= 5:
                        if self._rule_stop_cnt >= 8: self._rule_stop_cnt = 0
                        break
                    if action[agent_id] not in [10]:
                        exp_speed = 0.0
                        acc = 1.2
                    break
            # ? Rule-based lane changing condition 3
            # ! Turn off when training
            else:
                self._rule_stop_cnt = 0
                goal_lane = self.env.preserved_info[agent_id].goal_lane
                if self.env.preserved_info[agent_id].lane_index == self.env.target_lane_index[agent_id] and \
                    abs(heading - self.env.preserved_info[agent_id].lane_on_heading) < 0.05:                    
                    if action[agent_id] in [7, 8, 9]:
                        if goal_lane[2] and self.is_in_safe_box(agent_id, True): action[agent_id] = 2
                        elif goal_lane[0] and self.is_in_safe_box(agent_id, False): action[agent_id] = 5
                    elif action[agent_id] in [0, 1, 2]:
                        if not self.is_in_safe_box(agent_id, True): action[agent_id] += 7
                    elif action[agent_id] in [3, 4, 5]:
                        if not self.is_in_safe_box(agent_id, False): action[agent_id] += 4
                distance_to_goal = self.env.preserved_info[agent_id].distance_to_goal
                if not goal_lane[1] and distance_to_goal < 100 and distance_to_goal > 0 and action[agent_id] in [7,8,9]:
                    exp_speed *= distance_to_goal / 100.0
            
            # ? Rule-based slow down 4
            if self.is_parallel(agent_id): 
                exp_speed *= 0.6
                # print(f'{agent_id} slow down')
                    
            if use_preserved_target_wp: action[agent_id] = 8
            # keep_lane
            if action[agent_id] in [6, 7, 8, 9, 10]:
                exp_speed *= [0.0, 0.4, 0.7, 1.0, -0.2][action[agent_id] - 6]

            # change_lane_left
            elif action[agent_id] in [0, 1, 2]:
                exp_speed *= [0.4, 0.7, 1.0][action[agent_id] - 0]
                if self.env.target_lane_index[agent_id]+1 in all_lane_indeces and \
                    abs(heading - self.env.preserved_info[agent_id].lane_on_heading) < 0.05:
                    self.env.target_lane_index[agent_id] += 1
            
            # change_lane_right
            elif action[agent_id] in [3, 4, 5]:
                exp_speed *= [0.4, 0.7, 1.0][action[agent_id] - 3]
                if self.env.target_lane_index[agent_id]>0 and \
                    self.env.target_lane_index[agent_id]-1 in all_lane_indeces and \
                        abs(heading - self.env.preserved_info[agent_id].lane_on_heading) < 0.05:
                    self.env.target_lane_index[agent_id] -= 1
                    
            if not use_preserved_target_wp:
                st = np.where(all_lane_indeces == self.env.target_lane_index[agent_id])[0][0].item()
                target_wp = target_wps[st][:2]
            
            delta_pos = target_wp - pos
            delta_pos_dist = self.cal_distance(target_wp, pos)
            
            acc = 0.5 if exp_speed > speed else max(0.8, acc)
            # smooth expected speed
            if exp_speed > speed: exp_speed = np.clip(exp_speed, None, speed + acc)
            if exp_speed < speed: exp_speed = np.clip(exp_speed, speed - acc, None)
            
            exp_heading = np.arctan2(- delta_pos[0], delta_pos[1])
            hc_id = np.abs(2*np.pi*(np.arange(3)-1) + exp_heading - heading).argmin()
            exp_heading += (hc_id-1)*2*np.pi
            heading = np.clip(
                exp_heading,
                heading - 0.15, heading + 0.15
            )
            new_pos = pos + delta_pos / delta_pos_dist * min(exp_speed, 13.88) * 0.1
            wrapped_act[agent_id] = np.concatenate([new_pos, [heading, 0.1]])
        
        return wrapped_act
    



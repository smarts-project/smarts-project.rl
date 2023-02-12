from typing import Any, Dict, List, Tuple, Union

import numpy as np

from smarts.core.events import Events
from smarts.core.observations import (
    EgoVehicleObservation,
    Observation,
    VehicleObservation,
)
from smarts.core.road_map import Waypoint

_LIDAR_SHP = 300
_NEIGHBOR_SHP = 10
_WAYPOINT_SHP = (4, 20)

"""
Observations in numpy array format, suitable for vectorized processing.

StdObs = dict({
    Total distance travelled in meters.
    "distance_travelled": np.float32
    
    Ego vehicle state, with the following attributes.
    "ego_vehicle_state": dict({
        "angular_acceleration": 
            Angular acceleration vector. Requires `accelerometer` attribute 
            enabled in AgentInterface, else absent. shape=(3,). dtype=np.float32.
        "angular_jerk":
            Angular jerk vector. Requires `accelerometer` attribute enabled in
            AgentInterface, else absent. shape=(3,). dtype=np.float32.
        "angular_velocity":
            Angular velocity vector. shape=(3,). dtype=np.float32).
        "box":
            Length, width, and height of the vehicle bounding box. shape=(3,).
            dtype=np.float32.
        "heading":
            Vehicle heading in radians [-pi, pi]. dtype=np.float32.
        "lane_index":
            Vehicle's lane number. Rightmost lane has index 0 and increases 
            towards left. dtype=np.int8.
        "linear_acceleration":
            Vehicle acceleration in x, y, and z axes. Requires `accelerometer`
            attribute enabled in AgentInterface, else absent. shape=(3,).
            dtype=np.float32.
        "linear_jerk":
            Linear jerk vector. Requires `accelerometer` attribute enabled in
            AgentInterface, else absent. shape=(3,). dtype=np.float32.
        "linear_velocity":
            Vehicle velocity in x, y, and z axes. shape=(3,). dtype=np.float32.
        "pos":
            Coordinate of the center of the vehicle bounding box's bottom plane.
            shape=(3,). dtype=np.float64.
        "speed":
            Vehicle speed in m/s. dtype=np.float32.
        "steering":
            Angle of front wheels in radians [-pi, pi]. dtype=np.float32.
        "yaw_rate":
            Rotation speed around vertical axis in rad/s [0, 2pi].
            dtype=np.float32.
    )}
    
    A dictionary of event markers.
    "events": dict({
        "agents_alive_done":
            1 if `DoneCriteria.agents_alive` is triggered, else 0.
        "collisions":
            1 if any collisions occurred with ego vehicle, else 0.
        "not_moving":
            1 if `DoneCriteria.not_moving` is triggered, else 0.
        "off_road":
            1 if ego vehicle drives off road, else 0.
        "off_route":
            1 if ego vehicle drives off mission route, else 0.
        "on_shoulder":
            1 if ego vehicle drives on road shoulder, else 0.
        "reached_goal":
            1 if ego vehicle reaches its goal, else 0.
        "reached_max_episode_steps":
            1 if maximum episode steps reached, else 0.
        "wrong_way":
            1 if ego vehicle drives in the wrong traffic direction, else 0.
    })

    Lidar point cloud, with the following attributes.
    "lidar_point_cloud": dict({
        "hit":
            Binary array. 1 if an object is hit, else 0. shape(300,).
        "point_cloud":
            Coordinates of lidar point cloud. shape=(300,3). dtype=np.float64.
        "ray_origin":
            Ray origin coordinates. shape=(300,3). dtype=np.float64.
        "ray_vector":
            Ray vectors. shape=(300,3). dtype=np.float64.
    })
    
    Mission details for the ego agent.
    "mission": dict({
        "goal_pos":
            Achieve goal by reaching the end position. Defaults to np.array([0,0,0])
            for no mission. shape=(3,). dtype=np.float64. 
    })

    Feature array of 10 nearest neighborhood vehicles. If nearest neighbor
    vehicles are insufficient, default feature values are padded.
    "neighborhood_vehicle_states": dict({
        "box":
            Bounding box of neighbor vehicles. Defaults to np.array([0,0,0]) per
            vehicle. shape=(10,3). dtype=np.float32.
        "heading":
            Heading of neighbor vehicles in radians [-pi, pi]. Defaults to
            np.array([0]) per vehicle. shape=(10,). dtype=np.float32.
        "lane_index":
            Lane number of neighbor vehicles. Defaults to np.array([0]) per
            vehicle. shape=(10,). dtype=np.int8.
        "pos":
            Coordinate of the center of neighbor vehicles' bounding box's bottom
            plane. Defaults to np.array([0,0,0]) per vehicle. shape=(10,3).
            dtype=np.float64.
        "speed":
            Speed of neighbor vehicles in m/s. Defaults to np.array([0]) per
            vehicle. shape=(10,). dtype=np.float32.
    })

    Feature array of 20 waypoints ahead or in the mission route, from the 
    nearest 4 lanes. If lanes or waypoints ahead are insufficient, default 
    values are padded.
    "waypoint_paths": dict({
        "heading":
            Lane heading angle at a waypoint in radians [-pi, pi]. Defaults to
            np.array([0]) per waypoint. shape=(4,20). dtype=np.float32.
        "lane_index":
            Lane number at a waypoint. Defaults to np.array([0]) per waypoint.
            shape=(4,20). dtype=np.int8.
        "lane_width":
            Lane width at a waypoint in meters. Defaults to np.array([0]) per
            waypoint. shape=(4,20). dtype=np.float32.
        "pos":
            Coordinate of a waypoint. Defaults to np.array([0,0,0]).
            shape=(4,20,3). dtype=np.float64.
        "speed_limit":
            Lane speed limit at a waypoint in m/s. shape=(4,20). dtype=np.float32.
    }) 

})
"""


def formatter(observation: Observation) -> Dict[str, Any]:
    """Converts SMARTS observations to gym-compliant vectorized
    observations.
    """
    space = [
        "distance_travelled",
        "ego_vehicle_state",
        "events",
        "mission",
        "waypoint_paths",
        "neighborhood_vehicle_states",
        "lidar_point_cloud",
    ]
    wrapped_ob = {}
    for stdob in space:
        func = globals()[f"_std_{stdob}"]
        if stdob == "ego_vehicle_state":
            val = func(getattr(observation, stdob), True)
        elif stdob == "mission":
            val = func(getattr(observation, "ego_vehicle_state"))
        else:
            val = func(getattr(observation, stdob))
        wrapped_ob.update({stdob: val})

    return wrapped_ob


def _std_distance_travelled(val: float) -> float:
    return np.float32(val)


def _std_ego_vehicle_state(
    val: EgoVehicleObservation, accelerometer: bool
) -> Dict[str, Union[np.int8, np.float32, np.ndarray]]:

    std_ego = {
        "angular_velocity": val.angular_velocity.astype(np.float32),
        "box": np.array(val.bounding_box.as_lwh).astype(np.float32),
        "heading": np.float32(val.heading),
        "lane_index": np.int8(val.lane_index),
        "linear_velocity": val.linear_velocity.astype(np.float32),
        "pos": val.position.astype(np.float64),
        "speed": np.float32(val.speed),
        "steering": np.float32(val.steering),
        "yaw_rate": np.float32(val.yaw_rate) if val.yaw_rate else np.float32(0),
    }

    if accelerometer:
        std_ego.update(
            {
                "angular_acceleration": val.angular_acceleration.astype(np.float32),
                "angular_jerk": val.angular_jerk.astype(np.float32),
                "linear_acceleration": val.linear_acceleration.astype(np.float32),
                "linear_jerk": val.linear_jerk.astype(np.float32),
            }
        )

    return std_ego


def _std_events(val: Events) -> Dict[str, int]:
    return {
        "agents_alive_done": np.int64(val.agents_alive_done),
        "collisions": np.int64(len(val.collisions) > 0),
        "not_moving": np.int64(val.not_moving),
        "off_road": np.int64(val.off_road),
        "off_route": np.int64(val.off_route),
        "on_shoulder": np.int64(val.on_shoulder),
        "reached_goal": np.int64(val.reached_goal),
        "reached_max_episode_steps": np.int64(val.reached_max_episode_steps),
        "wrong_way": np.int64(val.wrong_way),
    }


def _std_lidar_point_cloud(
    val: Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
) -> Dict[str, np.ndarray]:

    des_shp = _LIDAR_SHP
    hit = np.array(val[1], dtype=np.int8)
    point_cloud = np.array(val[0], dtype=np.float64)
    point_cloud = np.nan_to_num(
        point_cloud,
        copy=False,
        nan=np.float64(0),
        posinf=np.float64(0),
        neginf=np.float64(0),
    )
    ray_origin, ray_vector = zip(*(val[2]))
    ray_origin = np.array(ray_origin, np.float64)
    ray_vector = np.array(ray_vector, np.float64)

    try:
        assert hit.shape == (des_shp,)
        assert point_cloud.shape == (des_shp, 3)
        assert ray_origin.shape == (des_shp, 3)
        assert ray_vector.shape == (des_shp, 3)
    except:
        raise Exception("Internal Error: Mismatched lidar point cloud shape.")

    return {
        "hit": hit,
        "point_cloud": point_cloud,
        "ray_origin": ray_origin,
        "ray_vector": ray_vector,
    }


def _std_mission(val: EgoVehicleObservation) -> Dict[str, np.ndarray]:

    if hasattr(val.mission.goal, "position"):
        goal_pos = np.array(val.mission.goal.position, dtype=np.float64)
    else:
        goal_pos = np.zeros((3,), dtype=np.float64)

    return {"goal_pos": goal_pos}


def _std_neighborhood_vehicle_states(
    nghbs: List[VehicleObservation],
) -> Dict[str, np.ndarray]:

    des_shp = _NEIGHBOR_SHP
    rcv_shp = len(nghbs)
    pad_shp = 0 if des_shp - rcv_shp < 0 else des_shp - rcv_shp

    if rcv_shp == 0:
        return {
            "box": np.zeros((des_shp, 3), dtype=np.float32),
            "heading": np.zeros((des_shp,), dtype=np.float32),
            "lane_index": np.zeros((des_shp,), dtype=np.int8),
            "pos": np.zeros((des_shp, 3), dtype=np.float64),
            "speed": np.zeros((des_shp,), dtype=np.float32),
        }

    nghbs = [
        (
            nghb.bounding_box.as_lwh,
            nghb.heading,
            nghb.lane_index,
            nghb.position,
            nghb.speed,
        )
        for nghb in nghbs[:des_shp]
    ]
    box, heading, lane_index, pos, speed = zip(*nghbs)

    box = np.array(box, dtype=np.float32)
    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    pos = np.array(pos, dtype=np.float64)
    speed = np.array(speed, dtype=np.float32)

    # fmt: off
    box = np.pad(box, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    heading = np.pad(heading, ((0,pad_shp)), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp)), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    speed = np.pad(speed, ((0,pad_shp)), mode='constant', constant_values=0)
    # fmt: on

    return {
        "box": box,
        "heading": heading,
        "lane_index": lane_index,
        "pos": pos,
        "speed": speed,
    }


def _std_waypoint_paths(
    rcv_paths: List[List[Waypoint]],
) -> Dict[str, np.ndarray]:

    # Truncate all paths to be of the same length
    min_len = min(map(len, rcv_paths))
    trunc_paths = list(map(lambda x: x[:min_len], rcv_paths))

    des_shp = _WAYPOINT_SHP
    rcv_shp = (len(trunc_paths), len(trunc_paths[0]))
    pad_shp = [0 if des - rcv < 0 else des - rcv for des, rcv in zip(des_shp, rcv_shp)]

    def extract_elem(waypoint):
        return (
            waypoint.heading,
            waypoint.lane_index,
            waypoint.lane_width,
            waypoint.pos,
            waypoint.speed_limit,
        )

    paths = [
        map(extract_elem, path[: des_shp[1]]) for path in trunc_paths[: des_shp[0]]
    ]
    heading, lane_index, lane_width, pos, speed_limit = zip(
        *[zip(*path) for path in paths]
    )

    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    lane_width = np.array(lane_width, dtype=np.float32)
    pos = np.array(pos, dtype=np.float64)
    speed_limit = np.array(speed_limit, dtype=np.float32)

    # fmt: off
    heading = np.pad(heading, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_width = np.pad(lane_width, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp[0]),(0,pad_shp[1]),(0,1)), mode='constant', constant_values=0)
    speed_limit = np.pad(speed_limit, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    # fmt: on

    return {
        "heading": heading,
        "lane_index": lane_index,
        "lane_width": lane_width,
        "pos": pos,
        "speed_limit": speed_limit,
    }

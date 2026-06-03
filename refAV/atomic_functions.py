"""
The complete list of functions that the LLM has access to. The LLM prompt directly reads the 
function headers and docstrings to give the LLM context on how to use the functions.

There are several things to note if you want to develop more functions yourself. 
First, the docstrings and typing do not reflect what is actually passed into these functions.
This is done to simplify logic for the atomic function developer while keeping the API intuitive to use.

Any function decorated with @composable takes in a track_uuid and returns a list of timestamps.
Any function decorated with @composable_relational takes in a track_uuid and list of candidate_uuids and 
returns a tuple of a list of timestamps and a dict keyed by candidate_uuids with list of timestamp values.
"""

import json
import numpy as np
from pathlib import Path
from typing import Literal
from copy import deepcopy
import inspect
import pandas as pd
import math

import refAV.paths as paths
from refAV.utils import (
    cache_manager, composable, composable_relational, #global cache_manager and decorators
    get_cuboid_from_uuid, get_ego_SE3, get_ego_uuid, get_log_split,
    get_map, get_nth_pos_deriv, get_nth_radial_deriv,
    get_nth_yaw_deriv, get_pedestrian_crossings,
    get_lane_segments, get_lane_segments_batch, get_pos_within_lane, get_road_side, get_scenario_lanes,
    get_scenario_timestamps, get_timestamps, get_uuids_of_category, get_lane_orientation,
    get_semantic_lane, cuboid_distance, cuboid_distance_batch,
    get_cuboid_vertices_batch, yaw_fallback_dirs, to_scenario_dict,
    _is_point_in_polygon_batch, _opposite_across_road,
    unwrap_func, dilate_convex_polygon, polygons_overlap, is_point_in_polygon,
    swap_keys_and_listed_values, has_free_will, at_stop_sign_, remove_empty_branches,
    scenario_at_timestamps, reconstruct_track_dict, create_mining_pkl,
    post_process_scenario, get_object, get_img_crops, get_best_crop,
    get_context_annotations, get_ego_annotations, get_turn_direction,
    get_median_polygons, _visual_filter,
    get_subcategory_text_embedding, get_siglip_logit_params, get_category_score_maps)
from tools.scene_context_extraction.src.schema import CONTEXT_SCHEMA as _CONTEXT_SCHEMA
from shapely.geometry import Point as _ShPoint
from functools import lru_cache


_PER_CAMERA_INFRA = frozenset(_CONTEXT_SCHEMA["infra"])
_EGO_INFRA = frozenset(_CONTEXT_SCHEMA["ego"])


@composable_relational
@cache_manager.create_cache('has_objects_in_relative_direction')
def has_objects_in_relative_direction(
    track_candidates:dict,
    related_candidates:dict, 
    log_dir:Path, 
    direction:Literal["forward", "backward", "left", "right"], 
    min_number:int=1, 
    max_number:int=np.inf, 
    within_distance:float=50, 
    lateral_thresh:float=np.inf) -> dict:
    """
    Identifies tracked objects with at least the minimum number of related candidates in the specified direction.
    If the minimum number is met, will create relationships equal to the max_number of closest objects. 

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        related_candidates: Candidates to check for in direction (scenario dictionary).
        log_dir: Path to scenario logs.
        direction: Direction to analyze from the track's point of view ('forward', 'backward', 'left', 'right').
        min_number: Minimum number of objects to identify in the direction per timestamp. Defaults to 1.
        max_number: Maximum number of objects to identify in the direction per timestamp. Defaults to infinity.
        within_distance: Maximum distance for considering an object in the direction. Defaults to infinity.
        lateral_thresh: Maximum lateral distance the related object can be from the sides of the tracked object. Defaults to infinity.

    Returns:
        dict: 
            A scenario dictionary where keys are track UUIDs and values are dictionaries containing related candidate UUIDs 
            and lists of timestamps when the condition is met for that relative direction.

    Example:
        vehicles_with_peds_in_front = has_objects_in_relative_direction(vehicles, pedestrians, log_dir, direction='forward', min_number=2)
    """

    track_uuid = track_candidates
    candidate_uuids = related_candidates

    if track_uuid == get_ego_uuid(log_dir):
        #Ford Fusion dimensions offset from ego_coordinate frame
        track_width = 1
        track_front = 4.877/2 + 1.422
        track_back = 4.877 - (4.877/2 + 1.422)
    else:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
        track_width = track_cuboid.width_m/2
        track_front = track_cuboid.length_m/2
        track_back = -track_cuboid.length_m/2
    
    timestamps_with_objects = []
    objects_in_relative_direction = {}
    in_direction_dict = {}

    for candidate_uuid in candidate_uuids:
        if candidate_uuid == track_uuid:
            continue

        pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        if len(timestamps) == 0:
            continue

        # Vectorized direction filter: same boolean as the original 4-way OR
        # condition, evaluated for every timestamp at once.
        pos_arr = np.asarray(pos)
        if direction == 'left':
            in_dir = (pos_arr[:,1] >  track_width) & (track_back-lateral_thresh < pos_arr[:,0]) & (pos_arr[:,0] < track_front+lateral_thresh)
        elif direction == 'right':
            in_dir = (pos_arr[:,1] < -track_width) & (track_back-lateral_thresh < pos_arr[:,0]) & (pos_arr[:,0] < track_front+lateral_thresh)
        elif direction == 'forward':
            in_dir = (pos_arr[:,0] >  track_front) & (-track_width-lateral_thresh < pos_arr[:,1]) & (pos_arr[:,1] < track_width+lateral_thresh)
        elif direction == 'backward':
            in_dir = (pos_arr[:,0] <  track_back)  & (-track_width-lateral_thresh < pos_arr[:,1]) & (pos_arr[:,1] < track_width+lateral_thresh)
        else:
            in_dir = np.zeros(len(timestamps), dtype=bool)

        hit_indices = np.where(in_dir)[0]
        if len(hit_indices) == 0:
            continue

        # Batch all distances in a single helper call (one DataFrame filter
        # per uuid instead of one per timestamp).
        hit_ts = [timestamps[i] for i in hit_indices]
        distances = cuboid_distance_batch(track_uuid, candidate_uuid, log_dir, hit_ts)

        for k, ts in enumerate(hit_ts):
            if not in_direction_dict.get(ts, None):
                in_direction_dict[ts] = []
            in_direction_dict[ts].append((candidate_uuid, distances[k]))

    for timestamp, objects in in_direction_dict.items():
        sorted_objects = sorted(objects, key=lambda row: row[1])

        count = 0
        true_uuids = []
        for candidate_uuid, distance in sorted_objects:
            if distance <= within_distance and count < max_number:
                count += 1
                true_uuids.append(candidate_uuid)

        if count >= min_number:
            for true_uuid in true_uuids:
                if true_uuid not in objects_in_relative_direction:
                    objects_in_relative_direction[true_uuid] = []
                objects_in_relative_direction[true_uuid].append(timestamp)
                timestamps_with_objects.append(timestamp)

    return timestamps_with_objects, objects_in_relative_direction


@cache_manager.create_cache('get_objects_in_relative_direction')
def get_objects_in_relative_direction(
    track_candidates:dict,
    related_candidates:dict, 
    log_dir:Path, 
    direction:Literal["forward", "backward", "left", "right"], 
    min_number:int=0, 
    max_number:int=np.inf, 
    within_distance:float=50, 
    lateral_thresh:float=np.inf)->dict:
    """
    Returns a scenario dictionary of the related candidates that are in the relative direction of the track candidates.
    

    Args:
        track_candidates: Tracks  (scenario dictionary).
        related_candidates: Candidates to check for in direction (scenario dictionary).
        log_dir: Path to scenario logs.
        direction: Direction to analyze from the track's point of view ('forward', 'backward', 'left', 'right').
        min_number: Minimum number of objects to identify in the direction per timestamp. Defaults to 0.
        max_number: Maximum number of objects to identify in the direction per timestamp. Defaults to infinity.
        within_distance: Maximum distance for considering an object in the direction. Defaults to infinity.
        lateral_thresh: Maximum lateral distance the related object can be from the sides of the tracked object. Lateral distance is 
        distance is the distance from the sides of the object that are parallel to the specified direction. Defaults to infinity.

    Returns:
        dict: 
            A scenario dictionary where keys are track UUIDs and values are dictionaries containing related candidate UUIDs 
            and lists of timestamps when the condition is met for that relative direction.

    Example:
        peds_in_front_of_vehicles = get_objects_in_relative_direction(vehicles, pedestrians, log_dir, direction='forward', min_number=2)
    """
    
    tracked_objects = \
    reverse_relationship(has_objects_in_relative_direction)(track_candidates, related_candidates, log_dir, direction,
        min_number=min_number, max_number=max_number, within_distance=within_distance, lateral_thresh=lateral_thresh)

    return tracked_objects


def get_objects_of_category(log_dir, category)->dict:
    """
    Returns all objects from a given category from the log annotations. This method accepts the 
    super-categories "ANY" and "VEHICLE".

    Args:
        log_dir: Path to the directory containing scenario logs and data.
        category: the category of objects to return

    Returns: 
        dict: A scenario dict that where keys are the unique id (uuid) of the object and values 
        are the list of timestamps the object is in view of the ego-vehicle.

    Example:
        trucks = get_objects_of_category(log_dir, category='TRUCK')
    """
    return to_scenario_dict(get_uuids_of_category(log_dir, category), log_dir)


@composable
def is_category(track_candidates:dict, log_dir:Path, category:str):
    """
    Returns all objects from a given category from track_candidates dict. This method accepts the 
    super-categories "ANY" and "VEHICLE".

    Args:
        track_candidates: The scenario dict containing the objects to filter down
        log_dir: Path to the directory containing scenario logs and data.
        category: the category of objects to return

    Returns: 
        dict: A scenario dict that where keys are the unique id of the object of the given category and values 
        are the list of timestamps the object is in view of the ego-vehicle.

    Example:
        box_trucks = is_category(vehicles, log_dir, category='BOX_TRUCK')
    """


    track_uuid = track_candidates
    if track_uuid in get_uuids_of_category(log_dir, category):
        non_composable_get_object = unwrap_func(get_object)
        return non_composable_get_object(track_uuid, log_dir)
    else:
        return []


@cache_manager.create_cache('_is_weather_log_majority')
def _is_weather_log_majority(log_dir: Path, condition: str) -> bool:
    """Log-level majority vote for weather condition. Independent of track_uuid,
    so factor it out of `is_weather`'s body to avoid re-reading JSON per-track."""
    key = "clear_day" if condition == "clear" else condition
    context_dir = paths.CONTEXT_ANNOTATIONS_DIR / f"{get_log_split(log_dir)}_processed" / Path(log_dir).name
    if not context_dir.exists():
        return False
    trues = total = 0
    for json_file in context_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        front = data.get("per_camera", data.get("cameras", {})).get("ring_front_center", {})
        if front.get("weather", {}).get(key, False):
            trues += 1
        total += 1
    return trues * 2 > total


@composable
@cache_manager.create_cache('is_weather')
def is_weather(
    track_candidates: dict,
    log_dir: Path,
    condition: Literal["rain", "snow", "clear", "cloudy"],
) -> dict:
    """
    Returns objects from the log when the scene's weather matches the given condition,
    as labeled by the Scene Context VLM on the front camera. Weather is evaluated at
    log level: the whole log either matches or does not, determined by a majority vote
    across per-timestamp front-camera annotations.

    Args:
        track_candidates: The objects you want to filter from (scenario dictionary).
        log_dir: Path to scenario logs.
        condition: Weather condition. Must be one of 'rain', 'snow', 'clear', 'cloudy'.

    Returns:
        dict:
            A filtered scenario dictionary where:
            - Keys are track UUIDs when the log matches the condition.
            - Values are nested dictionaries containing timestamps.

    Example:
        cars_in_rain = is_weather(cars, log_dir, condition='rain')
    """
    track_uuid = track_candidates

    if condition not in ("rain", "snow", "clear", "cloudy"):
        print(f"Specified weather condition must be one of "
              f"['rain', 'snow', 'clear', 'cloudy']. Got '{condition}'. Returning empty.")
        return []

    # Majority vote depends only on (log_dir, condition) -> cached separately to
    # avoid re-reading every JSON per track_uuid.
    if _is_weather_log_majority(log_dir, condition):
        return get_timestamps(track_uuid, log_dir)
    return []


@cache_manager.create_cache('_is_time_of_day_log_majority')
def _is_time_of_day_log_majority(log_dir: Path, period: str) -> bool:
    """Log-level majority vote for time of day. Independent of track_uuid."""
    key = "dusk_dawn" if period == "dusk_or_dawn" else period
    context_dir = paths.CONTEXT_ANNOTATIONS_DIR / f"{get_log_split(log_dir)}_processed" / Path(log_dir).name
    if not context_dir.exists():
        return False
    trues = total = 0
    for json_file in context_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        front = data.get("per_camera", data.get("cameras", {})).get("ring_front_center", {})
        if front.get("time_of_day", {}).get(key, False):
            trues += 1
        total += 1
    return trues * 2 > total


@composable
@cache_manager.create_cache('is_time_of_day')
def is_time_of_day(
    track_candidates: dict,
    log_dir: Path,
    period: Literal["daylight", "dusk_or_dawn"],
) -> dict:
    """
    Returns objects from the log when the scene's time of day matches the given period,
    as labeled by the Scene Context VLM on the front camera. Time of day is evaluated
    at log level: the whole log either matches or does not, determined by a majority
    vote across per-timestamp front-camera annotations.

    Args:
        track_candidates: The objects you want to filter from (scenario dictionary).
        log_dir: Path to scenario logs.
        period: Time-of-day period. Must be one of 'daylight' or 'dusk_or_dawn'.

    Returns:
        dict:
            A filtered scenario dictionary where:
            - Keys are track UUIDs when the log matches the period.
            - Values are nested dictionaries containing timestamps.

    Example:
        peds_at_dusk = is_time_of_day(pedestrians, log_dir, period='dusk_or_dawn')
    """
    track_uuid = track_candidates

    if period not in ("daylight", "dusk_or_dawn"):
        print(f"Specified time of day must be one of "
              f"['daylight', 'dusk_or_dawn']. Got '{period}'. Returning empty.")
        return []

    if _is_time_of_day_log_majority(log_dir, period):
        return get_timestamps(track_uuid, log_dir)
    return []


@composable
@cache_manager.create_cache('near_infrastructure')
def near_infrastructure(
    track_candidates: dict,
    log_dir: Path,
    infrastructure: Literal[
        # per-camera labels (visibility in a ring camera)
        "bus_stop", "parking_lot", "railroad_tracks", "fence",
        "red_painted_lane", "construction_zone", "gas_station",
        # ego labels (ego vehicle is on/inside)
        "bridge", "brick_street", "pothole", "filled_pothole", "storm_grate",
        "road_damage", "streetcar_tracks", "roundabout", "school_zone",
        "speed_limit_zone", "shadow_of_building",
        "green_light", "yellow_light", "broken_traffic_light",
    ],
) -> dict:
    """Filter tracks to timestamps where a Scene Context VLM infrastructure
    label holds. The category of ``infrastructure`` (not the track type) drives
    the matching path. Labels are either per-camera (VLM flag in a ring camera)
    or ego-only (scene-level: ego vehicle is on/inside the item); see the
    Literal type for exact members.

    Pass conditions:
                    | per-camera label              | ego label
        ------------+-------------------------------+-----------------------------
        non-ego     | a ring cam flags label AND    | scene-level: any ts at which
        track       | the track's cuboid projects   | ego.<label> is true passes
                    | into that same camera         | (the track's own position
                    |                               |  is NOT checked)
        ------------+-------------------------------+-----------------------------
        ego track   | >= 2 ring cameras flag label  | ego.<label> is true at ts

    Dual label ('construction_zone' only): non-ego uses the per-camera path
    only; ego uses per-camera OR ego flag.

    Note on non-ego + ego label (scene-level): this is the right path when the
    natural-language intent is "object co-occurring while ego is on/inside the
    infrastructure" (e.g., the prompt "person directing traffic in a school
    zone", where the person — not the ego — is the referred track).
    If the intent is instead "the object itself is on/inside the infrastructure",
    combine with a positional atomic — ego labels carry no per-object signal.

    Args:
        track_candidates: Tracks to filter (scenario dictionary).
        log_dir: Path to scenario logs.
        infrastructure: Scene Context label (see Literal for members). An
            unrecognized label returns an empty result.

    Returns:
        dict: A filtered scenario dictionary where keys are track UUIDs and
        values are lists of timestamps when the condition holds.

    Example:
        # ego label on ego track
        ego_on_bridge = near_infrastructure(ego_vehicle, log_dir, infrastructure='bridge')
        # per-camera label on non-ego track (positional match)
        peds_at_bus_stop = near_infrastructure(pedestrians, log_dir, infrastructure='bus_stop')
        # ego label on non-ego track (scene-level: pedestrians co-occurring while ego is in a school zone)
        peds_while_ego_in_school_zone = near_infrastructure(pedestrians, log_dir, infrastructure='school_zone')
        # dual label
        vehicles_near_construction = near_infrastructure(vehicles, log_dir, infrastructure='construction_zone')
    """
    track_uuid = track_candidates

    in_pc = infrastructure in _PER_CAMERA_INFRA
    in_ego = infrastructure in _EGO_INFRA

    per_cam = get_context_annotations(log_dir) if in_pc else {}
    ego_ann = get_ego_annotations(log_dir) if in_ego else {}
    if not per_cam and not ego_ann:
        return []

    is_ego = (track_uuid == get_ego_uuid(log_dir))
    need_crops = (not is_ego) and in_pc
    img_crops = get_img_crops(track_uuid, log_dir) if need_crops else None

    matched = []
    for ts in get_timestamps(track_uuid, log_dir):
        ts_i = int(ts)

        ego_hit = bool(ego_ann.get(ts_i, {}).get(infrastructure)) if in_ego else False

        pc_hit = False
        if in_pc:
            cam_block = per_cam.get(ts_i, {}) or {}
            positive_cams = [
                cam for cam, blk in cam_block.items()
                if blk.get("infra", {}).get(infrastructure, False)
            ]
            if positive_cams:
                if is_ego:
                    pc_hit = len(positive_cams) >= 2
                else:
                    pc_hit = any(
                        img_crops.get(cam, {}).get(ts) is not None
                        for cam in positive_cams
                    )

        if is_ego:
            if pc_hit or ego_hit:
                matched.append(ts)
        else:
            if in_pc and not in_ego:
                if pc_hit:
                    matched.append(ts)
            elif in_ego and not in_pc:
                if ego_hit:
                    matched.append(ts)
            else:  # dual label (construction_zone), non-ego: per-camera path only
                if pc_hit:
                    matched.append(ts)

    return matched


@composable
@cache_manager.create_cache('within_camera_view')
def within_camera_view(
    track_candidates: dict,
    log_dir: Path,
    camera_name:str
) -> dict:
    """
    Returns objects that are within view of the specified camera.

    Args:
        track_candidates: The objects you want to filter from (scenario dictionary).
        log_dir: Path to scenario logs.
        camera_name: The name of the camera.

    Returns:
        dict:
            A filtered scenario dictionary where:
            - Keys are track UUIDs that are within view of the specified camera.
            - Values are nested dictionaries containing timestamps.

    Example:
        front_objects = within_camera_view(vehicles, log_dir, camera_name='ring_front_center')
    """
    track_uuid = track_candidates

    all_views = get_img_crops(track_uuid, log_dir)
    camera_views = all_views[camera_name]
    within_view_timestamps = [timestamp for (timestamp, box) in camera_views.items() if box is not None]

    return within_view_timestamps


@composable
@cache_manager.create_cache('turning')
def turning(
    track_candidates: dict, 
    log_dir:Path,
    direction:Literal["left", "right", None]=None)->dict:
    """
    Returns objects that are turning in the given direction. 

    Args:
        track_candidates: The objects you want to filter from (scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the turn, from the track's point of view ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            - Keys are track UUIDs that meet the turning criteria.
            - Values are nested dictionaries containing timestamps.

    Example:
        turning_left = turning(vehicles, log_dir, direction='left')
    """
    track_uuid = track_candidates

    if direction and direction != 'left' and direction != 'right':
        direction = None
        print("Specified direction must be 'left', 'right', or None. Direction set to \
              None automatically.")
    
    TURN_ANGLE_THRESH = 45 #degrees 
    ANG_VEL_THRESH = 5 #deg/s

    ang_vel, timestamps = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self', in_degrees=True)

    turn_dict = {'left': [], 'right':[]}

    start_index = 0
    end_index = start_index
    
    while start_index < len(timestamps)-1:
        #Check if the object is continuing to turn in the same direction
        if ((ang_vel[start_index] > 0 and ang_vel[end_index] > 0 
        or ang_vel[start_index] < 0 and ang_vel[end_index] < 0) 
        and end_index < len(timestamps)-1):
            end_index += 1
        else:
            #Check if the object's angle has changed enough to define a turn
            s_per_timestamp = float(timestamps[1] - timestamps[0])/1E9
            if np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) > TURN_ANGLE_THRESH:
                turn_dict['left'].extend(timestamps[start_index:end_index+1])
            elif np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) < -TURN_ANGLE_THRESH:
                turn_dict['right'].extend(timestamps[start_index:end_index+1])
            #elif (unwrap_func(near_intersection)(track_uuid, log_dir) 
            #and (start_index == 0 and unwrap_func(near_intersection)(track_uuid, log_dir)[0] == timestamps[0]
            #    or end_index == len(timestamps)-1 and unwrap_func(near_intersection)(track_uuid, log_dir)[-1] == timestamps[-1])):

                if (((start_index==0 and ang_vel[start_index] > ANG_VEL_THRESH) 
                    or (end_index==len(timestamps)-1 and ang_vel[end_index] > ANG_VEL_THRESH))
                and np.mean(ang_vel[start_index:end_index+1]) > ANG_VEL_THRESH
                and np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) > TURN_ANGLE_THRESH/3):
                    turn_dict['left'].extend(timestamps[start_index:end_index+1])
                elif (((start_index==0 and ang_vel[start_index] < -ANG_VEL_THRESH) 
                    or (end_index==len(timestamps)-1 and ang_vel[end_index] < -ANG_VEL_THRESH))
                and np.mean(ang_vel[start_index:end_index+1]) < -ANG_VEL_THRESH
                and np.sum(ang_vel[start_index:end_index+1]*s_per_timestamp) < -TURN_ANGLE_THRESH/3):
                    turn_dict['right'].extend(timestamps[start_index:end_index+1])

            start_index = end_index
            end_index += 1 
    
    if direction:
        return turn_dict[direction]
    else:
        return turn_dict['left'] + turn_dict['right']  


@composable
@cache_manager.create_cache('changing_lanes')
def changing_lanes(
    track_candidates:dict, 
    log_dir:Path,
    direction:Literal["left", "right", None]=None) -> dict:
    """
    Identifies lane change events for tracked objects in a scenario.

    Args:
        track_candidates: The tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the lane change. None indicates tracking either left or right lane changes ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            Keys are track UUIDs that meet the lane change criteria.
            Values are nested dictionaries containing timestamps and related data.

    Example:
        left_lane_changes = changing_lanes(vehicles, log_dir, direction='left')
    """
    track_uuid = track_candidates

    if direction is not None and direction != 'right' and direction != 'left':
        print("Direction must be 'right', 'left', or None.")
        print("Setting direction to None.")
        direction = None

    COS_SIMILARITY_THRESH = .5 #vehicle must be headed in a direction at most 45 degrees from the direction of the lane boundary
    SIDEWAYS_VEL_THRESH = .1 #m/s

    lane_traj = get_scenario_lanes(track_uuid, log_dir)
    positions, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    velocities, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    #Each index stored in dict indicates the exact timestep where the track crossed lanes
    lane_changes_exact = {'left': [], 'right':[]}
    for i in range(1, len(timestamps)):
        prev_lane = lane_traj.get(timestamps[i-1])
        cur_lane = lane_traj.get(timestamps[i])

        if prev_lane and cur_lane and abs(velocities[i,1]) >= SIDEWAYS_VEL_THRESH:
            if prev_lane.right_neighbor_id == cur_lane.id:

                #caclulate lane orientation
                closest_waypoint_idx = np.argmin(np.linalg.norm(prev_lane.right_lane_boundary.xyz[:,:2]-positions[i,:2], axis=1))
                start_idx = max(0, closest_waypoint_idx-1)
                end_idx = min(len(prev_lane.right_lane_boundary.xyz)-1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.right_lane_boundary.xyz[end_idx,:2] - prev_lane.right_lane_boundary.xyz[start_idx,:2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i,:2] / np.linalg.norm(velocities[i,:2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['right'].append(i)
            elif prev_lane.left_neighbor_id == cur_lane.id:
                #caclulate lane orientation
                closest_waypoint_idx = np.argmin(np.linalg.norm(prev_lane.left_lane_boundary.xyz[:,:2]-positions[i,:2], axis=1))

                # [FIX] min(0, ...) -> max(0, ...). Negative index made left lane boundary direction wrong (mirrors right-lane logic above).
                start_idx = max(0, closest_waypoint_idx - 1)
                end_idx = min(len(prev_lane.left_lane_boundary.xyz)-1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.left_lane_boundary.xyz[end_idx,:2] - prev_lane.left_lane_boundary.xyz[start_idx,:2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i,:2] / np.linalg.norm(velocities[i,:2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['left'].append(i)

    lane_changes = {'left': [], 'right':[]}

    for index in lane_changes_exact['left']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start], lane_traj.get(timestamps[lane_change_start]))
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start+1], lane_traj.get(timestamps[lane_change_start+1]))

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) or lane_change_start == index-1:
                lane_changes['left'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break
            
        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end-1], lane_traj.get(timestamps[lane_change_end-1]))
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end], lane_traj.get(timestamps[lane_change_end]))

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) or lane_change_end == index:
                lane_changes['left'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break
    
    for index in lane_changes_exact['right']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start], lane_traj.get(timestamps[lane_change_start]))
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start+1], lane_traj.get(timestamps[lane_change_start+1]))

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 or lane_change_start == index-1:
                lane_changes['right'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break
            
        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end-1], lane_traj.get(timestamps[lane_change_end-1]))
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end], lane_traj.get(timestamps[lane_change_end]))

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 or lane_change_end == index:
                lane_changes['right'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    if direction:
        lane_changing_timestamps = lane_changes[direction]
    else:
        lane_changing_timestamps = sorted(list(set(lane_changes['left'] + (lane_changes['right']))))

    turning_timestamps = unwrap_func(turning)(track_uuid, log_dir)
    return sorted(list(set(lane_changing_timestamps).difference(set(turning_timestamps))))


@composable
@cache_manager.create_cache('has_lateral_acceleration')
def has_lateral_acceleration(
    track_candidates:dict,
    log_dir:Path,
    min_accel=-np.inf,
    max_accel=np.inf) -> dict:
    """
    Objects with a lateral acceleartion between the minimum and maximum thresholds. 
    Most objects with a high lateral acceleration are turning. Postive values indicate accelaration
    to the left while negative values indicate acceleration to the right. 

    Args:
        track_candidates: The tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        direction: The direction of the lane change. None indicates tracking either left or right lane changes ('left', 'right', None).

    Returns:
        dict: 
            A filtered scenario dictionary where:
            Keys are track UUIDs that meet the lane change criteria.
            Values are nested dictionaries containing timestamps and related data.

    Example:
        jerking_left = has_lateral_acceleration(non_turning_vehicles, log_dir, min_accel=2)
    """
    track_uuid = track_candidates

    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    # Vectorized: same mask as the per-row `min_accel <= accel[1] <= max_accel` (y-axis lateral).
    accelerations = np.asarray(accelerations)
    if len(accelerations):
        mask = (min_accel <= accelerations[:, 1]) & (accelerations[:, 1] <= max_accel)  # m/s^2
        hla_timestamps = [timestamps[i] for i in np.where(mask)[0]]
    else:
        hla_timestamps = []

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return hla_timestamps

@composable_relational
@cache_manager.create_cache('cut_in_front_of')
def cut_in_front_of(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path) -> dict:
    """
    Identifies cut-in events where a track candidate enters the forward lateral band
    of a related candidate from the side.

    A cut-in is a track candidate A moving laterally into the short forward
    lateral band of a related candidate B (in B's frame), while A and B travel in
    roughly the same direction and B is moving. It covers standard lane-change
    cut-ins, brief turn-off cut-ins, and aggressive swerves, and excludes
    vehicles already ahead, longitudinal same-lane approaches, U-turns,
    oncoming traffic, B changing lanes behind A, and a stationary B. All
    thresholds are internal constants.

    Args:
        track_candidates: Tracks that potentially cut in (scenario dictionary).
        related_candidates: Tracks that are potentially cut in front of (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        A filtered scenario dictionary of track_candidates that cut in front of
        at least one related candidate, with matched timestamps per pair.

    Example:
        vehicle_cut_in_to_ego = cut_in_front_of(vehicles, ego, log_dir)
    """
    # === Lateral band geometry ===
    WITHIN_DISTANCE_M = 30.0          # forward longitudinal limit ahead of B (m)
    INNER_MARGIN_M = 1.0                # inner lateral half-width = B.width/2 + this (m)
    OUTER_MARGIN_M = 1.0              # extra lateral gap to the "was outside" line (m)
    # === Entry transition ===
    TRANSITION_WINDOW_S = 2.0       # lookback time for entry transition (s)
    HEADING_COS_THRESH = 0.5        # same-direction threshold (~60 deg)
    MIN_RELATED_SPEED_MPS = 0.5         # min speed of B (m/s)
    # === Output window (asymmetric) ===
    PRE_ENTRY_WINDOW_S = 2.0        # fixed hold before entry (s); covers the lateral approach
    POST_ENTRY_MAX_S = 4.0          # fixed hold after entry (s); held purely by time so cut-then-turn-out tails are kept

    track_uuid = track_candidates               # A: track candidate

    cut_in_timestamps = []
    cutting_pairs = {}

    for candidate in related_candidates:        # B: related candidate
        if candidate == track_uuid:
            continue

        b_cuboid = get_cuboid_from_uuid(candidate, log_dir)
        lat_inner = b_cuboid.width_m / 2 + INNER_MARGIN_M
        lat_outer = lat_inner + OUTER_MARGIN_M

        # Step 1: get time series of A and B
        pos_A_in_B, ts = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=candidate)
        yaw_A_in_B, _ = get_nth_yaw_deriv(track_uuid, 0, log_dir, coordinate_frame=candidate)
        vel_B, _ = get_nth_pos_deriv(candidate, 1, log_dir, coordinate_frame=track_uuid)

        pos_A_in_B = np.asarray(pos_A_in_B)
        yaw_A_in_B = np.asarray(yaw_A_in_B)
        speed_B = np.linalg.norm(np.asarray(vel_B)[:, :2], axis=1)

        T = [int(t) for t in ts]                       # time axis
        if len(T) < 2 or len(yaw_A_in_B) != len(T) or len(speed_B) != len(T):
            continue                                   
        
        X = pos_A_in_B[:, 0]
        Y = pos_A_in_B[:, 1]
        VB = speed_B
        COS = np.cos(yaw_A_in_B)
        
        dt_ns = T[1] - T[0]                            
        if dt_ns <= 0:
            continue                                   # guard divide-by-zero / non-increasing timestamps
        delta = max(1, int(round(TRANSITION_WINDOW_S * 1e9 / dt_ns))) 
        if len(T) < delta + 1:
            continue                                   # track too short to look back delta frames

        # Step 2: cut in event check: A enters B's forward lateral band from the side
        entry_indices = []
        for i in range(delta, len(T)):
            lon_now    = X[i]                  
            lat_now    = abs(Y[i])             

            b_is_in_front_zone = (0 < lon_now < WITHIN_DISTANCE_M) and (lat_now < lat_inner)

            b_is_swept_in = any(abs(Y[j]) > lat_outer for j in range(i - delta, i))

            b_is_heading_aligned = all(COS[j] > HEADING_COS_THRESH for j in range(i - delta, i + 1))

            b_is_moving = VB[i] > MIN_RELATED_SPEED_MPS

            if b_is_in_front_zone and b_is_swept_in and b_is_heading_aligned and b_is_moving:
                entry_indices.append(i)

        if not entry_indices:
            continue

        # Steps 3-4: once an entry triggers, hold a fixed time window around it
        matched = set()
        for i_entry in entry_indices:
            t_entry = T[i_entry]
            t_start = t_entry - PRE_ENTRY_WINDOW_S * 1e9
            t_end = t_entry + POST_ENTRY_MAX_S * 1e9

            for t in T:
                if t_start <= t <= t_end:
                    matched.add(t)

        # Step 5: accumulate.
        if matched:
            cutting_pairs[candidate] = sorted(matched)
            cut_in_timestamps.extend(matched)

    return sorted(set(cut_in_timestamps)), cutting_pairs

@composable_relational
@cache_manager.create_cache('facing_toward')
def facing_toward(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path,
    within_angle:float=22.5,
    max_distance:float=50)->dict:
    """
    Identifies objects in track_candidates that are FACING (orientation-based)
    toward objects in related candidates. Uses the track's forward axis
    (heading), not its velocity vector — so it works for stationary subjects
    too. Compare with heading_toward (velocity-based).

    The related candidate must lie within ``within_angle`` degrees on either
    side of the track-candidate's forward axis.

    Args:
        track_candidates: Tracks to test (the subjects that may be facing).
        related_candidates: Targets the subject may be facing toward.
        log_dir: Path to scenario logs.
        within_angle: Half-angle (degrees) of the cone around the forward
            axis. A related candidate is "faced toward" if it lies within
            within_angle on either side. Default 22.5.
        max_distance: Max distance (m) the related candidate can be away.
            Default 50.

    Returns:
        Filtered scenario dict containing the subset of track candidates
        facing toward at least one of the related candidates.

    Example:
        pedestrian_facing_away = scenario_not(facing_toward)(pedestrian, ego_vehicle, log_dir, within_angle=180)
    """

    track_uuid = track_candidates
    facing_toward_timestamps = []
    facing_toward_objects = {}

    for candidate_uuid in related_candidates:

        if candidate_uuid == track_uuid:
            continue

        traj, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        if len(timestamps) == 0:
            continue

        # Vectorized angle filter (same as the per-row abs(angle) <= within_angle check).
        traj = np.asarray(traj)
        angles = np.rad2deg(np.arctan2(traj[:, 1], traj[:, 0]))
        angle_mask = np.abs(angles) <= within_angle
        ang_hit = np.where(angle_mask)[0]
        if len(ang_hit) == 0:
            continue

        # Batch the distance check only for angle-passing timestamps.
        hit_ts = [timestamps[i] for i in ang_hit]
        distances = cuboid_distance_batch(track_uuid, candidate_uuid, log_dir, hit_ts)
        dist_mask = distances <= max_distance
        for k, ts in enumerate(hit_ts):
            if dist_mask[k]:
                facing_toward_timestamps.append(ts)
                if candidate_uuid not in facing_toward_objects:
                    facing_toward_objects[candidate_uuid] = []
                facing_toward_objects[candidate_uuid].append(ts)

    return facing_toward_timestamps, facing_toward_objects


@composable_relational
@cache_manager.create_cache('heading_toward')
def heading_toward(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path,
    angle_threshold:float=22.5,
    minimum_speed:float=.5,
    max_distance:float=np.inf)->dict:
    """
    Identifies objects in track_candidates whose VELOCITY VECTOR points at
    objects in related_candidates (motion-based). The track candidate's
    velocity vector must be within angle_threshold of the relative-position
    vector to the related candidate, AND the velocity component toward the
    related candidate must exceed minimum_speed.

    NOT the same as "approaching":
      * heading_toward requires the SUBJECT (track) has measurable velocity
        in the target's direction. For most "X approaching Y" prompts where
        Y is static / stopped / at a location (peds at crossing, parked X,
        stopped truck), use has_objects_in_relative_direction(X, Y,
        direction='forward', ...) instead — it does not require subject motion.
      * Use heading_toward only when both sides clearly have motion AND the
        prompt is explicitly motion-based ("driving toward", "heading to").

    Args:
        track_candidates: Tracks that could be heading toward another track.
        related_candidates: Target objects to test.
        log_dir: Path to the directory containing scenario logs and data.
        angle_threshold: Maximum angle (deg) between the velocity vector and
            the relative-position vector. Default 22.5.
        minimum_speed: Minimum magnitude of the velocity component toward the
            related candidate. Default 0.5 m/s.
        max_distance: Maximum distance (m) the related candidate can be away
            from the track to count. Default infinity.

    Returns:
        Filtered scenario dict: subset of track candidates heading toward at
        least one related candidate.

    Example:
        # Motion-based: vehicle driving toward another moving vehicle.
        heading_toward_traffic_cone = heading_toward(vehicles, traffic_cone, log_dir)
    """

    track_uuid = track_candidates
    heading_toward_timestamps = []
    heading_toward_objects = {}

    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)
    # O(n) list.index -> O(1) dict lookup, also avoids repeated `timestamp in track_timestamps` scans.
    track_ts_to_idx = {int(t): i for i, t in enumerate(track_timestamps)}
    track_vel = np.asarray(track_vel)

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        related_pos, related_timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        track_radial_vel, _ = get_nth_radial_deriv(
            track_uuid, 1, log_dir, coordinate_frame=candidate_uuid)
        if len(related_timestamps) == 0:
            continue
        related_pos = np.asarray(related_pos)
        track_radial_vel = np.asarray(track_radial_vel)

        # Vectorized intersection: indices into track_vel for each related timestamp
        # (or -1 if absent). Same dtype as the legacy `track_timestamps.index(timestamp)` lookup,
        # just precomputed once per candidate.
        track_idx = np.array([track_ts_to_idx.get(int(t), -1) for t in related_timestamps])
        has_track = track_idx >= 0
        if not np.any(has_track):
            continue

        # Compute angle and the radial-speed gate for all (intersecting) related timestamps at once.
        # Replaces the per-ts arccos+dot block. Mirror the original normalisation forms exactly:
        #   vel_direction         = timestamp_vel / (np.linalg.norm(timestamp_vel) + 1e-8)
        #   direction_of_related  = related_pos[i] / np.linalg.norm(related_pos[i] + 1e-8)
        safe_track_vel = track_vel[np.where(has_track, track_idx, 0)]   # shape (N, 3); rows ignored when ~has_track
        vel_norms = np.linalg.norm(safe_track_vel, axis=1) + 1e-8
        vel_dir = safe_track_vel / vel_norms[:, None]
        rel_norms = np.linalg.norm(related_pos + 1e-8, axis=1)
        rel_dir = related_pos / rel_norms[:, None]
        cos_per_ts = np.sum(vel_dir * rel_dir, axis=1)
        cos_per_ts = np.clip(cos_per_ts, -1.0, 1.0)  # guard arccos against tiny float overshoot
        angles = np.rad2deg(np.arccos(cos_per_ts))

        gate = has_track & (-track_radial_vel >= minimum_speed) & (angles <= angle_threshold)
        gate_idx = np.where(gate)[0]
        if len(gate_idx) == 0:
            continue

        # Batch cuboid_distance only for gate-passing timestamps.
        hit_ts = [related_timestamps[i] for i in gate_idx]
        distances = cuboid_distance_batch(track_uuid, candidate_uuid, log_dir, hit_ts)
        dist_mask = distances <= max_distance
        for k, ts in enumerate(hit_ts):
            if dist_mask[k]:
                heading_toward_timestamps.append(ts)
                if candidate_uuid not in heading_toward_objects:
                    heading_toward_objects[candidate_uuid] = []
                heading_toward_objects[candidate_uuid].append(ts)

    return heading_toward_timestamps, heading_toward_objects


@composable
@cache_manager.create_cache('accelerating')
def accelerating(
    track_candidates:dict,
    log_dir:Path,
    min_accel:float=.65,
    max_accel:float=np.inf)->dict:
    """
    Identifies objects in track_candidates that have a forward acceleration above a threshold.
    Values under -1 reliably indicates braking. Values over 1.0 reliably indiciates accelerating.

    Args:
        track_candidates: The tracks to analyze for acceleration (scenario dictionary)
        log_dir:  Path to the directory containing scenario logs and data.
        min_accel: The lower bound of acceleration considered
        max_accel: The upper bound of acceleration considered

    Returns:
        A filtered scenario dictionary containing the objects with an acceleration between the lower and upper bounds.

    Example:
        accelerating_motorcycles = accelerating(motorcycles, log_dir)

    """
    track_uuid = track_candidates

    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    # Vectorized: same mask as the per-row `min_accel <= accel[0] <= max_accel` check.
    accelerations = np.asarray(accelerations)
    if len(accelerations):
        mask = (min_accel <= accelerations[:, 0]) & (accelerations[:, 0] <= max_accel)  # m/s^2
        acc_timestamps = [timestamps[i] for i in np.where(mask)[0]]
    else:
        acc_timestamps = []

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return acc_timestamps


@composable
@cache_manager.create_cache('has_velocity')
def has_velocity(
    track_candidates:dict,
    log_dir:Path, 
    min_velocity:float=.5, 
    max_velocity:float=np.inf)->dict:
    """
    Identifies objects with a velocity between the given maximum and minimum velocities in m/s.
    Stationary objects may have a velocity up to 0.5 m/s due to annotation jitter.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        min_velocity: Minimum velocity (m/s). Defaults to 0.5.
        max_velocity: Maximum velocity (m/s)

    Returns:
        Filtered scenario dictionary of objects meeting the velocity criteria.

    Example:
        fast_vehicles = has_min_velocity(vehicles, log_dir, min_velocity=5)
    """
    track_uuid = track_candidates

    vels, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    # Vectorized: |vel| per timestamp at once, same threshold mask as the per-row form.
    vels = np.asarray(vels)
    if len(vels):
        speeds = np.linalg.norm(vels, axis=1)  # m/s
        mask = (min_velocity <= speeds) & (speeds <= max_velocity)
        vel_timestamps = [timestamps[i] for i in np.where(mask)[0]]
    else:
        vel_timestamps = []

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return vel_timestamps


@composable
@cache_manager.create_cache('at_pedestrian_crossing')
def at_pedestrian_crossing(
    track_candidates:dict,
    log_dir:Path,
    within_distance:float=1)->dict:
    """
    Identifies objects that within a certain distance from a pedestrian crossing. A distance of zero indicates
    that the object is within the boundaries of the pedestrian crossing.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        within_distance: Distance in meters the track candidate must be from the pedestrian crossing. A distance of zero
            means that the object must be within the boundaries of the pedestrian crossing.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        vehicles_at_ped_crossing = at_pedestrian_crossing(vehicles, log_dir)
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    all_pcs = avm.get_scenario_ped_crossings()

    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)
    verts_per_ts = get_cuboid_vertices_batch(track_uuid, log_dir, timestamps)  # (T, 8, 3)

    # Dilate each pedestrian-crossing polygon once (constant across timestamps).
    dilated_pcs = [dilate_convex_polygon(pc.polygon[:, :2], distance=within_distance)
                   for pc in all_pcs]

    timestamps_at_object = []
    for t, timestamp in enumerate(timestamps):
        city_vertices = ego_poses[timestamp].transform_from(verts_per_ts[t])
        track_poly = np.array([city_vertices[2], city_vertices[6], city_vertices[7],
                               city_vertices[3], city_vertices[2]])[:, :2]

        for pc_poly in dilated_pcs:
            if polygons_overlap(track_poly, pc_poly):
                timestamps_at_object.append(timestamp)
                break   # avoid double-counting the same ts across crossings

    return timestamps_at_object


@composable
@cache_manager.create_cache('on_lane_type')
def on_lane_type(
    track_uuid:dict,
    log_dir,
    lane_type:Literal["BUS", "VEHICLE", "BIKE"])->dict:
    """
    Identifies objects on a specific lane type.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        lane_type: Type of lane to check ('BUS', 'VEHICLE', or 'BIKE').

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        vehicles_on_bus_lane = on_lane_type(vehicles, log_dir, lane_type="BUS")
    """

    scenario_lanes = get_scenario_lanes(track_uuid, log_dir)
    timestamps = scenario_lanes.keys()

    return [timestamp for timestamp in timestamps if scenario_lanes[timestamp] and scenario_lanes[timestamp].lane_type == lane_type]


@composable
@cache_manager.create_cache('near_intersection')
def near_intersection(
    track_uuid:dict,
    log_dir:Path,
    threshold:float=5)->dict:
    """
    Identifies objects within a specified threshold of an intersection in meters.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        threshold: Distance threshold (in meters) to define "near" an intersection.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        bicycles_near_intersection = near_intersection(bicycles, log_dir, threshold=10.0)
    """


    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    lane_segments = avm.get_scenario_lane_segments()

    ls_polys = []
    for ls in lane_segments:
        if ls.is_intersection:
            ls_polys.append(ls.polygon_boundary)

    dilated_intersections = []
    for ls in ls_polys:
        dilated_intersections.append(dilate_convex_polygon(ls[:,:2], threshold))
    
    # Vectorized: for each dilated intersection, test all trajectory points at once.
    # Same multiset of (ts, polygon) hits as the per-row loop; the resulting list may
    # be in a different order, but callers always sort/setify (parity test checks both).
    near_intersection_timestamps = []
    traj_xy = np.asarray(traj)[:, :2]
    for dilated_intersection in dilated_intersections:
        mask = _is_point_in_polygon_batch(traj_xy, dilated_intersection)
        for i in np.where(mask)[0]:
            near_intersection_timestamps.append(timestamps[i])

    return near_intersection_timestamps


@composable
@cache_manager.create_cache('on_intersection')
def on_intersection(track_candidates:dict, log_dir:Path):
    """
    Identifies objects located on top of an road intersection.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        strollers_on_intersection = on_intersection(strollers, log_dir)
    """
    track_uuid = track_candidates

    scenario_lanes = get_scenario_lanes(track_uuid, log_dir)
    timestamps = scenario_lanes.keys()

    timestamps_on_intersection = []
    for timestamp in timestamps:
        if scenario_lanes[timestamp] is not None and scenario_lanes[timestamp].is_intersection:
            timestamps_on_intersection.append(timestamp)

    return timestamps_on_intersection

@composable
@cache_manager.create_cache('on_median')
def on_median(track_candidates:dict, log_dir:Path):
    """
    Identifies objects located on top of a road median (center divider, traffic island,
    or boulevard median planted strip).

    Use ONLY when the prompt says the object is ON the median surface
    ("X on the median", "X sitting on the median"). Do NOT use for "X near
    the median", "X crossing the median", or descriptive mentions where the
    median is just a road feature ("turning from a road with a large median").
    Map-derived (not VLM).

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        Filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps.

    Example:
        # "vehicle sitting on the median"
        vehicles = get_objects_of_category(log_dir, category='VEHICLE')
        vehicles_on_median = on_median(vehicles, log_dir)

        # "pedestrian on the median"
        peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
        peds_on_median = on_median(peds, log_dir)

        # "person in a wheelchair on the median"
        wheelchairs = get_objects_of_category(log_dir, category='WHEELCHAIR')
        wheelchairs_on_median = on_median(wheelchairs, log_dir)

        # "stopped vehicle sitting on the median" — compose with a motion filter
        stopped_on_median = on_median(stationary(vehicles, log_dir), log_dir)
    """
    track_uuid = track_candidates

    medians = get_median_polygons(log_dir)
    if not medians:
        return []

    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    pos_xy = np.asarray(pos)[:, :2]

    timestamps_on_median = []
    for i, (x, y) in enumerate(pos_xy):
        pt = _ShPoint(float(x), float(y))
        if any(m.covers(pt) for m in medians):
            timestamps_on_median.append(timestamps[i])

    return timestamps_on_median


@composable_relational
@cache_manager.create_cache('being_crossed_by')
def being_crossed_by(
    track_candidates:dict, 
    related_candidates:dict, 
    log_dir:Path,
    direction:Literal["forward", "backward", "left", "right"]="forward",
    in_direction:Literal['clockwise','counterclockwise','either']='either',
    forward_thresh:float=10,
    lateral_thresh:float=5)->dict:
    """
    Strict perpendicular-cross relation: the related candidate's centroid
    crosses a half-midplane of the tracked object's bounding box.

    Use ONLY for prompts where one object passes laterally across another's
    path: "X being crossed by Y", "X being overtaken on the [left|right] by Y",
    "vehicle crossed by jaywalking pedestrian". Both subject and crosser must
    have velocity (VELOCITY_THRESH = 0.2 m/s internally).

    DO NOT use for:
      * "X waiting for Y to cross"            -> X stopped + has_objects_in_relative_direction(X, Y, forward)
      * "X nearly collides with side of Y"    -> get_objects_in_relative_direction(Y, X, left|right, within_distance=5)
      * "X approaching Y" / "X approached by Y" -> APPROACHING rule (mostly has_objects_in_relative_direction)
      * "X near Y" / "X next to Y"            -> near_objects or get_objects_in_relative_direction

    Args:
        track_candidates: Tracks to analyze (subject of the cross relation).
        related_candidates: Candidates (e.g., pedestrians or vehicles) that
            cross the subject.
        log_dir: Path to scenario logs.
        direction: Axis and side of the half-midplane:
            'forward'/'backward' (longitudinal cross — e.g. jaywalking across path),
            'left'/'right'       (lateral cross — e.g. overtaking).
        in_direction: Direction the related candidate must move to cross the
            midplane ('clockwise', 'counterclockwise', or 'either'). Default
            'either' — use this unless the prompt names a specific side
            ("overtaken on the left" -> 'clockwise').
        forward_thresh: Extends the midplane region by `track.length_m/2 +
            forward_thresh` on each side of the track (NOT just `forward_thresh`).
        lateral_thresh: Two planes offset from the midplane. Once a related
            candidate crosses the midplane, it stays "crossing" until it goes
            past lateral_thresh.

    Returns:
        Filtered scenario dictionary { track_uuid: { related_uuid: [ts...] } }
        for tracks crossed by the related candidates.

    Example:
        overtaking_on_left = being_crossed_by(moving_cars, moving_cars, log_dir, direction="left", in_direction="clockwise", forward_thresh=4)
        vehicles_crossed_by_peds = being_crossed_by(vehicles, pedestrians, log_dir)
    """
    track_uuid = track_candidates
    VELOCITY_THRESH   = .2 #m/s 

    crossings = {}
    crossed_timestamps = []
    
    track = get_cuboid_from_uuid(track_uuid, log_dir)
    forward_thresh = track.length_m/2 + forward_thresh
    left_bound = -track.width_m/2
    right_bound = track.width_m/2

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        #Transform from city to tracked_object coordinate frame
        candidate_pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid, direction=direction)
        candidate_vel, timestamps = get_nth_pos_deriv(candidate_uuid, 1, log_dir, coordinate_frame=track_uuid, direction=direction)

        if len(candidate_pos) < 2:
            continue

        # Vectorized trigger detection: pairs (i-1, i) where candidate y crosses
        # either bound, |y_vel| > thresh, and x is within the forward strip.
        # Same boolean as the original 4-way OR + and conditions on line 1173.
        y0 = candidate_pos[:-1, 1]
        y1 = candidate_pos[1:,  1]
        y_vel = candidate_vel[1:, 1]
        x_at_i = candidate_pos[1:, 0]
        cross_lb = ((y0 < left_bound)  & (left_bound  < y1)) | ((y1 < left_bound)  & (left_bound  < y0))
        cross_rb = ((y0 < right_bound) & (right_bound < y1)) | ((y1 < right_bound) & (right_bound < y0))
        trigger = (cross_lb | cross_rb) & (np.abs(y_vel) > VELOCITY_THRESH) \
                  & (track.length_m/2 <= x_at_i) & (x_at_i <= forward_thresh)
        trigger_i_arr = np.where(trigger)[0] + 1  # map index 0..(n-2) back to original i = 1..(n-1)

        for i in trigger_i_arr:
            # [FIX] Was overwriting the 'direction' parameter (a string) with +/-1.
            # In the next candidate iteration, get_nth_pos_deriv(...) received the numeric value
            # instead of the original string. Use a separate variable 'cross_sign'.
            # +1 if moving left, -1 if moving right
            cross_sign = (candidate_pos[i, 1] - candidate_pos[i-1, 1]) / abs(candidate_pos[i, 1] - candidate_pos[i-1, 1])
            start_index = i - 1
            end_index = i
            updated = True

            if (cross_sign == 1 and in_direction == 'clockwise'
            or cross_sign == -1 and in_direction == 'counterclockwise'):
                #The object is not moving in the specified crossing direction
                continue

            while updated:
                updated = False
                if start_index>=0 and cross_sign*candidate_pos[start_index, 1] < lateral_thresh \
                and cross_sign*candidate_vel[start_index,1] > VELOCITY_THRESH:
                    if candidate_uuid not in crossings:
                        crossings[candidate_uuid] = []
                    crossings[candidate_uuid].append(timestamps[start_index])
                    crossed_timestamps.append(timestamps[start_index])
                    updated = True
                    start_index -= 1

                if end_index < len(timestamps) and cross_sign*candidate_pos[end_index, 1] < lateral_thresh \
                and cross_sign*candidate_vel[end_index, 1] > VELOCITY_THRESH:
                    if candidate_uuid not in crossings:
                        crossings[candidate_uuid] = []
                    crossings[candidate_uuid].append(timestamps[end_index])
                    crossed_timestamps.append(timestamps[end_index])
                    updated = True
                    end_index += 1

    return crossed_timestamps, crossings


@composable_relational
@cache_manager.create_cache('being_overtaken')
def being_overtaken(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path,
    direction:Literal["left", "right"]="left",
    lateral_thresh:float=10,
    longitudinal_thresh:float=5)->dict:
    """
    Identifies tracks that are being overtaken (passed) on their left or right
    by a related candidate moving in the same direction. An overtake is when the
    candidate travels from behind the track to in front of it while staying on
    the specified side of the track.

    Returns the TRACK (the one being passed). To make the overtaker the
    referred object (e.g. "X overtaking Y"), wrap with reverse_relationship.

    Note: only same-direction passes are detected. Oncoming or cross-traffic
    that moves opposite to the track will not trigger this.

    Matches prompts like "overtaking", "passing", "being passed by",
    "overtaken on the left/right" — same-direction maneuvers only.

    Args:
        track_candidates: Tracks being overtaken (the slower / passed object).
        related_candidates: Candidates that may overtake the track.
        log_dir: Path to scenario logs.
        direction: Track-relative side the overtaker passes on ("left" or "right").
        lateral_thresh: How far past the track's side edge the overtake corridor extends.
        longitudinal_thresh: Extra buffer past the track's front/back; the matched
            window keeps extending while the candidate stays within this buffer.

    Returns:
        Filtered scenario dict of track candidates overtaken by the related candidates.

    Example:
        overtaken_on_left = being_overtaken(moving_cars, moving_cars, log_dir, direction="left")
        car_overtaking = reverse_relationship(being_overtaken)(fast_cars, slow_cars, log_dir, direction="right")
    """
    track_uuid = track_candidates
    VELOCITY_THRESH = .2  # m/s

    crossings = {}
    crossed_timestamps = []

    track = get_cuboid_from_uuid(track_uuid, log_dir)
    half_length = track.length_m / 2
    half_width  = track.width_m  / 2

    # Side strip in track local y. left = +y side, right = -y side.
    if direction == "left":
        y_min, y_max = half_width, half_width + lateral_thresh
    else:
        y_min, y_max = -(half_width + lateral_thresh), -half_width

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        candidate_pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        candidate_vel, timestamps = get_nth_pos_deriv(candidate_uuid, 1, log_dir, coordinate_frame=track_uuid)

        if len(candidate_pos) < 2:
            continue
        
        x0 = candidate_pos[:-1, 0]
        x1 = candidate_pos[1:,  0]
        x_vel  = candidate_vel[1:, 0]
        y_at_i = candidate_pos[1:, 1]

        # Trigger: candidate's longitudinal position crosses either track edge
        # going forward (vx > 0), while inside the side strip on the chosen side.
        cross_rear  = (x0 < -half_length) & (x1 > -half_length)
        cross_front = (x0 <  half_length) & (x1 >  half_length)

        trigger = ((cross_front | cross_rear)
                   & (x_vel > VELOCITY_THRESH)
                   & (y_at_i >= y_min) & (y_at_i <= y_max))
        trigger_i_arr = np.where(trigger)[0] + 1
        if trigger_i_arr.size == 0:
            continue

        # Pre-compute expansion masks once per candidate.
        # start (backward) expansion only checks forward velocity (positional
        # check is commented out in the original loop -> preserve that).
        # end (forward) expansion: |x| < half_length + longitudinal_thresh AND vel > thresh.
        T = len(timestamps)
        vel_pos = candidate_vel[:, 0] > VELOCITY_THRESH
        start_mask = vel_pos
        end_mask = (np.abs(candidate_pos[:, 0]) < half_length + longitudinal_thresh) & vel_pos

        # For each trigger i: include the maximal run of True in start_mask
        # ending at i-1 (going backward) and the maximal run of True in
        # end_mask starting at i (going forward). Original loop interleaves
        # both directions; we keep all candidates that ANY trigger would have
        # included by taking the union via a boolean coverage array.
        coverage = np.zeros(T, dtype=bool)
        for i in trigger_i_arr:
            # Backward run: [left, i-1]
            if i > 0 and start_mask[i - 1]:
                before = start_mask[:i]
                false_idx = np.flatnonzero(~before)
                left = (false_idx[-1] + 1) if false_idx.size else 0
                coverage[left:i] = True
            # Forward run: [i, right]
            if i < T and end_mask[i]:
                after = end_mask[i:]
                false_idx = np.flatnonzero(~after)
                right = (false_idx[0] + i) if false_idx.size else T
                coverage[i:right] = True

        covered_idx = np.flatnonzero(coverage)
        if covered_idx.size == 0:
            continue
        ts_arr = [timestamps[j] for j in covered_idx]
        crossings[candidate_uuid] = ts_arr
        crossed_timestamps.extend(ts_arr)

    return crossed_timestamps, crossings


@composable_relational
@cache_manager.create_cache('near_objects')
def near_objects(
    track_uuid:dict, 
    candidate_uuids:dict, 
    log_dir:Path,
    distance_thresh:float=10, 
    min_objects:int=1,
    include_self:bool=False)->dict:
    """
    Identifies timestamps when a tracked object is near a specified set of related objects.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        related_candidates: Candidates to check for proximity (scenario dictionary).
        log_dir: Path to scenario logs.
        distance_thresh: Maximum distance in meters a related candidate can be away to be considered "near".
        min_objects: Minimum number of related objects required to be near the tracked object.

    Returns:
        dict:
            A relational scenario dictionary keyed by TRACK UUID:
                { track_uuid: { related_uuid: [timestamp_ns, ...] } }
            (Keys are track UUIDs that satisfy the proximity condition;
             inner dict maps each qualifying related UUID to the timestamps
             when proximity holds.)

    Example:
        # Cross-category proximity (vehicle near multiple peds) — use near_objects,
        # NOT group_of (group_of is for same-category cluster of the SUBJECT).
        vehicles_near_peds = near_objects(vehicles, pedestrians, log_dir, min_objects=3)
    """
    return _near_objects_inner(
        track_uuid, candidate_uuids, log_dir,
        distance_thresh=distance_thresh,
        min_objects=min_objects,
        include_self=include_self,
    )


@composable_relational
@cache_manager.create_cache('following')
def following(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path) -> dict:
    """
    Identifies timestamps when a tracked object is following behind a candidate object.

    Strict semantic: the track candidate must satisfy ALL of:
      (a) it shares lane membership with the related candidate (same lane group),
      (b) its heading is similar to the lead candidate's (cosine similarity >= 0.5),
      (c) it is positioned BEHIND the lead candidate along the lane direction.
    Not a simple distance filter.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        related_candidates: Candidates that are potentially being followed (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        A filtered scenario dictionary containing all of the tracked candidates that are likely
        following one of the related candidates.

    Example:
        car_following_bike = following(cars, bikes, log_dir)
    """
    track_uuid = track_candidates

    lead_timestamps = []
    leads = {}

    avm = get_map(log_dir)
    track_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)
    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
    track_width = track_cuboid.width_m/2
    track_length = track_cuboid.length_m/2

    FOLLOWING_THRESH = 25 + track_length #m
    LATERAL_TRHESH = 5 #m
    HEADING_SIMILARITY_THRESH = .5 #cosine similarity

    for j, candidate in enumerate(related_candidates):
        if candidate == track_uuid:
            continue

        candidate_pos, _ = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_vel, _ = get_nth_pos_deriv(candidate, 1, log_dir, coordinate_frame=track_uuid)
        candidate_yaw, timestamps = get_nth_yaw_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_lanes = get_scenario_lanes(candidate, log_dir, avm=avm)

        overlap_track_vel = track_vel[np.isin(track_timestamps, timestamps)]
        candidate_pos  = np.asarray(candidate_pos)
        candidate_vel  = np.asarray(candidate_vel)
        candidate_yaw  = np.asarray(candidate_yaw)
        overlap_track_vel = np.asarray(overlap_track_vel)

        candidate_cuboid = get_cuboid_from_uuid(candidate, log_dir)
        candidate_width = candidate_cuboid.width_m/2

        if len(timestamps) == 0:
            continue

        # Vectorized heading similarity (replaces inner ts loop, line 1351-1364).
        # Same fallback rules: use velocity direction if |vel_3d| > 0.5, otherwise
        # use cuboid yaw (candidate) or local x-axis [1,0] (track frame).
        cvel_norm_3d = np.linalg.norm(candidate_vel, axis=1)
        cvel_xy = candidate_vel[:, :2]
        cvel_xy_safe_norm = np.linalg.norm(cvel_xy + 1e-8, axis=1)
        cand_head_vel = cvel_xy / cvel_xy_safe_norm[:, None]
        cand_head_yaw = np.stack([np.cos(candidate_yaw), np.sin(candidate_yaw)], axis=1)
        candidate_heading = np.where((cvel_norm_3d > .5)[:, None], cand_head_vel, cand_head_yaw)

        tvel_norm_3d = np.linalg.norm(overlap_track_vel, axis=1)
        tvel_xy = overlap_track_vel[:, :2]
        tvel_xy_safe_norm = np.linalg.norm(tvel_xy + 1e-8, axis=1)
        track_head_vel = tvel_xy / tvel_xy_safe_norm[:, None]
        track_head_fb  = np.array([1.0, 0.0])  # track-frame x-axis (coords already in track frame)
        track_heading  = np.where((tvel_norm_3d > .5)[:, None], track_head_vel, track_head_fb)

        candidate_heading_similarity = np.sum(track_heading * candidate_heading, axis=1)

        # Distance / lateral conditions vectorized (LaneSegment compares still need a Python loop).
        cand_x = candidate_pos[:, 0]
        cand_y = candidate_pos[:, 1]
        in_forward_strip  = (track_length < cand_x) & (cand_x < FOLLOWING_THRESH)
        in_lateral_strip  = (-LATERAL_TRHESH < cand_y) & (cand_y < LATERAL_TRHESH)
        # neighbor-lane width strip: -track_width <= cand_y +/- cand_width <= track_width
        in_neighbor_strip = ((-track_width <= cand_y + candidate_width) & (cand_y + candidate_width <= track_width)) \
                          | ((-track_width <= cand_y - candidate_width) & (cand_y - candidate_width <= track_width))
        sim_ok = candidate_heading_similarity > HEADING_SIMILARITY_THRESH

        for i in range(len(timestamps)):
            tl = track_lanes[timestamps[i]]
            cl = candidate_lanes[timestamps[i]]
            if not (tl and cl):
                continue
            same_or_successor = (tl.id == cl.id) or (cl.id in tl.successors)
            is_neighbor = (tl.left_neighbor_id == cl.id) or (tl.right_neighbor_id == cl.id)
            if ((same_or_successor and in_forward_strip[i] and in_lateral_strip[i] and sim_ok[i])
                or (is_neighbor and in_forward_strip[i] and in_neighbor_strip[i] and sim_ok[i])):
                if candidate not in leads:
                    leads[candidate] = []
                leads[candidate].append(timestamps[i])
                lead_timestamps.append(timestamps[i])
        
    return lead_timestamps, leads


@composable_relational
@cache_manager.create_cache('heading_in_relative_direction_to')
def heading_in_relative_direction_to(track_candidates, related_candidates, log_dir, direction:Literal['same', 'opposite', 'perpendicular']):
    """Returns the subset of track candidates that are traveling in the given direction compared to the related canddiates.

    Arguements:
        track_candidates: The set of objects that could be traveling in the given direction
        related_candidates: The set of objects that the direction is relative to
        log_dir: The path to the log data
        direction: The direction that the positive tracks are traveling in relative to the related candidates
            "opposite" indicates the track candidates are traveling in a direction 135-180 degrees from the direction the related candidates
            are heading toward.
            "same" indicates the track candidates that are traveling in a direction 0-45 degrees from the direction the related candiates
            are heading toward.
            "same" indicates the track candidates that are traveling in a direction 45-135 degrees from the direction the related candiates
            are heading toward.

    Returns:
        the subset of track candidates that are traveling in the given direction compared to the related candidates.

    Example:
        oncoming_traffic = heading_in_relative_direction_to(vehicles, ego_vehicle, log_dir, direction='opposite')    
    """
    track_uuid = track_candidates

    track_pos, _ = get_nth_pos_deriv(track_uuid, 0, log_dir)
    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)

    traveling_in_direction_timestamps = []
    traveling_in_direction_objects = {}
    ego_to_city = get_ego_SE3(log_dir)

    # Precompute yaw-fallback directions for the track once (over all its timestamps).
    # The original per-ts fallback called get_cuboid_from_uuid -> DataFrame filter every
    # iteration; this replaces it with a single batched SE3 compose.
    track_fb = yaw_fallback_dirs(track_uuid, log_dir, track_pos, track_timestamps)

    for related_uuid in related_candidates:
        if track_uuid == related_uuid:
            continue

        related_pos, _ = get_nth_pos_deriv(related_uuid, 0, log_dir)
        related_vel, related_timestamps = get_nth_pos_deriv(related_uuid, 1, log_dir)
        related_fb = yaw_fallback_dirs(related_uuid, log_dir, related_pos, related_timestamps)
        # O(n) -> O(1) lookup: cache `timestamp -> related_timestamps index` once.
        # Replaces the per-iteration `list(related_timestamps).index(timestamp)` and
        # `timestamp in related_timestamps` (linear scan) inside the loop body.
        related_ts_to_idx = {int(t): j for j, t in enumerate(related_timestamps)}
        for i, timestamp in enumerate(track_timestamps):

            j = related_ts_to_idx.get(int(timestamp))
            if j is None:
                continue

            track_dir = track_vel[i]
            related_dir = related_vel[j]

            if np.linalg.norm(track_dir) < 1 and has_free_will(track_uuid,log_dir) and np.linalg.norm(related_dir) > 1:
                #Velocity too low to be a reliable heading direction; fall back to
                #cuboid yaw projected forward in city coords (precomputed above).
                track_dir = track_fb[i]

            elif np.linalg.norm(related_dir) < 1 and has_free_will(related_uuid,log_dir) and np.linalg.norm(track_dir) > .5:
                related_dir = related_fb[j]

            elif np.linalg.norm(track_dir) < 1 or np.linalg.norm(related_dir) < 1:
                continue

            track_dir = track_dir/np.linalg.norm(track_dir + 1e-8)
            related_dir = related_dir/np.linalg.norm(related_dir + 1e-8)
            angle = np.rad2deg(np.arccos(np.dot(track_dir, related_dir)))

            if (angle <= 45 and direction == 'same'
            or 45 < angle < 135 and direction == 'perpendicular'
            or 135 <= angle < 180 and direction == 'opposite'):
                if related_uuid not in traveling_in_direction_objects:
                    traveling_in_direction_objects[related_uuid] = []
                traveling_in_direction_objects[related_uuid].append(timestamp)
                traveling_in_direction_timestamps.append(timestamp)

    return traveling_in_direction_timestamps, traveling_in_direction_objects


@composable
@cache_manager.create_cache('stationary')
def stationary(track_candidates:dict, log_dir:Path):
    """
    Returns objects that moved less than 2m over their length of observation in the scneario.
    This object is only intended to separate parked from active vehicles. 
    Use has_velocity() with thresholding if you want to indicate vehicles that are temporarily stopped.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is stationary.

    Example:
        parked_vehicles = stationary(vehicles, log_dir)
    """
    track_uuid = track_candidates

    #Displacement threshold needed because of annotation jitter
    DISPLACMENT_THRESH = 3

    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    max_displacement = np.max(pos, axis=0) - np.min(pos, axis=0)

    if np.linalg.norm(max_displacement) < DISPLACMENT_THRESH:
        return list(timestamps)
    else:
        return []


@cache_manager.create_cache('at_stop_sign')
def at_stop_sign(track_candidates:dict, log_dir:Path, forward_thresh:float=10):
    """
    Identifies timestamps when a tracked object is in a lane corresponding to a stop sign. The tracked
    object must be within 15m of the stop sign. This may highlight vehicles using street parking near a stopped sign.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        forward_thresh: Distance in meters the vehcile is from the stop sign in the stop sign's front direction

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is at a stop sign.

    Example:
        vehicles_at_stop_sign = at_stop_sign(vehicles, log_dir)
    """

    stop_sign_uuids = get_uuids_of_category(log_dir, 'STOP_SIGN')
    return at_stop_sign_(track_candidates, stop_sign_uuids, log_dir, forward_thresh=forward_thresh)


@composable
@cache_manager.create_cache('in_drivable_area')
def in_drivable_area(track_candidates:dict, log_dir:Path)->dict:
    """
    Identifies objects within track_candidates that are within a drivable area.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is in a drivable area.

    Example:
        buses_in_drivable_area = in_drivable_area(buses, log_dir)
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    drivable_areas = avm.get_scenario_vector_drivable_areas()

    # Vectorized: test all trajectory points against each drivable polygon at once.
    # Original loop did `break` on first hit per timestamp -> use a single OR-mask
    # across polygons to dedupe (same multiset of timestamps either way).
    pos_xy = np.asarray(pos)[:, :2]
    hit_mask = np.zeros(len(timestamps), dtype=bool)
    for da in drivable_areas:
        hit_mask |= _is_point_in_polygon_batch(pos_xy, da.xyz[:, :2])

    return [timestamps[i] for i in np.where(hit_mask)[0]]


@composable 
@cache_manager.create_cache('on_road')
def on_road(
    track_candidates:dict, 
    log_dir:Path)->dict:
    """
    Identifies objects that are on a road or bike lane. 
    This function should be used in place of in_driveable_area() when referencing objects that are on a road. 
    The road does not include parking lots or other driveable areas connecting the road to parking lots.

    Args:
        track_candidates: Tracks to filter (scenario dictionary).
        log_dir: Path to scenario logs.

    Returns:
        The subset of the track candidates that are currently on a road.

    Example:
        animals_on_road = on_road(animals, log_dir)   
    """

    timestamps = []
    lanes_keyed_by_timetamp = get_scenario_lanes(track_candidates, log_dir)
    
    for timestamp, lanes in lanes_keyed_by_timetamp.items():
        if lanes is not None:
            timestamps.append(timestamp)

    return timestamps


@composable_relational
@cache_manager.create_cache('in_same_lane')
def in_same_lane(
    track_candidates:dict,
    related_candidates:dict, 
    log_dir:Path) -> dict:
    """"
    Identifies tracks that are in the same road lane as a related candidate. 

    Args:
        track_candidates: Tracks to filter (scenario dictionary)
        related_candidates: Potential objects that could be in the same lane as the track (scenario dictionary)
        log_dir: Path to scenario logs.

    Returns:
        dict: 
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is on a road lane.

    Example:
        bicycle_in_same_lane_as_vehicle = in_same_lane(bicycle, regular_vehicle, log_dir)    
    """

    track_uuid = track_candidates
    avm = get_map(log_dir)
    track_ls = get_scenario_lanes(track_uuid, log_dir, avm=avm)
    semantic_lanes = {timestamp:get_semantic_lane(ls, log_dir, avm=avm) for timestamp, ls in track_ls.items()}
    timestamps = track_ls.keys()

    same_lane_timestamps = []
    sharing_lanes = {}

    for i, related_uuid in enumerate(related_candidates):

        if related_uuid == track_uuid:
            continue

        related_ls = get_scenario_lanes(related_uuid, log_dir, avm=avm)

        for timestamp in timestamps:
            if (timestamp in related_ls and related_ls[timestamp] is not None and 
            related_ls[timestamp] in semantic_lanes[timestamp]):
                if related_uuid not in sharing_lanes:
                    sharing_lanes[related_uuid] = []
                
                same_lane_timestamps.append(timestamp)
                sharing_lanes[related_uuid].append(timestamp)

    return same_lane_timestamps, sharing_lanes


@composable_relational
@cache_manager.create_cache('on_relative_side_of_road')
def on_relative_side_of_road(
    track_candidates:dict,
    related_candidates:dict,
    log_dir:Path,
    side=Literal['same', 'opposite']) -> dict:
    """
    Identifies tracks that are on the same or opposite side of the road as a related candidate.

    Args:
        track_candidates: Tracks to filter (scenario dictionary)
        related_candidates: Potential objects that could be on the same/opposite side as the track (scenario dictionary)
        log_dir: Path to scenario logs.

    Returns:
        dict:
            A filtered scenario dictionary where keys are track UUIDs and values are lists of timestamps when the object is on a road lane.

    Example:
        police_opposite_ego = on_relative_side_of_road(police_cars, ego, log_dir, side='opposite')
    """

    track_uuid = track_candidates
    track_xy, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    track_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)
    side_lanes = {ts: get_road_side(track_lanes[ts], log_dir, side=side, avm=avm) for ts in timestamps}

    # Agent positions are only needed for the divided-road geometric fallback (side='opposite').
    use_geometry_fallback = side == 'opposite'
    track_pos = ({ts: np.asarray(track_xy[i])[:2] for i, ts in enumerate(timestamps)}
                 if use_geometry_fallback else {})

    matched_timestamps = []
    sharing_lanes = {}

    for related_uuid in related_candidates:
        if related_uuid == track_uuid:
            continue

        related_lanes = get_scenario_lanes(related_uuid, log_dir, avm=avm)
        related_pos = {}
        if use_geometry_fallback:
            related_xy, related_ts = get_nth_pos_deriv(related_uuid, 0, log_dir)
            related_pos = {ts: np.asarray(related_xy[j])[:2] for j, ts in enumerate(related_ts)}

        for ts in timestamps:
            related_lane = related_lanes.get(ts)
            if related_lane is None:
                continue

            on_requested_side = related_lane in side_lanes[ts]

            # Divided-road (median) fallback: lane topology found no opposite lanes here.
            if not on_requested_side and use_geometry_fallback and not side_lanes[ts]:
                on_requested_side = _opposite_across_road(
                    track_lanes.get(ts), related_lane, track_pos.get(ts), related_pos.get(ts))

            if on_requested_side:
                sharing_lanes.setdefault(related_uuid, []).append(ts)
                matched_timestamps.append(ts)

    return matched_timestamps, sharing_lanes


@cache_manager.create_cache('scenario_and')
def scenario_and(scenario_dicts:list[dict])->dict:
    """
    Returns a composed scenario where the track objects are the intersection of all of the track objects
    with the same uuid and timestamps.

    Args:
        scenario_dicts: the scenarios to combine 

    Returns:
        dict:
            a filtered scenario dictionary that contains tracked objects found in all given scenario dictionaries
    
    Example:
        jaywalking_peds = scenario_and([peds_on_road, peds_not_on_pedestrian_crossing])

    """
    composed_dict = {}

    composed_track_dict = deepcopy(reconstruct_track_dict(scenario_dicts[0]))
    for i in range(1, len(scenario_dicts)):
        scenario_dict = scenario_dicts[i]
        track_dict = reconstruct_track_dict(scenario_dict)
        
        for track_uuid, timestamps in track_dict.items():
            if track_uuid not in composed_track_dict:
                continue

            composed_track_dict[track_uuid] = sorted(set(composed_track_dict[track_uuid]).intersection(timestamps))

        for track_uuid in list(composed_track_dict.keys()):
            if track_uuid not in track_dict:
                composed_track_dict.pop(track_uuid)

    for track_uuid, intersecting_timestamps in composed_track_dict.items():
        for scenario_dict in scenario_dicts:
            if track_uuid not in composed_dict:
                composed_dict[track_uuid] = scenario_at_timestamps(scenario_dict[track_uuid], intersecting_timestamps)
            else:
                related_children =  scenario_at_timestamps(scenario_dict[track_uuid],intersecting_timestamps)

                if isinstance(related_children, dict) and isinstance(composed_dict[track_uuid], dict):
                    composed_dict[track_uuid] = scenario_or([composed_dict[track_uuid], related_children])
                elif isinstance(related_children, dict) and not isinstance(composed_dict[track_uuid], dict):
                    related_children[track_uuid] = composed_dict[track_uuid]
                    composed_dict[track_uuid] = related_children
                elif not isinstance(related_children, dict) and isinstance(composed_dict[track_uuid], dict):
                    composed_dict[track_uuid][track_uuid] = related_children
                else:
                    composed_dict[track_uuid] = set(composed_dict[track_uuid]).intersection(related_children)

    return composed_dict


@cache_manager.create_cache('scenario_or')
def scenario_or(scenario_dicts:list[dict]):
    """
    Returns a composed scenario where that tracks all objects and relationships in all of the input scenario dicts.

    Args:
        scenario_dicts: the scenarios to combine 

    Returns:
        dict:
            an expanded scenario dictionary that contains every tracked object in the given scenario dictionaries
    
    Example:
        be_cautious_around = scenario_or([animal_on_road, stroller_on_road])
    """

    composed_dict = deepcopy(scenario_dicts[0])
    for i in range(1, len(scenario_dicts)):
        for track_uuid, child in scenario_dicts[i].items():
            if track_uuid not in composed_dict:
                composed_dict[track_uuid] = child
            elif isinstance(child, dict) and isinstance(composed_dict[track_uuid], dict):
                composed_dict[track_uuid] = scenario_or([composed_dict[track_uuid], child])
            elif isinstance(child, dict) and not isinstance(composed_dict[track_uuid], dict):
                child[track_uuid] = composed_dict[track_uuid]
                composed_dict[track_uuid] = child
            elif not isinstance(child, dict) and isinstance(composed_dict[track_uuid], dict):
                composed_dict[track_uuid][track_uuid] = child
            else:
                composed_dict[track_uuid] = set(composed_dict[track_uuid]).union(child)

    return composed_dict


def reverse_relationship(func):
    """
    Wraps relational functions to switch the top level tracked objects and relationships formed by the function. 

    Args:
        relational_func: Any function that takes track_candidates and related_candidates as its first and second arguements

    Returns:
        dict:
            scenario dict with swapped top-level tracks and related candidates

    Example:
        group_of_peds_near_vehicle = reverse_relationship(near_objects)(vehicles, peds, log_dir, min_objects=3)
    """
    def wrapper(track_candidates, related_candidates, log_dir, *args, **kwargs):

        if func.__name__ == 'get_objects_in_relative_direction':
            return has_objects_in_relative_direction(track_candidates, related_candidates, log_dir, *args, **kwargs)

        track_dict = to_scenario_dict(track_candidates, log_dir)
        related_dict = to_scenario_dict(related_candidates, log_dir)
        remove_empty_branches(track_dict)
        remove_empty_branches(related_dict)

        scenario_dict:dict = func(track_dict, related_dict, log_dir, *args, **kwargs)
        remove_empty_branches(scenario_dict)

        #Look for new relationships
        tc_uuids = list(track_dict.keys())
        rc_uuids = list(related_dict.keys())

        new_relationships = []
        for track_uuid, related_objects in scenario_dict.items():
            for related_uuid in related_objects.keys():
                # [FIX] Operator precedence (and > or) made 'track_uuid != related_uuid' apply only to the second OR branch.
                # A self-relationship (uuid, uuid) could be added when the first branch was true.
                # Parenthesize so the self-check guards both branches.
                if ((track_uuid in tc_uuids and related_uuid in rc_uuids)
                    or (track_uuid in rc_uuids and related_uuid in tc_uuids)) \
                    and track_uuid != related_uuid:
                    new_relationships.append((track_uuid, related_uuid))

        #Reverese the scenario dict using these new relationships
        reversed_scenario_dict = {}
        for track_uuid, related_uuid in new_relationships:
            related_timestamps = get_scenario_timestamps(scenario_dict[track_uuid][related_uuid])
            removed_related:dict = deepcopy(scenario_dict[track_uuid])

            # I need a new data structure
            for track_uuid2, related_uuid2 in new_relationships:
                if track_uuid2 == track_uuid:
                    removed_related.pop(related_uuid2)

            if len(removed_related) == 0 or len(get_scenario_timestamps(removed_related)) == 0:
                removed_related = related_timestamps

            filtered_removed_related = scenario_at_timestamps(removed_related, related_timestamps)
            filtered_removed_related = {track_uuid : filtered_removed_related}

            if related_uuid not in reversed_scenario_dict:
                reversed_scenario_dict[related_uuid] = filtered_removed_related
            else:
                reversed_scenario_dict[related_uuid] = scenario_or([filtered_removed_related, reversed_scenario_dict[related_uuid]])

        return reversed_scenario_dict
    return wrapper


def scenario_not(func):
    """
    Wraps composable functions to return the difference of the input track dict and output scenario dict.
    Using scenario_not with a composable relational function will not return any relationships.

    Args:
        composable_func: Any function that takes track_candidates as its first input

    Returns:

    Example:
        # Generic negation of a composable filter.
        peds_not_at_crossing = scenario_not(at_pedestrian_crossing)(peds, log_dir)
        # For "moving / active" use active() directly; for "parked / stationary"
        # use stationary() directly.
    """
    def wrapper(track_candidates, *args, **kwargs):

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Determine the position of 'log_dir'
        if 'log_dir' in params:
            log_dir_index = params.index('log_dir') - 1
        else:
            raise ValueError("The function scenario_not wraps does not have 'log_dir' as a parameter.")

        log_dir = args[log_dir_index]

        if func.__name__ == 'get_objects_in_relative_direction':
            track_dict = to_scenario_dict(args[0], log_dir)
        else:
            track_dict = to_scenario_dict(track_candidates, log_dir)

        if log_dir_index == 0:
            scenario_dict = func(track_candidates, log_dir, *args[1:], **kwargs)
        elif log_dir_index == 1:
            #composable_relational function
            scenario_dict = func(track_candidates, args[0], log_dir, *args[2:], **kwargs)

        remove_empty_branches(scenario_dict)
        not_dict = {track_uuid: [] for track_uuid in track_dict.keys()}

        for uuid in not_dict:
            if uuid in scenario_dict:
                not_timestamps = list(set(
                    get_scenario_timestamps(track_dict[uuid])).difference(get_scenario_timestamps(scenario_dict[uuid])))
                
                not_dict[uuid] = scenario_at_timestamps(track_dict[uuid], not_timestamps)
            else:
                not_dict[uuid] = track_dict[uuid]

        return not_dict
    return wrapper


@cache_manager.create_cache('between_two_objects')
def between_two_objects(
    track_candidates: dict,
    anchors_a: dict,
    anchors_b: dict,
    log_dir: Path,
    max_pair_distance: float = 35.0,
    max_perp_distance: float = 2.0,
    max_anchor_distance: float = 15.0,
    min_anchor_distance: float = 1.0,
    pair_must_differ: bool = True,
    min_pairs: int = 1,
) -> dict:
    """
    Identifies tracks that are spatially between two anchor sets — axis-free geometric primitive.

    A subject S is "between" anchor a (from anchors_a) and anchor b (from anchors_b) at timestamp t
    if there exists a pair (a, b) at t such that S lies on (or within max_perp_distance of) the line
    segment a-b. Concretely:

        - ||a - b|| <= max_pair_distance
        - proj_t = ((S - a) . (b - a)) / ||b - a||^2 in [0, 1]
        - perpendicular distance from S to line ab <= max_perp_distance
        - a != b when pair_must_differ is True (use anchors_a == anchors_b for the symmetric case)

    Unlike has_objects_in_relative_direction, this primitive uses the world frame and does not
    depend on the subject's heading — robust to noisy yaw (e.g. pedestrians).

    Args:
        track_candidates: Subjects (scenario dictionary). Output is keyed by these uuids.
        anchors_a: First anchor set (scenario dictionary). Pass the same dict twice for symmetric "between two X".
        anchors_b: Second anchor set (scenario dictionary). Different from anchors_a for asymmetric anchors.
        log_dir: Path to scenario logs.
        max_pair_distance: Maximum allowed ||a - b||. Default 35.0 m (covers L-axis sandwiches like lane merging).
        max_perp_distance: Maximum allowed perpendicular distance from subject to line a-b. Default 2.0 m.
        max_anchor_distance: Maximum allowed ||subject - anchor|| for each anchor in the pair. Default 15.0 m.
            Caps how far an anchor can be from the subject. Prevents wide-spread anchor pairs that
            geometrically include the subject but are semantically unrelated (e.g. long barrier line).
        min_anchor_distance: Minimum allowed ||subject - anchor|| for each anchor. Default 1.0 m.
            Tangential safety: rejects pairs where subject is right next to an anchor.
        pair_must_differ: When True, a == b pairs are excluded (required for symmetric anchors).
        min_pairs: Minimum number of distinct valid (a, b) pairs at a timestamp for the subject to qualify.

    Default values were derived from a parallel grid search over 47 configurations on val set
    between-prompts (best HOTA-T = 0.368 with these params, vs LLM baseline 0.306).

    Returns:
        dict: Subject-keyed relational scenario dictionary
              { subj_uuid: { anchor_uuid: [timestamp_ns, ...], ... }, ... }
              Inner anchor uuids are the union of all valid anchors that participated in a qualifying pair.

    Example:
        motorcycles = get_objects_of_category(log_dir, category='MOTORCYCLE')
        vehicles    = get_objects_of_category(log_dir, category='VEHICLE')
        between     = between_two_objects(motorcycles, vehicles, vehicles, log_dir)
    """
    # ── [Step 1] Input normalization ──────────────────────────────────────
    # Accept list / set / dict from callers and convert to a unified scenario_dict.
    subject_dict = to_scenario_dict(track_candidates, log_dir)
    a_dict = to_scenario_dict(anchors_a, log_dir)
    b_dict = to_scenario_dict(anchors_b, log_dir)

    subject_uuids = list(subject_dict.keys())
    a_uuids = list(a_dict.keys())
    b_uuids = list(b_dict.keys())

    # If any of the three sets is empty the between relation cannot hold -> empty dict.
    if not subject_uuids or not a_uuids or not b_uuids:
        return {}

    # ── [Step 2] Precompute anchor positions bucketed by timestamp ────────
    # Re-looking up trajectories per (subject × ts × a × b) tuple is O(N^2);
    # bucketing by ts reduces the inner loop to a dict lookup.
    #
    # IMPORTANT: respect the anchor scenario_dict values (= allowed timestamps).
    # Composable filters such as stationary() / turning() only keep ts where the
    # object satisfies the predicate and leave other ts as empty lists. The
    # trajectory itself, however, exists at every frame; skipping the
    # allowed_ts filter would let a "stationary vehicle at non-stopped ts"
    # contradiction slip in as an anchor.
    def collect_positions_by_ts(anchor_dict):
        # {timestamp_ns: [(uuid, xy), ...]}
        by_ts: dict[int, list[tuple[str, np.ndarray]]] = {}
        for uuid, scen in anchor_dict.items():
            # The dict value may be a list of ts, a nested relational dict, or
            # None — get_scenario_timestamps flattens it into an allowed-ts set.
            allowed_ts = set(int(t) for t in get_scenario_timestamps(scen))
            if not allowed_ts:
                continue   # empty composable entry -> this uuid is not an anchor candidate
            pos, ts = get_nth_pos_deriv(uuid, 0, log_dir)   # world frame trajectory
            for i in range(len(ts)):
                t = int(ts[i])
                if t not in allowed_ts:
                    continue   # skip ts outside the allowed set
                # Only store xy (axis-free between uses planar distance; z irrelevant).
                by_ts.setdefault(t, []).append((uuid, pos[i, :2].astype(np.float64)))
        return by_ts

    a_by_ts = collect_positions_by_ts(a_dict)
    b_by_ts = collect_positions_by_ts(b_dict)

    # ── [Step 3] Square the distance thresholds (avoid sqrt per pair) ─────
    # Calling sqrt for every pair is expensive; compare squared distances instead.
    max_pair_d2   = float(max_pair_distance) ** 2
    max_perp_d2   = float(max_perp_distance) ** 2
    max_anchor_d2 = float(max_anchor_distance) ** 2 if max_anchor_distance != float('inf') else float('inf')
    min_anchor_d2 = float(min_anchor_distance) ** 2

    # ── [Step 4] Iterate subject × ts × (a, b) pair + 5 geometric checks ──
    # Pass conditions (ALL must hold):
    #   (1) ‖a − b‖ ≤ max_pair_distance              (pair length cap)
    #   (2) proj_t = ((S−a)·(b−a)) / ‖b−a‖² ∈ [0, 1]  (S's foot of perpendicular is inside segment)
    #   (3) ‖(S−a) − proj_t·(b−a)‖ ≤ max_perp_distance  (perpendicular distance to segment)
    #   (4) ‖S−a‖, ‖S−b‖ ≤ max_anchor_distance       (each anchor not too far)
    #   (5) ‖S−a‖, ‖S−b‖ ≥ min_anchor_distance       (avoid tangential degeneracy)
    # Additionally: a ≠ b when pair_must_differ, and subject ≠ anchor uuid.
    #
    # Efficiency — check the anchor distance gates (4)(5) first to short-circuit
    # before the more expensive perp/proj_t math. Helps runtime as well as correctness.
    result: dict = {}
    for subj_uuid in subject_uuids:
        # Respect the subject's allowed ts just like the anchor side does.
        # A composable filter such as scenario_not(stationary)(vehicles, log_dir)
        # only keeps ts where the object passes; without this gate the entire
        # trajectory of a stationary vehicle would re-enter the between check,
        # and an object that step2_moving filtered out could be re-tagged as
        # REFERRED by step4_between (regression bug).
        subj_allowed_ts = set(int(t) for t in get_scenario_timestamps(subject_dict[subj_uuid]))
        if not subj_allowed_ts:
            continue
        subj_pos, subj_ts = get_nth_pos_deriv(subj_uuid, 0, log_dir)
        related_dict: dict[str, list[int]] = {}   # {anchor_uuid: [ts...]} for this subject

        for i in range(len(subj_ts)):
            t = int(subj_ts[i])
            if t not in subj_allowed_ts:
                continue   # subject not a candidate at this ts (e.g. stationary)
            s_xy = subj_pos[i, :2].astype(np.float64)
            a_list = a_by_ts.get(t)   # a anchors available at this ts
            b_list = b_by_ts.get(t)   # b anchors available at this ts
            if not a_list or not b_list:
                continue   # one side has no anchor at this ts -> between impossible

            count = 0                            # number of (a, b) pairs passing at this ts
            involved_this_ts: set[str] = set()   # all anchor uuids that appear in any passing pair

            for a_uuid, a_xy in a_list:
                if a_uuid == subj_uuid:
                    continue   # (0) subject cannot be its own anchor

                # (4)(5) S-a distance gate (quick reject before looking at b)
                sa = s_xy - a_xy
                sa_d2 = float(sa @ sa)
                if sa_d2 > max_anchor_d2 or sa_d2 < min_anchor_d2:
                    continue

                for b_uuid, b_xy in b_list:
                    if b_uuid == subj_uuid:
                        continue   # (0) subject cannot be its own anchor
                    if pair_must_differ and a_uuid == b_uuid:
                        continue   # remove the trivial a == b pair in the SYM case

                    # (4)(5) S-b distance gate
                    sb = s_xy - b_xy
                    sb_d2 = float(sb @ sb)
                    if sb_d2 > max_anchor_d2 or sb_d2 < min_anchor_d2:
                        continue

                    # (1) pair length check + (2) proj_t denominator safety
                    ab = b_xy - a_xy
                    ab2 = float(ab @ ab)            # ‖a − b‖²
                    if ab2 < 1e-6 or ab2 > max_pair_d2:
                        continue   # a ≈ b -> degenerate segment / too long -> reject

                    # (2) proj_t — normalized coord of S's foot of perpendicular on segment ab
                    proj_t = float(((s_xy - a_xy) @ ab) / ab2)
                    if proj_t < 0.0 or proj_t > 1.0:
                        continue   # S is outside the segment range -> not between

                    # (3) perpendicular distance from S to segment
                    perp_vec = (s_xy - a_xy) - proj_t * ab
                    perp_d2 = float(perp_vec @ perp_vec)
                    if perp_d2 > max_perp_d2:
                        continue   # too far from the line

                    # ✅ all 5 checks passed — this (a, b) pair is valid
                    count += 1
                    involved_this_ts.add(a_uuid)
                    involved_this_ts.add(b_uuid)

            # If this ts has >= min_pairs valid pairs, accept the subject as between
            # and record every anchor that participated in a passing pair.
            if count >= min_pairs:
                for r_uuid in involved_this_ts:
                    related_dict.setdefault(r_uuid, []).append(t)

        # ── [Step 5] Store subject result (anchor-wise: sort + dedupe ts) ──
        if related_dict:
            result[subj_uuid] = {r: sorted(set(ts)) for r, ts in related_dict.items()}

    return result

# A-TOM Made Functions
@composable_relational
@cache_manager.create_cache('group_of')
def group_of(
    track_candidates: dict,
    related_candidates: dict,
    log_dir: Path,
    min_objects: int = 3,
    within_distance: float = 3.0,
) -> dict:
    """
    Returns track_candidates that belong to a group of >= min_objects same-category
    objects within within_distance. **min_objects COUNTS THE TRACK ITSELF**:
    "group of three" -> min_objects=3 (the track + 2 neighbours).

    Thin semantic wrapper around near_objects() for prompts that mention 'group'.

    Args:
        track_candidates: tracks to test for group membership.
        related_candidates: candidates that may form a group with the track
            (typically the same set as track_candidates, e.g. peds & peds).
        log_dir: scenario logs path.
        min_objects: minimum total group size including the track itself.
        within_distance: maximum spacing (m) between any two group members.

    Returns:
        A scenario dict mapping each group member uuid to its qualifying
        neighbours and timestamps. Same shape as near_objects().

    Translation to near_objects: near_objects counts OTHER candidates only, so
    'group of three including self' (min_objects=3) maps to near_objects'
    min_objects=2.

    Example:
        # group of pedestrians
        peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
        ped_groups = group_of(peds, peds, log_dir, min_objects=3)

        # vehicle heading toward a pedestrian group
        moving_vehicles = active(
            get_objects_of_category(log_dir, category='VEHICLE'), log_dir)
        v_heading = heading_toward(moving_vehicles, ped_groups, log_dir)
    """
    # Clamp min_objects: a "group of 1" is trivial. Values <= 1 would map to
    # _near_objects_inner(min_objects=0), which falls into the
    # `if not min_objects: min_objects = len(candidate_uuids)` branch and
    # silently flips the meaning to "all candidates near". Force pair semantics.
    min_objects = max(2, int(min_objects))
    return _near_objects_inner(
        track_candidates, related_candidates, log_dir,
        distance_thresh=within_distance,
        min_objects=min_objects - 1,
        include_self=False,
    )


def _near_objects_inner(track_uuid, candidate_uuids, log_dir,
                        distance_thresh: float, min_objects: int, include_self: bool):
    """
    Internal helper: shared body for near_objects() and group_of().

    Counts how many candidates are within distance_thresh of track_uuid at each
    timestamp; emits timestamps where the count reaches min_objects.

    Args:
        track_uuid: single track uuid (called from inside composable_relational).
        candidate_uuids: iterable of candidate uuids.
        log_dir: scenario logs path.
        distance_thresh: max cuboid distance (m) to count as "near".
        min_objects: minimum number of qualifying candidates per timestamp.
            Note: candidates exclude the track itself unless include_self=True,
            so the count is always over OTHER objects.
        include_self: if False, skip candidate == track_uuid.

    Returns:
        (timestamps, near_objects_dict) — same shape as composable_relational
        functions return inside the decorator.
    """
    if not min_objects:
        min_objects = len(candidate_uuids)

    near_objects_dict = {}
    for candidate in candidate_uuids:
        if candidate == track_uuid and not include_self:
            continue

        _, timestamps = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        if len(timestamps) == 0:
            continue

        # Batch all per-timestamp distances in a single call, then mask.
        distances = cuboid_distance_batch(track_uuid, candidate, log_dir, timestamps)
        mask = distances <= distance_thresh
        for i in np.where(mask)[0]:
            timestamp = timestamps[i]
            if timestamp not in near_objects_dict:
                near_objects_dict[timestamp] = []
            near_objects_dict[timestamp].append(candidate)

    timestamps = []
    keys = list(near_objects_dict.keys())
    for timestamp in keys:
        if len(near_objects_dict[timestamp]) >= min_objects:
            timestamps.append(timestamp)
        else:
            near_objects_dict.pop(timestamp)

    near_objects_dict = swap_keys_and_listed_values(near_objects_dict)

    return timestamps, near_objects_dict

@cache_manager.create_cache('active')
def active(track_candidates:dict, log_dir:Path, min_speed:float=0.5)->dict:
    """
    Returns objects that are actively moving (i.e., not parked). Preferred
    wrapper for generic motion language ("moving", "walking", "driving",
    "active") over scenario_not(stationary).

    Applicable to ANY motion-tracking subject: vehicles, pedestrians,
    bicyclists, motorcyclists, etc.

    An object is active when it is **not stationary over the whole scenario**,
    i.e. scenario_not(stationary). With the default min_speed (0.5 m/s,
    annotation-jitter floor), per-timestamp velocity adds no extra filter and
    the result equals scenario_not(stationary)(track_candidates, log_dir);
    timestamps where the moving object is briefly halted (e.g. waiting at a
    stop sign) are kept. Pass a larger min_speed to additionally restrict the
    output to timestamps where the instantaneous speed is at least that value.

    Do NOT wrap active() on a track set whose filter already implies motion:
    at_pedestrian_crossing, being_crossed_by, turning, changing_lanes,
    accelerating, has_lateral_acceleration. Those filters already select an
    active subset; stacking active() on top is redundant AND narrows the
    timestamp window unnecessarily.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        min_speed: Minimum instantaneous speed in m/s. Values above 0.5 also
            intersect with has_velocity(min_velocity=min_speed).
            Convert km/h to m/s: m/s = km/h / 3.6.

    Returns:
        dict:
            A filtered scenario dictionary keyed by track_uuid.

    Example:
        active_vehicles = active(get_objects_of_category(log_dir, 'VEHICLE'), log_dir)
        fast_vehicles = active(vehicles, log_dir, min_speed=5.0)
        # Same for cyclists/peds — applicable to any motion-tracking subject
        cyclists = get_objects_of_category(log_dir, category='BICYCLIST')
        moving_cyclists = active(cyclists, log_dir)
    """
    not_stationary = scenario_not(stationary)(track_candidates, log_dir)
    if min_speed <= 0.5:
        return not_stationary
    moving = has_velocity(track_candidates, log_dir, min_velocity=min_speed)
    return scenario_and([not_stationary, moving])

@composable
@cache_manager.create_cache('near_construction_objects')
def near_construction_objects(
    track_candidates: dict,
    log_dir: Path,
    distance_thresh: float = 12.0,
    min_objects: int = 1,
) -> dict:
    """
    Identifies tracks within distance_thresh of at least min_objects
    construction objects (CONSTRUCTION_BARREL, CONSTRUCTION_CONE, or SIGN).

    Distinct from near_infrastructure(..., 'construction_zone'): this function
    is 3D proximity to physical construction markers (per-track, per-timestamp),
    whereas near_infrastructure uses Scene Context VLM labels (scene-level).
    Prefer this for "X near/in construction zone", "X near construction
    cone/barrel", "X at work zone".

    SIGN is included because construction work zones are commonly marked by
    temporary signage (e.g., "ROAD WORK AHEAD", "DETOUR") alongside barrels
    and cones.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        distance_thresh: Maximum cuboid distance from a construction object (m).
        min_objects: Minimum number of construction objects (barrels, cones,
            and signs combined) within distance_thresh required to qualify.

    Returns:
        Filtered scenario dictionary containing tracks within distance_thresh of
        at least min_objects construction objects per timestamp.

    Example:
        vehicles_near_construction = near_construction_objects(vehicles, log_dir)
    """
    barrels = get_objects_of_category(log_dir, category='CONSTRUCTION_BARREL')
    cones = get_objects_of_category(log_dir, category='CONSTRUCTION_CONE')
    signs = get_objects_of_category(log_dir, category='SIGN')
    construction_objs = scenario_or([barrels, cones, signs])

    # Call the inner helper directly: this function is already wrapped by
    # @composable, so its body runs inside a worker pool. Calling near_objects()
    # (which is @composable_relational and spawns another pool) would trigger
    # "daemonic processes are not allowed to have children". Same pattern as
    # group_of() → _near_objects_inner().
    return _near_objects_inner(
        track_candidates,
        construction_objs,
        log_dir,
        distance_thresh=distance_thresh,
        min_objects=min_objects,
        include_self=False,
    )
    
@composable
@cache_manager.create_cache('in_turn_lane')
def in_turn_lane(
    track_candidates: dict,
    log_dir: Path,
    side: Literal['left', 'right', 'center', None] = None,
) -> dict:
    """
    Identifies tracks currently in a turn lane — either inside an intersection
    turn lane, or in a lane immediately preceding one.

    Location condition (per ts, both filtered by `side`):
    - (a) The current lane is inside an intersection AND the lane itself is a
      turn lane (get_turn_direction(ls) matches side), OR
    - (b) The current lane has at least one successor whose get_turn_direction
      matches side (mixed straight+turn lanes still qualify — at least one
      successor is a turn lane).

    The 'center' option is reserved for center / median two-way left-turn
    lanes (commonly outside intersections) and currently returns an empty
    result — kept in the signature so LLM-generated callers that pass it
    don't TypeError.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        side: Filter by turn direction.
            None: any turn (left or right) qualifies.
            'left': only left turns.
            'right': only right turns.
            'center': center turn lane (currently unsupported, returns empty).

    Returns:
        Scenario dictionary mapping each input track UUID to the sorted list
        of timestamps where that track is in / approaching a turn lane
        matching the side filter.

    Example:
        left_turn_lane_vehicles = in_turn_lane(vehicles, log_dir, side='left')
        vehicles_in_turn_lanes  = in_turn_lane(vehicles, log_dir)
    """
    track_uuid = track_candidates

    if side == 'left':
        target_dirs = {'left'}
    elif side == 'right':
        target_dirs = {'right'}
    elif side == 'center':
        return []
    elif side is None:
        target_dirs = {'left', 'right'}
    else:
        return []

    avm = get_map(log_dir)
    scenario_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)

    result_ts: set = set()
    for ts, ls in scenario_lanes.items():
        if ls is None:
            continue

        # (a) The current lane is itself a turn lane inside an intersection.
        if ls.is_intersection and get_turn_direction(ls) in target_dirs:
            result_ts.add(int(ts))
            continue

        # (b) A successor lane is an intersection turn lane (the current lane is right before the intersection).
        if not ls.successors:
            continue
        for succ_id in ls.successors:
            succ_ls = avm.vector_lane_segments.get(succ_id)
            if succ_ls is None:
                continue
            if get_turn_direction(succ_ls) in target_dirs:
                result_ts.add(int(ts))
                break

    return sorted(result_ts)

@composable
@cache_manager.create_cache('waiting_to_turn')
def waiting_to_turn(
    track_candidates: dict,
    log_dir: Path,
    side: Literal['left', 'right', 'center', None] = None,
    max_speed: float = 2.0,
) -> dict:
    """
    Identifies tracks that are waiting to turn — the front-third reference
    point of the cuboid is inside an intersection turn lane while the track is
    moving at low speed.

    Location condition (per ts, filtered by `side`):
    - A lane containing the track's "front-1/3 point" (cuboid centroid shifted
      by length/6 forward along its yaw) is an intersection lane AND its turn
      direction (get_turn_direction(ls)) matches `side`.
    - The front-1/3 reference fires as soon as the leading third of the
      vehicle has entered the intersection turn lane (instead of waiting for
      the centroid to enter).

    Speed condition (per ts): the track's instantaneous speed at that ts is
    <= max_speed. Computed directly from get_nth_pos_deriv(..., 1) instead of
    has_velocity so that fully-stationary tracks (e.g., stopped at a red
    arrow) are not excluded by the trajectory-level stationary filter inside
    has_velocity, and so the [0, max_speed] band is honored end-to-end.

    The 'center' option is reserved for center / median two-way left-turn
    lanes and currently returns an empty result.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        side: Filter by turn direction.
            None: any turn (left or right) qualifies.
            'left': only left turns.
            'right': only right turns.
            'center': returns empty (not supported).
        max_speed: Maximum speed (m/s) to qualify as "waiting". Default 0.5.

    Returns:
        Sorted list of timestamps where the track is inside an intersection on
        a turn lane matching the side filter and moving at low speed.

    Example:
        waiting_left_turn_vehicles = waiting_to_turn(vehicles, log_dir, side='left')
    """
    track_uuid = track_candidates

    if side == 'left':
        target_dirs = {'left'}
    elif side == 'right':
        target_dirs = {'right'}
    elif side == 'center':
        return []
    elif side is None:
        target_dirs = {'left', 'right'}
    else:
        return []

    avm = get_map(log_dir)

    # Reference point: 1/3 from the front of the cuboid (= length/6 forward of
    # the centroid along the track's forward heading), in city coords.
    center_city, ts_arr = get_nth_pos_deriv(track_uuid, 0, log_dir)
    if len(ts_arr) == 0:
        return []
    cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
    forward_offset = (cuboid.length_m / 6.0) if cuboid is not None else 0.0
    fwd_vec = yaw_fallback_dirs(track_uuid, log_dir, center_city, ts_arr)
    fwd_unit = fwd_vec / (np.linalg.norm(fwd_vec, axis=1, keepdims=True) + 1e-9)
    ref_pts = np.asarray(center_city) + forward_offset * fwd_unit

    # Batch all per-timestamp lane lookups in a single call: groups ref_pts by
    # 1m-quantized cell so Stage-1 (`_nearby_lane_candidates`) is amortized,
    # and runs Stage-2 polygon checks vectorized per-polygon across all points
    # in the cell. Same per-position results as calling get_lane_segments per ts.
    segments_per_ts = get_lane_segments_batch(avm, ref_pts)

    location_ts: set = set()
    for i, t in enumerate(ts_arr):
        # Intersection turn lane check: among lanes containing the front-1/3
        # reference point, accept if any lane is an intersection lane whose
        # turn direction matches target_dirs.
        for ls in segments_per_ts[i]:
            if ls.is_intersection and get_turn_direction(ls) in target_dirs:
                location_ts.add(int(t))
                break

    if not location_ts:
        return []

    # Per-ts low-speed check: compute velocity directly via get_nth_pos_deriv(..., 1).
    # has_velocity is unusable here because (1) its default min_velocity=0.5
    # collides with max_velocity=0.5, and (2) it short-circuits to [] for
    # trajectory-level stationary tracks — which would drop fully-stopped
    # vehicles waiting at a red arrow. Run a per-ts check directly to avoid both.
    # Vectorized: same mask as the per-row `norm(vel) <= max_speed` check.
    vels, vel_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    vels = np.asarray(vels)
    if len(vels):
        speeds = np.linalg.norm(vels, axis=1)
        low_speed_ts = {int(vel_timestamps[i]) for i in np.where(speeds <= max_speed)[0]}
    else:
        low_speed_ts: set = set()

    return sorted(location_ts & low_speed_ts)

@composable
@cache_manager.create_cache('braking')
def braking(
    track_candidates: dict,
    log_dir: Path,
    max_accel: float = -0.65,
) -> dict:
    """
    Identifies tracks that are braking (forward deceleration ≤ max_accel m/s²).

    Thin wrapper around accelerating(): selects frames whose forward acceleration
    falls in (-inf, max_accel]. Use this for generic "braking" prompts where the
    natural-language strength is unspecified ("is braking", "slowing to stop",
    "braking at crosswalk"). For prompts that explicitly call out *hard* or
    *heavy* braking, prefer braking_hard() (default -2.5 m/s²) instead.

    The default threshold -0.65 follows the calibration note in accelerating()'s
    docstring: "Values under -0.65 reliably indicates braking". This is sloppier
    than braking_hard so it catches plateau / late-stage braking frames that
    sit between gentle coasting and a true hard-stop event, but tight enough
    to reject noise-driven negative-accel spikes during cruise.

    Args:
        track_candidates: The tracks to analyze (scenario dictionary).
        log_dir: Path to the directory containing scenario logs.
        max_accel: Upper bound of forward acceleration considered "braking"
            (m/s²). Default -0.65; pass a stricter value (e.g. -1.5, -2.0)
            to require firmer braking, or use braking_hard() for the
            heavy-braking endpoint (-2.5).

    Returns:
        Filtered scenario dictionary with timestamps where forward accel
        ≤ max_accel.

    Examples:
        # 1) Plain "vehicle is braking".
        braking_vehicles = braking(vehicles, log_dir)

        # 2) "vehicle two cars ahead is braking" — compose with
        #    nth_object_in_direction so the predicate applies only to the
        #    2nd vehicle in front of ego.
        vehicles = get_objects_of_category(log_dir, category='VEHICLE')
        second_ahead = nth_object_in_direction(vehicles, log_dir,
                                               direction='forward', n=2)
        braking_second_ahead = braking(second_ahead, log_dir)

        # 3) "vehicle braking at pedestrian crossing" — intersect with
        #    at_pedestrian_crossing so only crossing-located braking events
        #    survive.
        braking_at_xing = scenario_and([
            braking(vehicles, log_dir),
            at_pedestrian_crossing(vehicles, log_dir),
        ])

        # 4) Stricter cutoff (e.g. semi-heavy braking).
        firm_braking = braking(vehicles, log_dir, max_accel=-1.5)

        # 5) If you need "hard / heavy" braking specifically, switch to
        #    braking_hard (default -2.5).
        # heavy = braking_hard(vehicles, log_dir)
    """
    # accelerating() is @composable; calling it from inside another @composable
    # body would spawn a nested worker pool ("daemonic processes are not
    # allowed to have children"). Use unwrap_func to invoke the inner per-uuid
    # implementation, same pattern as braking_hard / unwrap_func(stationary).
    return unwrap_func(accelerating)(
        track_candidates,
        log_dir,
        min_accel=-np.inf,
        max_accel=max_accel,
    )

@composable
@cache_manager.create_cache('braking_hard')
def braking_hard(
    track_candidates: dict,
    log_dir: Path,
    max_accel: float = -2.5,
) -> dict:
    """
    Identifies tracks that are braking hard (forward deceleration ≤ max_accel m/s²).

    Thin wrapper around accelerating(): selects frames whose forward acceleration
    falls in (-inf, max_accel]. Single-threshold criterion on one signal —
    no jerk / windowed / sustained checks — chosen because "braking hard" in
    natural language is a strength judgment, not a smoothness judgment, and
    requiring a sharp jerk onset was dropping the plateau frames of genuine
    sustained hard brakes.

    Args:
        track_candidates: The tracks to analyze (scenario dictionary).
        log_dir: Path to the directory containing scenario logs.
        max_accel: Upper bound of forward acceleration considered "hard
            braking" (m/s²). Default -2.5 calibrated against val GT
            hard-braking distribution ("vehicle braking heavily" prompts,
            n=5 tracks / 116 frames, GT accel range [-5.35, -2.51]). More
            negative values catch stronger brake events (e.g. -3.5 for
            emergency-only braking).

    Returns:
        Filtered scenario dictionary with timestamps where forward accel
        ≤ max_accel.

    Example:
        hard_braking_vehicles = braking_hard(vehicles, log_dir)
    """
    # accelerating() is @composable; calling it from inside another @composable
    # body would spawn a nested worker pool ("daemonic processes are not
    # allowed to have children"). Use unwrap_func to invoke the inner per-uuid
    # implementation, same pattern as unwrap_func(stationary)(...) elsewhere
    # in this module.
    return unwrap_func(accelerating)(
        track_candidates,
        log_dir,
        min_accel=-np.inf,
        max_accel=max_accel,
    )

@composable
@cache_manager.create_cache('reversing')
def reversing(
    track_candidates: dict,
    log_dir: Path,
    min_speed: float = 0.2) -> dict:
    """
    Returns objects that are reversing, defined as having negative forward velocity in the
    object's own body frame (x-axis). Parked objects are excluded.

    Equivalently: the city-frame velocity vector and the body-frame yaw vector
    are oriented opposite (dot product < -min_speed × |v|). Trajectory-level
    stationary tracks are dropped to suppress jitter-induced false positives.

    Args:
        track_candidates: The objects you want to filter from (scenario dictionary).
        log_dir: Path to scenario logs.
        min_speed: Minimum backward speed in m/s. Default 0.2 accommodates
            precise parking maneuvers (slow reverse ≥ 0.3 m/s) while still
            filtering annotation jitter. Aligned with driving_wrong_direction's
            min_speed convention.

    Returns:
        dict:
            A filtered scenario dictionary where:
            - Keys are track UUIDs that are reversing.
            - Values are lists of timestamps during which the object moves backward.

    Example:
        reversing_vehicles = reversing(vehicles, log_dir)
    """
    track_uuid = track_candidates

    if unwrap_func(stationary)(track_uuid, log_dir):
        return []

    velocities, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame='self')
    # Vectorized: same mask as the per-row `vel[0] < -min_speed` check.
    velocities = np.asarray(velocities)
    if len(velocities):
        mask = velocities[:, 0] < -min_speed
        reversing_timestamps = [timestamps[i] for i in np.where(mask)[0]]
    else:
        reversing_timestamps = []

    return reversing_timestamps


# Per-worker in-memory cache for the parking-row detection step.
# Inlined into in_parallel_parking but cached here so multiple per-uuid calls
# in the same worker don't re-run the heavy row scan. Key = (log_dir, params).
_PARKING_ROWS_CACHE: dict = {}

@composable
@cache_manager.create_cache('in_parallel_parking')
def in_parallel_parking(
    track_candidates: dict,
    log_dir: Path,
    # zone-membership params (subject-position vs row)
    zone_perp_max_m: float = 6.0,
    zone_along_margin_m: float = 5.0,
    # parking-row detection params (static filter + alignment clustering)
    static_speed_max_mps: float = 0.1,
    static_disp_max_m: float = 2.0,
    static_n_frames_min: int = 12,
    row_perp_max_m: float = 3.0,
    row_along_max_m: float = 70.0,
    row_yaw_align_max_deg: float = 20.0,
    row_min_count: int = 3,
) -> list:
    """
    Identifies timestamps where the subject is INSIDE a parallel-parking zone.

    Single-function pipeline:
        STEP 1 — detect parking rows from STATIC + aligned VEHICLE clusters:
            (a) iterate all VEHICLE-category tracks
            (b) keep tracks with city-frame MEDIAN speed <= static_speed_max_mps
                AND total displacement <= static_disp_max_m
                AND observation length >= static_n_frames_min
                (median makes the filter robust to tracker position jitter —
                 a truly static car may have mean speed > 80 m/s due to a single
                 frame outlier, but its median over the window stays < 0.1 m/s.)
            (c) for each unused static seed, gather neighbors within
                row_perp_max_m perp distance, ±row_along_max_m along, yaw
                parallel (or anti-parallel) within row_yaw_align_max_deg
            (d) accept cluster when size >= row_min_count, PCA-fit the line

        STEP 2 — zone membership: for each subject timestamp, project the
                 position into the row's (along, perp) frame and test
                 |perp| ≤ zone_perp_max_m  AND  |along| ≤ span/2 + along_margin.

    Subject heading is ignored — only position is tested ("did it ENTER the
    zone?"). Use scenario_and / scenario_or to combine with motion filters
    (reversing, stationary, has_velocity, …) for prompt-specific semantics.

    Timestamp safety:
        @composable wrapper (refAV/utils.py:248-251) INTERSECTS returned ts
        with the input scenario_dict's allowed ts via scenario_at_timestamps().
        So chains (reversing → in_parallel_parking, stationary → …) preserve
        the upstream filter — no timestamp leakage.

    Performance: STEP 1 is the heavy step (iterates all VEHICLE tracks). It is
    cached at module level (_PARKING_ROWS_CACHE) by (log_dir, params) so
    multiple per-uuid calls in the same worker process share one row scan.

    Args:
        track_candidates: Subjects (scenario_dict).
        log_dir: scenario logs path.
        zone_perp_max_m: half-width perpendicular to the parking row line.
            Default 6.0 m — chosen via 2D ablation (zone=2 vs zone=6): zone=6
            gives PRED HOTA-T 0.247 / GT 0.466 vs zone=2's 0.153 / 0.225.
            Wider band absorbs tracker position noise & row-line tilt error,
            so subjects close to the spot still get counted.
        zone_along_margin_m: extra extent past row endpoints, each side. Default 5.0 m.
        static_speed_max_mps: MEDIAN speed threshold (m/s). Default 0.1.
        static_disp_max_m: total displacement threshold (m). Default 2.0.
        static_n_frames_min: minimum observation length. Default 12.
        row_perp_max_m: max perp distance from candidate row line to include a
            neighbor as row member. Default 3.0 m.
        row_along_max_m: max along-axis range for cluster seeds. Default 70.0 m.
        row_yaw_align_max_deg: max heading deviation (∥ or anti-∥). Default 20°.
        row_min_count: minimum cluster size to accept a row. Default 3.

    Returns:
        list of timestamps (int) where the subject is inside any parking zone.

    Example:
        # vehicle that is reversing AND inside the parking zone
        vehicles  = get_objects_of_category(log_dir, category='VEHICLE')
        reversing_v = reversing(vehicles, log_dir)
        reverse_in_park = in_parallel_parking(reversing_v, log_dir)
    """
    track_uuid = track_candidates  # @composable convention

    # ── STEP 1: detect parking rows (cached per (log_dir, params)) ────────
    rows_key = (str(log_dir),
                static_speed_max_mps, static_disp_max_m, static_n_frames_min,
                row_perp_max_m, row_along_max_m, row_yaw_align_max_deg, row_min_count)
    rows = _PARKING_ROWS_CACHE.get(rows_key)
    if rows is None:
        # ── static-vehicle filter ────────────────────────────────────────
        static = {}
        for uuid in list(get_uuids_of_category(log_dir, 'VEHICLE')):
            pos, ts = get_nth_pos_deriv(uuid, 0, log_dir)
            if len(ts) < static_n_frames_min:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            ts_arr = np.asarray(ts, dtype=np.int64)
            x, y = pos[:, 0], pos[:, 1]
            dt = np.diff(ts_arr).astype(np.float64) / 1e9
            ds = np.hypot(np.diff(x), np.diff(y))
            if len(dt) == 0:
                continue
            v = ds / np.maximum(dt, 1e-3)
            # MEDIAN — robust to tracker jitter peaks.
            median_speed = float(np.median(v))
            disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
            if median_speed > static_speed_max_mps or disp > static_disp_max_m:
                continue
            yaw_z, _ = get_nth_yaw_deriv(uuid, 0, log_dir)
            if len(yaw_z) == 0:
                continue
            yaw_z = np.asarray(yaw_z, dtype=np.float64)
            mean_yaw = float(np.arctan2(np.mean(np.sin(yaw_z)),
                                         np.mean(np.cos(yaw_z))))
            static[uuid] = {
                'mean_x': float(np.mean(x)), 'mean_y': float(np.mean(y)),
                'mean_yaw': mean_yaw, 'n_frames': int(len(ts)),
            }

        # ── greedy alignment clustering ──────────────────────────────────
        items = list(static.items())
        yaw_align_max_rad = float(np.deg2rad(row_yaw_align_max_deg))
        rows = []
        used: set = set()
        for tid_i, s_i in items:
            if tid_i in used:
                continue
            yaw = s_i['mean_yaw']
            u_hat = np.array([np.cos(yaw), np.sin(yaw)])
            n_hat = np.array([-np.sin(yaw), np.cos(yaw)])
            x0, y0 = s_i['mean_x'], s_i['mean_y']
            cluster = [tid_i]
            for tid_j, s_j in items:
                if tid_j in used or tid_j == tid_i:
                    continue
                dx = s_j['mean_x'] - x0
                dy = s_j['mean_y'] - y0
                along = dx * u_hat[0] + dy * u_hat[1]
                perp = abs(dx * n_hat[0] + dy * n_hat[1])
                dyaw = (s_j['mean_yaw'] - yaw + np.pi) % (2 * np.pi) - np.pi
                yaw_align = min(abs(dyaw), abs(abs(dyaw) - np.pi))
                if (perp <= row_perp_max_m
                        and abs(along) <= row_along_max_m
                        and yaw_align <= yaw_align_max_rad):
                    cluster.append(tid_j)
            if len(cluster) < row_min_count:
                continue
            for t in cluster:
                used.add(t)
            xs = np.array([static[t]['mean_x'] for t in cluster], dtype=np.float64)
            ys = np.array([static[t]['mean_y'] for t in cluster], dtype=np.float64)
            cx, cy = float(xs.mean()), float(ys.mean())
            xy_c = np.column_stack([xs - cx, ys - cy])
            if len(xy_c) >= 2:
                _, _, vh = np.linalg.svd(xy_c, full_matrices=False)
                u_dir = vh[0]
                line_yaw = float(np.arctan2(u_dir[1], u_dir[0]))
            else:
                line_yaw = yaw
            along_proj = (xs - cx) * np.cos(line_yaw) + (ys - cy) * np.sin(line_yaw)
            span = float(along_proj.max() - along_proj.min())
            rows.append({
                'members': list(cluster),
                'member_xy': [(float(xs[k]), float(ys[k])) for k in range(len(cluster))],
                'line_dir': (float(np.cos(line_yaw)), float(np.sin(line_yaw))),
                'line_origin': (cx, cy),
                'yaw_rad': line_yaw,
                'span_m': span,
            })
        _PARKING_ROWS_CACHE[rows_key] = rows

    if not rows:
        return []

    # ── STEP 2: zone-membership check for subject trajectory ──────────────
    pos, ts = get_nth_pos_deriv(track_uuid, 0, log_dir)
    if len(ts) == 0:
        return []
    pos = np.asarray(pos, dtype=np.float64)
    px, py = pos[:, 0], pos[:, 1]

    in_zone_any = np.zeros(len(ts), dtype=bool)
    for r in rows:
        cx, cy = r['line_origin']
        u_x, u_y = r['line_dir']
        n_x, n_y = -u_y, u_x   # perp (left of along)
        dx = px - cx
        dy = py - cy
        along = dx * u_x + dy * u_y
        perp = np.abs(dx * n_x + dy * n_y)
        half_span = r['span_m'] / 2.0
        in_zone_any |= (perp <= zone_perp_max_m) & (np.abs(along) <= half_span + zone_along_margin_m)

    return [int(ts[i]) for i in np.where(in_zone_any)[0]]

@lru_cache(maxsize=128)
def _front_cam_fov_params_cached(log_dir_str: str):
    log_dir = Path(log_dir_str)
    try:
        ext = pd.read_feather(log_dir / 'calibration/egovehicle_SE3_sensor.feather')
        intr = pd.read_feather(log_dir / 'calibration/intrinsics.feather')
    except Exception:
        split = get_log_split(log_dir)
        base = paths.AV2_DATA_DIR / split / log_dir.name
        ext = pd.read_feather(base / 'calibration/egovehicle_SE3_sensor.feather')
        intr = pd.read_feather(base / 'calibration/intrinsics.feather')
    fc_ext = ext[ext.sensor_name == 'ring_front_center'].iloc[0]
    fc_int = intr[intr.sensor_name == 'ring_front_center'].iloc[0]
    apex_x = float(fc_ext.tx_m)
    apex_y = float(fc_ext.ty_m)
    half_angle = math.atan(float(fc_int.width_px) / 2.0 / float(fc_int.fx_px))
    return apex_x, apex_y, half_angle


def _front_cam_fov_params(log_dir):
    """Front camera (ring_front_center) FOV parameters in ego frame.

    Returns (apex_x, apex_y, half_angle_rad). half_angle is the horizontal
    half-FOV, computed from intrinsics.fx and image width as
        half_angle = atan(width / 2 / fx)
    No distortion / image projection involved — the BEV approximation is a
    pure circular sector. Uses functools.lru_cache (not cache_manager) to avoid
    multiprocessing-pool nesting when called from inside @composable workers.
    """
    return _front_cam_fov_params_cached(str(Path(log_dir)))

@cache_manager.create_cache('nth_object_in_direction')
def nth_object_in_direction(
    track_candidates,
    log_dir: Path,
    direction: Literal["forward", "backward", "left", "right"] = "forward",
    n: int = 1,
    view_point=None,
    within_distance: float = 60.0,
    lateral_thresh: float = 4.0,
    active_only: bool = True,
    min_active_speed: float = 0.0,
    lane_lateral_thresh: float = 2.0,
) -> dict:
    """
    Selects the n-th nearest object inside a directional region from the
    `view_point`, per timestamp. n=1 returns the nearest, n=2 the second
    nearest, etc. Only the single n-th object is returned per timestamp.

    Args:
        track_candidates: scenario dict (or list/uuid) — the POOL of candidates
            to rank. Output uuids are drawn from this set.
        log_dir: scenario log directory.
        direction: 'forward' uses an FOV cone modeled after ego ring_front_center
            (47° horizontal, 60 m cap). 'backward' is lane-aware: the candidate
            must be longitudinally behind the view_point AND within
            `lane_lateral_thresh` of the centerline of vp's current lane or
            up to 2 best-direction-similarity predecessor lanes. 'left'/'right'
            use a simple rectangular region in view_point body frame.
        n: ordinal rank (1-based).
        view_point: scenario dict, uuid string, or None. Defines the perspective
            (origin and forward axis). Defaults to ego.
        within_distance: range cap (m) for non-forward directions.
        lateral_thresh: lateral offset cap (m) for left/right directions
            (NOT used by backward — backward uses lane_lateral_thresh).
        active_only: when True (default), the ranking pool is filtered by
            scenario-level active() — drops candidates that never moved
            >3 m total (pure parked vehicles). When `min_active_speed` > 0,
            an additional per-timestamp speed gate is applied on top. Pass
            False to disable all active filtering.
        min_active_speed: per-frame ground speed threshold (m/s) for the
            optional gate on top of scenario-level active(). Default 0.0 —
            disabled because val-eval showed that "stopped-for-action" GT
            vehicles (e.g., a vehicle stopped while a pedestrian crosses in
            front) have full-trajectory max speeds (~0.1–0.4 m/s) that
            overlap with parked-curb FPs, so per-frame speed alone cannot
            separate them and the gate silently drops GT. Set to >0 only
            when the prompt guarantees GT vehicles are actively moving
            (e.g., "braking", "changing lanes", "turning") AND parked-
            vehicle FPs are the dominant error mode. Lane-aware filtering
            is the more robust long-term fix for the stopped-vs-parked
            ambiguity.
        lane_lateral_thresh: max perpendicular distance (m) from a backward
            lane chain centerline for a candidate to qualify. Default 2.0 —
            restricts the rank pool to ego/vp's own lane (lane half-width
            ≈ 1.75 m). Increase to capture adjacent-lane vehicles: ~3.5 for
            ego + 1 neighbor, ~5.0 for full neighbor lane. Only used when
            direction='backward'.

    Returns:
        dict: flat scenario dict mapping the n-th uuid to its timestamps. The view_point is NOT
        included as a key (it is the perspective, not the answer).

    Example:
        # default view_point = ego, scenario-level active() only
        second_ahead = nth_object_in_direction(vehicles, log_dir, direction='forward', n=2)
        # explicit non-ego view_point
        second_ahead_of_X = nth_object_in_direction(
            vehicles, log_dir, direction='forward', n=2, view_point=other_vehicle_dict)
        # opt out of all active filtering (rank parked + active together)
        second_any = nth_object_in_direction(
            vehicles, log_dir, direction='forward', n=2, active_only=False)
        # opt-in to per-frame speed gate (use only when GT is guaranteed moving)
        second_strict = nth_object_in_direction(
            vehicles, log_dir, direction='forward', n=2, min_active_speed=0.1)
    """

    candidate_ts_map = _candidate_uuid_timestamps(track_candidates)
    vp_uuid = _resolve_view_point(view_point, log_dir)
    ego_uuid = get_ego_uuid(log_dir)
    is_ego_view = (vp_uuid == ego_uuid)
    get_object_raw = unwrap_func(get_object)

    # Active-only pruning (scenario level): drop candidates that are
    # stationary over the whole scenario (parked vehicles, max displacement
    # <3 m). Cheap scenario-level prune before the per-ts ranking loop.
    # View_point is exempt — it is the perspective, not a rankable candidate.
    if active_only and candidate_ts_map:
        try:
            active_scenario = active(track_candidates, log_dir)
        except Exception:
            active_scenario = None
        if active_scenario is not None:
            active_uuid_set: set = set()
            for k, v in active_scenario.items():
                if isinstance(v, dict):
                    active_uuid_set.update(v.keys())
                else:
                    active_uuid_set.add(k)
            # Keep vp_uuid (will be filtered below) + active candidates only.
            candidate_ts_map = {
                u: ts for u, ts in candidate_ts_map.items()
                if u == vp_uuid or u in active_uuid_set
            }

    # Optional per-timestamp speed gate (opt-in via min_active_speed > 0).
    # Disabled by default (0.0) because per-frame speed cannot separate
    # "stopped-for-action" GT (e.g., waiting for ped to cross) from parked-
    # curb FPs — their full-trajectory speed distributions overlap. Pre-
    # compute per-uuid {ts: speed} so the per-ts loop can do an O(1) check.
    speed_by_uuid_ts: dict = {}
    apply_speed_gate = active_only and min_active_speed > 0.0
    if apply_speed_gate:
        for uuid in candidate_ts_map.keys():
            if uuid == vp_uuid:
                continue
            try:
                vel, vts = get_nth_pos_deriv(uuid, 1, log_dir)
            except Exception:
                continue
            vel = np.asarray(vel)
            if vel.ndim != 2 or vel.shape[0] == 0:
                continue
            spd = np.linalg.norm(vel[:, :2], axis=1)
            speed_by_uuid_ts[uuid] = {int(t): float(s) for t, s in zip(vts, spd)}

    # ----- Region geometry (constants used by per-ts classification) -----
    if direction == 'forward':
        apex_x_ego, apex_y_ego, half_angle = _front_cam_fov_params(log_dir)
        # In view_point body frame, apex is camera position when ego, else origin.
        apex_x_vp = apex_x_ego if is_ego_view else 0.0
        apex_y_vp = apex_y_ego if is_ego_view else 0.0
    else:
        if is_ego_view:
            # AV2 ego: rear axle = origin, ego cuboid center at (1.422, 0).
            # Front bumper x = +3.8605, rear bumper x = -1.0165.
            track_width = 1.0
            track_front = 4.877/2 + 1.422
            track_back = 1.422 - 4.877/2
        else:
            vp_cuboid = get_cuboid_from_uuid(vp_uuid, log_dir)
            track_width = vp_cuboid.width_m / 2
            track_front = vp_cuboid.length_m / 2
            track_back = -vp_cuboid.length_m / 2

    # Lane-aware backward pre-compute: vp lane chain centerlines (city frame)
    # + per-ts ego_pose (used to lift candidate ego-frame center → city frame
    # for the lateral check). depth=2 follows the natural-language reach of
    # "the vehicle two cars behind" (vp's lane + 2 predecessors).
    backward_centerlines_by_ts: dict = {}
    backward_ego_pose_by_ts: dict = {}
    if direction == 'backward':
        avm = get_map(log_dir)
        lane_segments_map = avm.vector_lane_segments
        vp_lane_by_ts = get_scenario_lanes(vp_uuid, log_dir, avm=avm)
        ego_poses = get_ego_SE3(log_dir)
        for ts_key, ls in (vp_lane_by_ts or {}).items():
            if ls is None:
                continue
            chain = _backward_lane_chain(ls, lane_segments_map, avm, depth=2)
            if not chain:
                continue
            cls = [avm.get_lane_segment_centerline(l.id)[:, :2] for l in chain]
            backward_centerlines_by_ts[int(ts_key)] = cls
            backward_ego_pose_by_ts[int(ts_key)] = ego_poses[int(ts_key)]

    # Resolve per-uuid allowed timestamps (None → that uuid's full ts set).
    candidate_allowed_ts: dict = {}
    for uuid, allowed in candidate_ts_map.items():
        if uuid == vp_uuid:
            continue
        if allowed is None:
            allowed = {int(t) for t in get_object_raw(uuid, log_dir)}
        if allowed:
            candidate_allowed_ts[uuid] = allowed

    # Inverted index: ts → list of candidate uuids active at that ts.
    ts_to_candidates: dict = {}
    for uuid, ts_set in candidate_allowed_ts.items():
        for ts in ts_set:
            ts_to_candidates.setdefault(int(ts), []).append(uuid)

    # Pre-compute view_point pose in ego frame (only needed for non-ego forward case).
    vp_pose_by_ts: dict = {}
    if direction == 'forward' and not is_ego_view:
        vp_pos_ego, vp_pos_ts = get_nth_pos_deriv(
            vp_uuid, 0, log_dir, coordinate_frame=ego_uuid)
        vp_yaw_arr, _ = get_nth_yaw_deriv(
            vp_uuid, 0, log_dir, coordinate_frame=ego_uuid, in_degrees=False)
        for i, t in enumerate(vp_pos_ts):
            vp_pose_by_ts[int(t)] = (
                float(vp_pos_ego[i, 0]),
                float(vp_pos_ego[i, 1]),
                float(vp_yaw_arr[i]),
            )

    result: dict = {}

    # ── Per-timestamp loop (Steps 1–4 per user spec) ──────────────────────────
    for ts in sorted(ts_to_candidates.keys()):
        candidate_list = ts_to_candidates[ts]

        # Pose of view_point at this ts (only needed for non-ego forward).
        if direction == 'forward' and not is_ego_view:
            if ts not in vp_pose_by_ts:
                continue
            vx, vy, vyaw = vp_pose_by_ts[ts]
            cos_y, sin_y = np.cos(-vyaw), np.sin(-vyaw)
        else:
            vx = vy = 0.0
            cos_y, sin_y = 1.0, 0.0

        # ── Step 1: classify candidates by region (cuboid center in view_point body frame).
        # ── Step 2: compute distance from view_point center for each in-region candidate.
        in_region_distances = []
        for uuid in candidate_list:
            # Optional per-frame speed gate (only when min_active_speed > 0):
            # skip candidates effectively stopped at this ts. Cheap O(1)
            # lookup before the cuboid + region work.
            if apply_speed_gate and uuid != vp_uuid:
                if speed_by_uuid_ts.get(uuid, {}).get(int(ts), 0.0) < min_active_speed:
                    continue
            cub = get_cuboid_from_uuid(uuid, log_dir, timestamp=ts)
            if cub is None:
                continue
            v = cub.vertices_m
            cx_e = float(v[:, 0].mean())
            cy_e = float(v[:, 1].mean())

            if is_ego_view:
                cx_vp, cy_vp = cx_e, cy_e
            else:
                rx, ry = cx_e - vx, cy_e - vy
                cx_vp = rx * cos_y - ry * sin_y
                cy_vp = rx * sin_y + ry * cos_y

            if direction == 'forward':
                dx = cx_vp - apex_x_vp
                dy = cy_vp - apex_y_vp
                if dx <= 0:
                    continue
                if abs(np.arctan2(dy, dx)) > half_angle:
                    continue
                if np.sqrt(dx * dx + dy * dy) > within_distance:
                    continue
            elif direction == 'left':
                if not (cy_vp > track_width and
                        (track_back - lateral_thresh < cx_vp < track_front + lateral_thresh)):
                    continue
            elif direction == 'right':
                if not (cy_vp < -track_width and
                        (track_back - lateral_thresh < cx_vp < track_front + lateral_thresh)):
                    continue
            elif direction == 'backward':
                # Lane-aware backward: (i) longitudinally behind vp in vp's
                # body frame AND (ii) within lane_lateral_thresh of vp's
                # backward lane-chain centerline (vp's lane + 2 predecessors)
                # in city frame. Replaces the simple body-frame rectangle so
                # curved roads / lane changes do not drop GT.
                if cx_vp >= track_back:
                    continue
                ts_key = int(ts)
                centerlines = backward_centerlines_by_ts.get(ts_key)
                if not centerlines:
                    continue
                ego_pose_ts = backward_ego_pose_by_ts.get(ts_key)
                if ego_pose_ts is None:
                    continue
                # Candidate center (ego frame) → city frame.
                cand_ego = np.array([cx_e, cy_e, 0.0], dtype=np.float64)
                cand_city = ego_pose_ts.rotation @ cand_ego + ego_pose_ts.translation
                cand_xy_city = cand_city[:2]
                min_lat = min(
                    _point_to_polyline_lateral_xy(cand_xy_city, cl)
                    for cl in centerlines
                )
                if min_lat > lane_lateral_thresh:
                    continue
            else:
                continue

            distance = float(np.sqrt(cx_vp * cx_vp + cy_vp * cy_vp))
            if direction != 'forward' and distance > within_distance:
                continue
            in_region_distances.append((uuid, distance))

        # ── Step 3: pick the n-th nearest at this ts (skip if fewer than n in region).
        if len(in_region_distances) < n:
            continue
        in_region_distances.sort(key=lambda row: row[1])
        nth_uuid, _ = in_region_distances[n - 1]
        result.setdefault(nth_uuid, []).append(ts)

    # Determinism: sort ts lists.
    for k in result:
        result[k].sort()
    return result


def _resolve_view_point(view_point, log_dir):
    """Normalize view_point arg to a single uuid string.

    None → ego_uuid. Dict → first key. Str → as-is.
    """
    if view_point is None:
        return get_ego_uuid(log_dir)
    if isinstance(view_point, dict):
        keys = list(view_point.keys())
        if not keys:
            return get_ego_uuid(log_dir)
        return keys[0]
    return view_point


def _candidate_uuid_timestamps(track_candidates):
    """Return {uuid: allowed_ts_set | None}.

    None means 'no per-uuid filter' (input was a bare list/str of uuids).
    For dict inputs, recursively flattens nested scenario dicts via
    get_scenario_timestamps so timestamp filtering from upstream
    (e.g. scenario_not(stationary)) is preserved.
    """
    if isinstance(track_candidates, dict):
        out: dict = {}
        for uuid, child in track_candidates.items():
            ts_list = get_scenario_timestamps(child) if isinstance(child, dict) else list(child or [])
            out[uuid] = {int(t) for t in ts_list}
        return out
    if isinstance(track_candidates, str):
        return {track_candidates: None}
    return {u: None for u in track_candidates}


def _backward_lane_chain(start_ls, lane_segments_map, avm, depth=2):
    """vp's current lane + up to `depth` best-direction-similarity predecessors.

    Skips intersection lanes unless they are 'straight' (mirrors the heuristic
    in refAV.utils.get_semantic_lane). Used by nth_object_in_direction's
    lane-aware backward filter.
    """
    if start_ls is None:
        return []
    if start_ls.is_intersection and get_turn_direction(start_ls) != 'straight':
        return [start_ls]
    chain = [start_ls]
    chain_ids = {start_ls.id}
    current = start_ls
    cur_dir = get_lane_orientation(current, avm)
    for _ in range(depth):
        if not current.predecessors:
            break
        best, best_sim = None, -np.inf
        for pid in current.predecessors:
            if pid not in lane_segments_map or pid in chain_ids:
                continue
            pls = lane_segments_map[pid]
            if pls.is_intersection and get_turn_direction(pls) != 'straight':
                continue
            pdir = get_lane_orientation(pls, avm)
            denom = float(np.linalg.norm(cur_dir) * np.linalg.norm(pdir) + 1e-9)
            sim = float(np.dot(cur_dir, pdir) / denom)
            if sim > best_sim:
                best_sim, best = sim, pls
        if best is None:
            break
        chain.append(best)
        chain_ids.add(best.id)
        current = best
        cur_dir = get_lane_orientation(current, avm)
    return chain


def _point_to_polyline_lateral_xy(point_xy, polyline_xy):
    """Min perpendicular distance from a 2D point to a polyline (Nx2).

    Returns inf if polyline is empty. Single-point polyline → point-to-point.
    """
    if polyline_xy.shape[0] == 0:
        return float('inf')
    if polyline_xy.shape[0] == 1:
        return float(np.linalg.norm(point_xy - polyline_xy[0]))
    segs_a = polyline_xy[:-1]
    segs_b = polyline_xy[1:]
    seg = segs_b - segs_a
    seg_len2 = (seg ** 2).sum(axis=1) + 1e-9
    rel = point_xy - segs_a
    t = np.clip((rel * seg).sum(axis=1) / seg_len2, 0.0, 1.0)
    proj = segs_a + t[:, None] * seg
    return float(np.linalg.norm(point_xy - proj, axis=1).min())

# Yellow-dashed boundary markings used by a two-way left-turn lane (TWLTL).
_YELLOW_DASHED_MARKS = {
    'DASH_SOLID_YELLOW', 'SOLID_DASH_YELLOW', 'DOUBLE_DASH_YELLOW', 'DASHED_YELLOW',
}

def _is_twltl(ls) -> bool:
    """Center two-way left-turn lane signature in AV2: yellow-dashed markings on
    BOTH boundaries AND a dead-end (None) neighbor on the opposing-traffic side.
    Such a lane is referenced from only one direction's flow, so it is excluded
    from the lane count."""
    return (ls.left_mark_type.value in _YELLOW_DASHED_MARKS
            and ls.right_mark_type.value in _YELLOW_DASHED_MARKS
            and (ls.left_neighbor_id is None or ls.right_neighbor_id is None))


def _lateral_lane_counts(ls, avm, lane_types):
    """Count parallel lanes at the cross-section of ``ls``.

    Walks the left then right neighbor chains. Same-vs-opposite direction is
    decided by the reciprocal-neighbor pattern (AV2 neighbor links are always
    mutual): advancing side and the back-pointer side having opposite names
    (left<->right) means same direction; identical names (left-left /
    right-right) means the centerline was crossed -> opposite direction.

    TWLTL lanes are skipped (count not incremented, walk stops on that side).
    Lanes whose lane_type is not in ``lane_types`` are not counted but the walk
    continues past them.

    Returns (n_same, n_opposite); n_same includes ``ls`` itself when it matches
    the lane_types filter.
    """
    LS = avm.vector_lane_segments

    def _counts_for_lane_type(seg):
        return lane_types is None or seg.lane_type.value in lane_types

    n_same = 1 if _counts_for_lane_type(ls) else 0
    n_opp = 0

    for start_side in ('left', 'right'):
        current = ls
        advance = start_side
        orientation = 1  # +1 same direction as ls, -1 opposite
        visited = {ls.id}

        while True:
            nid = current.left_neighbor_id if advance == 'left' else current.right_neighbor_id
            if nid is None or nid in visited:
                break
            nxt = LS.get(nid)
            if nxt is None:
                break
            visited.add(nid)

            # Which side of nxt points back to current?
            if nxt.left_neighbor_id == current.id:
                back = 'left'
            elif nxt.right_neighbor_id == current.id:
                back = 'right'
            else:
                break  # non-reciprocal (not expected in AV2) — defensive stop

            # Opposite-named sides -> same direction; identical -> centerline crossed.
            if advance == back:
                orientation *= -1

            # TWLTL referenced from only one direction -> exclude and stop this side.
            if _is_twltl(nxt):
                break

            if _counts_for_lane_type(nxt):
                if orientation == 1:
                    n_same += 1
                else:
                    n_opp += 1

            # Continue outward: advance on the side opposite to the back-pointer.
            advance = 'right' if back == 'left' else 'left'
            current = nxt

    return n_same, n_opp


def _reference_lane_for_intersection(items, i, avm):
    """Resolve a non-intersection reference road lane for an intersection lane.

    ``items`` is the track's time-sorted [(ts, LaneSegment|None), ...] sequence;
    ``i`` is the index of an intersection lane. Lane-count machinery is meant for
    straight road cross-sections, but an intersection lane is a single turn path
    with sparse/meaningless lateral neighbors, so we substitute the road the
    track was actually on around the junction:

      1. before: scanning backward from i, the last non-intersection lane the
         track occupied (the approach road) — preferred.
      2. after: scanning forward from i, the first non-intersection lane (the
         exit road) — used only when there is no before lane.
      3. graph fallback: a non-intersection predecessor/successor of the
         intersection lane (1-hop, then 2-hop) — used when the track was only
         observed inside the junction.

    Returns the reference LaneSegment, or None if none can be resolved.
    """
    # 1. before (approach) — last non-intersection lane the track was on.
    for j in range(i - 1, -1, -1):
        lj = items[j][1]
        if lj is not None and not lj.is_intersection:
            return lj
    # 2. after (exit) — first non-intersection lane the track moves onto.
    for j in range(i + 1, len(items)):
        lj = items[j][1]
        if lj is not None and not lj.is_intersection:
            return lj
    # 3. graph fallback: non-intersection pred/succ of the intersection lane.
    cur = items[i][1]
    LS = avm.vector_lane_segments
    hop1 = list(cur.predecessors) + list(cur.successors)
    for nid in hop1:
        n = LS.get(nid)
        if n is not None and not n.is_intersection:
            return n
    for nid in hop1:
        n = LS.get(nid)
        if n is None:
            continue
        for nid2 in list(n.predecessors) + list(n.successors):
            n2 = LS.get(nid2)
            if n2 is not None and not n2.is_intersection:
                return n2
    return None


@composable
@cache_manager.create_cache('on_road_with_n_lanes')
def on_road_with_n_lanes(
    track_candidates: dict,
    log_dir: Path,
    n_lanes: int = None,
    lane_scope: Literal['same_direction', 'per_direction', 'total'] = 'same_direction',
    one_way: bool = False,
    lane_types: tuple = ('VEHICLE', 'BUS'),
) -> dict:
    """
    Identifies timestamps when a track is on a road with a given lane structure.

    At each timestamp the track's current lane is resolved, then its parallel
    lanes are counted laterally: n_same (lanes in the track's travel direction,
    including its own) and n_opposite (lanes in the opposing direction). Same-
    vs-opposite is determined from AV2's reciprocal neighbor links rather than a
    geometric gap. Center two-way left-turn lanes (TWLTL) are excluded, and only
    lanes whose type is in ``lane_types`` are counted.

    The ``lane_scope`` (what n_lanes counts) and ``one_way`` (no oncoming lanes)
    conditions are independent and combine with AND.

    Args:
        track_candidates: Tracks to analyze (scenario dictionary).
        log_dir: Path to scenario logs.
        n_lanes: The lane count to require. None means do not check the count —
            only the ``one_way`` / symmetric-road structural condition applies.
        lane_scope: What ``n_lanes`` counts.
            'same_direction' (default): lanes in the track's travel direction
                (n_same == n_lanes).
            'per_direction': the same count in BOTH directions
                (n_same == n_lanes AND n_opposite == n_lanes); when n_lanes is
                None, requires a symmetric road (n_same == n_opposite > 0).
            'total': all lanes on the road (n_same + n_opposite == n_lanes).
        one_way: When True, additionally require no oncoming lanes
            (n_opposite == 0). Maps directly to 'one-way road' prompts.
        lane_types: Lane types counted toward the totals. Default ('VEHICLE',
            'BUS') — BIKE lanes and other types are walked past but not counted.
            None counts every lane type.

    Intersection handling: intersection lane neighbor links are sparse and do
    not represent standard parallel lanes, so at an intersection timestamp the
    road structure is taken from the track's APPROACH road — the last
    non-intersection lane the track occupied before entering the junction
    (falling back to the exit road, then to a graph predecessor/successor).
    Timestamps where no reference road can be resolved (e.g. the track is only
    observed inside the junction) are dropped.

    Returns:
        Sorted list of timestamps where the lane-structure condition holds.

    Example:
        ego = get_objects_of_category(log_dir, category='EGO_VEHICLE')
        # 'three-lane one-way street'
        on_3lane_oneway = on_road_with_n_lanes(ego, log_dir, n_lanes=3, one_way=True)
        # 'two lanes per direction'
        on_2_per_dir = on_road_with_n_lanes(ego, log_dir, n_lanes=2, lane_scope='per_direction')
        # 'one way road' (count unspecified)
        on_oneway = on_road_with_n_lanes(ego, log_dir, one_way=True)
    """
    track_uuid = track_candidates
    avm = get_map(log_dir)
    scenario_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)

    result_ts = []
    count_cache: dict = {}  # ls.id -> (n_same, n_opp)
    items = sorted(scenario_lanes.items())  # time-ordered [(ts, lane|None), ...]
    for i, (ts, ls) in enumerate(items):
        if ls is None:
            continue

        if ls.is_intersection:
            # Substitute the track's approach road (before > after > graph).
            ref = _reference_lane_for_intersection(items, i, avm)
            if ref is None:
                continue
        else:
            ref = ls

        cached = count_cache.get(ref.id)
        if cached is None:
            cached = _lateral_lane_counts(ref, avm, lane_types)
            count_cache[ref.id] = cached
        n_same, n_opp = cached

        # Count condition (skipped when n_lanes is None, unless per_direction
        # wants the symmetric-road shortcut).
        if n_lanes is not None:
            if lane_scope == 'same_direction':
                ok = (n_same == n_lanes)
            elif lane_scope == 'per_direction':
                ok = (n_same == n_lanes and n_opp == n_lanes)
            elif lane_scope == 'total':
                ok = (n_same + n_opp == n_lanes)
            else:
                ok = False
        elif lane_scope == 'per_direction':
            ok = (n_same == n_opp and n_same > 0)  # symmetric road, count unspecified
        else:
            ok = True  # no count constraint

        # One-way condition (independent, ANDed).
        if one_way:
            ok = ok and (n_opp == 0)

        if ok:
            result_ts.append(int(ts))

    return sorted(result_ts)

# ---------------------------------------------------------------------------
# VLM-based visual filters (Qwen3.6-35B-A3B via vLLM). Two atoms:
#   get_visual_actor    (LAYER2_ACTOR)    -- identify what an actor visually IS
#   get_visual_behavior (LAYER3_BEHAVIOR) -- visible behavior / appearance / state
# Both are filter-only (subset of candidates, timestamps preserved) and share one
# Qwen call (_visual_filter, defined in utils.py); only the system/user prompt
# differs. Endpoints come from REFAV_VLM_ENDPOINTS, model from REFAV_VLM_MODEL.
# ---------------------------------------------------------------------------

def get_visual_actor(track_candidates: dict, log_dir: Path, description: str) -> dict:
    """
    Identify an ACTOR by what it visually IS, using a vision-language model (VLM) on each
    track's best camera crop. Returns the subset of track_candidates whose crop clearly shows
    the described actor type. FILTER-only: it never adds tracks, so the candidate set must
    already contain the target -- gather BROADLY first.

    Use this for an actor whose type is a recognizable look that AV2 does NOT label as its own
    category (so get_objects_of_category alone cannot find it):
        police car, ambulance, fire truck, tow truck, taxi, food truck, cement mixer,
        excavator, forklift, utility vehicle (UTV), construction worker, traffic officer, child, ...
    The description is a visual noun phrase and MAY carry a salient whole-object COLOR
    ('a white van', 'a red car') -- color is part of the actor's identity here; fold the
    color into the description (there is no separate color atom).

    Rule of thumb: if the actor's name is one of the 30 AV2 categories, use get_objects_of_category.
    If it is NOT, gather a BROAD candidate bucket and let the VLM filter -- do NOT narrow to the
    "obvious" class. The tracker mislabels subtypes under coarse classes (a forklift may land in
    WHEELED_DEVICE, LARGE_VEHICLE, or even REGULAR_VEHICLE), so any per-subtype class guess risks
    dropping the target before the VLM ever sees it. Pick the bucket by person-vs-vehicle:
        vehicle-type actor (police car, ambulance, food truck, forklift, cement mixer, excavator):
            scenario_or([get_objects_of_category(log_dir, category='VEHICLE'),
                         get_objects_of_category(log_dir, category='WHEELED_DEVICE')])
            -- 'VEHICLE' is a superclass; add WHEELED_DEVICE, which it does NOT cover, for
               forklift / UTV / excavator.
        person-type actor (construction worker, traffic officer, child):
            get_objects_of_category(log_dir, category='ANY')
            -- there is no PERSON superclass, so 'ANY' is the simplest safe gather (the VLM drops
               non-people).
        unsure / could be either: get_objects_of_category(log_dir, category='ANY').

    Do NOT use for:
      - a transient state, action, or added object on the actor (covered by a tarp,
        carrying a bicycle, on a ladder, exiting a vehicle) -> get_visual_behavior.
      - anything derivable from boxes or trajectory (position, speed, turning) -> kinematic/relational atomics.
      - environmental context around the actor (under a canopy, in a school zone) -> context-layer atomics.

    Args:
        track_candidates: Broad candidate scenario dict to filter (gather a whole bucket first).
        log_dir: Path to scenario logs.
        description: Short visual noun phrase for the actor type, e.g. 'a forklift', 'a police car'.

    Returns:
        dict: Filtered scenario dict (subset of track_candidates), timestamps preserved.

    Example:
        # forklift -- the tracker may label it WHEELED_DEVICE / LARGE_VEHICLE / REGULAR_VEHICLE,
        # so gather the whole vehicle bucket and let the VLM decide
        candidates = scenario_or([get_objects_of_category(log_dir, category='VEHICLE'),
                                  get_objects_of_category(log_dir, category='WHEELED_DEVICE')])
        forklifts = get_visual_actor(candidates, log_dir, 'a forklift')
        output_scenario(forklifts, description, log_dir, output_dir)
    """
    return _visual_filter(track_candidates, log_dir, description, "actor")

def get_visual_behavior(track_candidates: dict, log_dir: Path, description: str) -> dict:
    """
    Filter tracks by a VISIBLE behavior, appearance, or static state of the actor, using a
    vision-language model (VLM) on each track's best camera crop. Returns the subset of
    track_candidates whose crop clearly matches the description. FILTER-only.

    Apply this AFTER you already have the actor set (e.g. pedestrians, vehicles). The evidence
    must be confirmable from a SINGLE still crop of the actor itself, i.e. (1) visible in the
    actor's own padded crop and (2) salient -- not a tiny detail or a multi-frame motion.

    DO use for:
      - a visible action captured in one frame: 'a person exiting a vehicle', 'a person on a ladder',
        'a person pushing a shopping cart'.
      - a whole-object appearance / static state: 'a car covered by a tarp', 'a vehicle carrying a bicycle'.

    Do NOT use for:
      - TEMPORAL / motion behavior: 'turning', 'braking', 'accelerating', 'merging', 'flashing lights'
        -> motion / relational atomics (a single frame has no motion).
      - a tiny LOCALIZED cue: 'turned wheels', 'reverse lights on', 'turn signal on'
        -> geometry / state atomics (the VLM crop misses small details).
      - the actor's environmental surroundings: 'under a construction canopy', 'in a school zone'
        -> context-layer atomics (the surroundings fall outside the tight actor crop).

    Args:
        track_candidates: Actor scenario dict to filter (gather the actor first).
        log_dir: Path to scenario logs.
        description: Short phrase for the visible behavior / appearance,
            e.g. 'a person exiting a vehicle', 'covered by a tarp'.

    Returns:
        dict: Filtered scenario dict (subset of track_candidates), timestamps preserved.

    Example:
        peds = get_objects_of_category(log_dir, category='PEDESTRIAN')
        exiting = get_visual_behavior(peds, log_dir, 'a person exiting a vehicle')
        output_scenario(exiting, description, log_dir, output_dir)
    """
    return _visual_filter(track_candidates, log_dir, description, "behavior")


# Output Function
def output_scenario(
    scenario:dict,
    description:str,
    log_dir:Path,
    output_dir:Path,
    visualize:bool=False,
    **visualization_kwargs):
    """
    Outputs a file containing the predictions in an evaluation-ready format. Do not provide any visualization kwargs. 
    """
    still_positive = post_process_scenario(scenario, log_dir)
    if not still_positive:
        print('Scenario identification flipped from positive to negative after filtering!')

    Path(output_dir/log_dir.name).mkdir(parents=True, exist_ok=True)
    create_mining_pkl(description, scenario, log_dir, output_dir)

    if visualize:
        # PyVista and VTK can be a headache to set up on your machine. If this is the case,
        # set visualization to false
        from refAV.visualization import visualize_scenario

        log_scenario_visualization_path = Path(output_dir/log_dir.name/'scenario visualizations')
        log_scenario_visualization_path.mkdir(exist_ok=True)

        for file in log_scenario_visualization_path.iterdir():
            if file.is_file() and file.stem.split(sep='_')[0] == description:
                file.unlink()

        visualize_scenario(scenario, log_dir, log_scenario_visualization_path, description=description, **visualization_kwargs)

    


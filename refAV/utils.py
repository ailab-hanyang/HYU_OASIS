import math as _math
import numpy as np
import os
from pathlib import Path
from typing import Union, Callable, Any, Literal
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.ndimage
from scipy.spatial.transform import Rotation
from copy import deepcopy
from functools import wraps, lru_cache
import scipy
import json
import pickle
from tqdm import tqdm
from transformers import pipeline
from collections import OrderedDict
from PIL import Image
import base64 as _b64, io as _io, urllib.request as _urlreq
from concurrent.futures import ThreadPoolExecutor as _ThreadPool
from PIL import ImageFile as _ImageFile
_ImageFile.LOAD_TRUNCATED_IMAGES = True
from shapely.geometry import Polygon as _ShPolygon
from shapely.ops import unary_union as _sh_unary_union, polylabel as _sh_polylabel

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import StereoCameras
from av2.structures.cuboid import Cuboid, CuboidList
from av2.geometry.geometry import quat_to_mat
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.geometry.se3 import SE3
from av2.utils.io import read_feather as _read_feather_av2, read_city_SE3_ego
from av2.utils.synchronization_database import SynchronizationDB
from av2.evaluation.tracking.utils import save, load
from av2.datasets.sensor.splits import TEST, TRAIN, VAL
import refAV.paths as paths


class CacheManager:
    def __init__(self):
        self.caches = {}
        self.stats = {}
        self.num_processes = max(int(0.9 * os.cpu_count()), 1)
        self.semantic_lane_cache = None
        self.road_side_cache = None
        self.color_cache = None
        self.crop_embedding_cache = None

        # Global caches (tracker-independent, persist across log switches)
        self._global_semantic_lane_caches = {}  # log_id -> data
        self._global_road_side_caches = {}      # log_id -> data

    def set_num_processes(self, num):
        self.num_processes = max(min(os.cpu_count() - 1, num), 1)

    def make_hashable(self, obj):
        if isinstance(obj, (list, tuple, set)):
            return tuple(self.make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return tuple(obj.flatten())
        elif isinstance(obj, ArgoverseStaticMap):
            return obj.log_id
        elif isinstance(obj, LaneSegment):
            return obj.id
        elif isinstance(obj, Cuboid):
            return obj.track_uuid
        else:
            return obj

    def create_cache(self, name, maxsize=512):
        if name not in self.caches:
            self.caches[name] = OrderedDict()
            self.stats[name] = {'hits': 0, 'misses': 0}
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = (
                    self.make_hashable(args),
                    self.make_hashable(kwargs)
                )
                
                cache:OrderedDict = self.caches[name]
                
                if key in cache:
                    cache.move_to_end(key)
                    self.stats[name]['hits'] += 1
                    return cache[key]
                
                result = func(*args, **kwargs)
                self.stats[name]['misses'] += 1

                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                    
                return result
            
            wrapper.clear_cache = lambda: self.caches[name].clear()
            wrapper.cache_info = lambda: {
                'name': name,
                'current_size': len(self.caches[name]),
                'maxsize': maxsize
            }
            
            return wrapper
        return decorator
    
    def clear_all(self):
        for cache in self.caches.values():
            cache.clear()
    
    def info(self):
        return {name: len(cache) for name, cache in self.caches.items()}
    
    def get_stats(self, name=None):
        if name:
            stats = self.stats[name]
            total = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total if total > 0 else 0
            return {
                'name': name,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'hit_rate': f"{hit_rate:.2%}",
                'cache_size': len(self.caches[name])
            }
        return {
            name: self.get_stats(name) for name in self.stats
        }

    def load_custom_caches(self, log_dir: Path):
        """Load per-log caches.

        Semantic_lane_cache and road_side_cache are tracker-independent, loaded
        from GLOBAL_CACHE_PATH/{log_id}/ and kept in memory across log switches.
        Color_cache is tracker-dependent, loaded from {log_dir}/cache/.
        """
        cache_dir = log_dir / 'cache'
        log_id = log_dir.name
        global_cache_dir = paths.GLOBAL_CACHE_PATH / log_id
        self.current_log_dir = log_dir

        # Tracker-independent: reuse in-memory copy if already loaded
        self.semantic_lane_cache = self._global_semantic_lane_caches.get(log_id)
        self.road_side_cache = self._global_road_side_caches.get(log_id)

        if self.semantic_lane_cache is None:
            try:
                with open(global_cache_dir / 'semantic_lane_cache.json', 'r') as file:
                    self.semantic_lane_cache = json.load(file)
                    self._global_semantic_lane_caches[log_id] = self.semantic_lane_cache
            except:
                pass

        if self.road_side_cache is None:
            try:
                with open(global_cache_dir / 'road_side_cache.json', 'r') as file:
                    self.road_side_cache = json.load(file)
                    self._global_road_side_caches[log_id] = self.road_side_cache
            except:
                pass

        self.color_cache = None
        try:
            with open(cache_dir / 'color_cache.json', 'r') as file:
                self.color_cache = json.load(file)
        except:
            pass

        # Tracker-dependent: SigLIP2 crop embeddings {uuid: vec(D) fp16}
        self.crop_embedding_cache = None
        try:
            z = np.load(cache_dir / 'crop_embeddings.npz', allow_pickle=False)
            self.crop_embedding_cache = dict(zip(z['uuids'].tolist(), z['embs']))
        except:
            pass

cache_manager = CacheManager()


@cache_manager.create_cache('read_feather')
def read_feather(path):
    """Cached wrapper around av2.utils.io.read_feather.

    Same call signature; same return value. Cached by absolute path string
    (CacheManager.make_hashable casts Path -> str). Safe because every callsite
    in utils.py treats the returned DataFrame as read-only (filtering, .to_numpy(),
    .unique()).
    """
    return _read_feather_av2(path)


class EasyDataLoader(AV2SensorDataLoader):
    """Dataloader to load both NuScenes and AV2 data given only a log_id"""

    def __init__(self, log_dir):

        dataset = get_dataset(log_dir)
        split = get_log_split(log_dir)

        if dataset == 'AV2':
            data_dir = paths.AV2_DATA_DIR / split 
            labels_dir = log_dir.parent
        elif dataset == 'NUSCENES':
            data_dir = paths.NUSCENES_AV2_DATA_DIR / split
            labels_dir = log_dir.parent

        self._data_dir = data_dir
        self._labels_dir = labels_dir
        self._sdb = SynchronizationDB(str(data_dir), collect_single_log_id=log_dir.name)
        self._sdb.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF = 100E6 # 100ms, adjusting for 10hz annotations

    def project_ego_to_img_motion_compensated(self, points_lidar_time, cam_name, timestamp_ns, log_id):
        img_path = super().get_closest_img_fpath(log_id, cam_name, timestamp_ns)
        if img_path is None:
            n = len(points_lidar_time)
            return np.zeros((n, 2)), np.zeros((n, 3)), np.zeros(n, dtype=bool)

        cam_timestamp_ns = int(img_path.stem)
        return super().project_ego_to_img_motion_compensated(points_lidar_time, cam_name, cam_timestamp_ns, timestamp_ns, log_id)


def composable(composable_func):
    """
    A decorator to evaluate track crossings in parallel for the given composable function.
    
    Args:
        composable_func (function): A function that is evaluated on the track and candidate data.
    
    Returns:
        function: A new function that wraps `composable_func` and adds parallel evaluation.
    """
    @wraps(composable_func)
    def wrapper(track_candidates, log_dir, *args, **kwargs):
        """
        The wrapper function that adds parallel processing and filtering to the decorated function.
        
        Args:
            tracks (dict): Keys are track UUIDs, values are lists of valid timestamps.
            candidates (dict): Keys are candidate UUIDs, values are lists of valid timestamps.
            log_dir (Path): Directory containing log data.
            *args, **kwargs: Additional arguments passed to `composable_func`.
            
        Returns:
            dict: Subset of `track_dict` containing tracks being crossed and their crossing timestamps.
            dict: Nested dict where keys are track UUIDs, values are dicts of candidate UUIDs with their crossing timestamps.
        """
        # Process tracks and candidates into dictionaries
        track_dict = to_scenario_dict(track_candidates, log_dir)

        # Parallelize processing of the UUIDs
        all_uuids = list(track_dict.keys())

        true_tracks, _ = parallelize_uuids(composable_func, all_uuids, log_dir, *args, **kwargs)
        # Apply filtering
        scenario_dict = {}

        for track_uuid, unfiltered_related_objects in track_dict.items():
            if true_tracks.get(track_uuid, None) is not None:
                prior_related_objects = scenario_at_timestamps(unfiltered_related_objects, get_scenario_timestamps(true_tracks[track_uuid]))
                scenario_dict[track_uuid] = prior_related_objects   

        return scenario_dict

    return wrapper

def composable_relational(composable_func):
    """
    A decorator to evaluate track crossings in parallel for the given composable function.
    
    Args:
        composable_func (function): A function that is evaluated on the track and candidate data.
    
    Returns:
        function: A new function that wraps `composable_func` and adds parallel evaluation.
    """
    @wraps(composable_func)
    def wrapper(track_candidates, related_candidates, log_dir, *args, **kwargs):
        """
        The wrapper function that adds parallel processing and filtering to the decorated function.
        
        Args:
            tracks (dict): Keys are track UUIDs, values are lists of valid timestamps.
            candidates (dict): Keys are candidate UUIDs, values are lists of valid timestamps.
            log_dir (Path): Directory containing log data.
            *args, **kwargs: Additional arguments passed to `composable_func`.
            
        Returns:
            dict: Subset of `track_dict` containing tracks being crossed and their crossing timestamps.
            dict: Nested dict where keys are track UUIDs, values are dicts of candidate UUIDs with their crossing timestamps.
        """
        # Process tracks and candidates into dictionaries
        track_dict = to_scenario_dict(track_candidates, log_dir)
        related_candidate_dict = to_scenario_dict(related_candidates, log_dir)
        track_dict, related_candidate_dict = remove_nonintersecting_timestamps(track_dict, related_candidate_dict)

        # Parallelize processing of the UUIDs
        track_uuids = list(track_dict.keys())
        candidate_uuids = list(related_candidate_dict.keys())

        _, relationship_dict = parallelize_uuids(composable_func, track_uuids, candidate_uuids, log_dir, *args, **kwargs)

        # Apply filtering
        scenario_dict = {track_uuid: {} for track_uuid in relationship_dict.keys()}

        for track_uuid, unfiltered_related_objects in track_dict.items():
            if isinstance(unfiltered_related_objects, dict) and track_uuid in relationship_dict:
                prior_related_objects = scenario_at_timestamps(unfiltered_related_objects, get_scenario_timestamps(relationship_dict[track_uuid]))
                scenario_dict[track_uuid] = prior_related_objects   

        for track_uuid, unfiltered_related_objects in relationship_dict.items():
            for related_uuid, related_timestamps in unfiltered_related_objects.items():
                eligible_timestamps = sorted(set(related_timestamps).intersection(get_scenario_timestamps(track_dict[track_uuid])))
                scenario_dict[track_uuid][related_uuid] = scenario_at_timestamps(related_candidate_dict[related_uuid], eligible_timestamps)            

        return scenario_dict

    return wrapper


def scenario_at_timestamps(scenario_dict:dict, kept_timestamps):
    scenario_with_timestamps = deepcopy(scenario_dict)

    if not isinstance(scenario_dict, dict):
        return sorted(list(set(scenario_dict).intersection(kept_timestamps)))

    keys_to_remove = []
    for uuid, relationship in scenario_with_timestamps.items():
        relationship = scenario_at_timestamps(relationship, kept_timestamps)
        scenario_with_timestamps[uuid] = relationship
        
        if len(relationship) == 0:
            keys_to_remove.append(uuid)

    for key in keys_to_remove:
        scenario_with_timestamps.pop(key)

    return scenario_with_timestamps


def remove_nonintersecting_timestamps(dict1:dict[str,list], dict2:dict[str,list]):

    dict1_timestamps = get_scenario_timestamps(dict1)
    dict2_timestamps = get_scenario_timestamps(dict2)

    dict1 = scenario_at_timestamps(dict1, dict2_timestamps)
    dict2 = scenario_at_timestamps(dict2, dict1_timestamps)

    return dict1, dict2

@cache_manager.create_cache('get_ego_uuid')
def get_ego_uuid(log_dir):
    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_df = df[df['category'] == 'EGO_VEHICLE']
    return ego_df['track_uuid'].iloc[0]


def get_cuboids_of_category(cuboids: list[Cuboid], category):
    objects_of_category = []
    for cuboid in cuboids:
        if cuboid.category == category:
            objects_of_category.append(cuboid)
    return objects_of_category 


def get_uuids_of_category(log_dir:Path, category:str):
    """
    Returns all uuids from a given category from the log annotations. This method accepts the 
    super classes "ANY" and "VEHICLE".

    Args:
        log_dir: Path to the directory containing scenario logs and data.
        category: the category of objects to return

    Returns: 
        list: the uuids of objects that fall within the category

    Example:
        trucks = get_uuids_of_category(log_dir, category='TRUCK')
    """

    df = read_feather(log_dir / 'sm_annotations.feather')

    if category == 'ANY':
        uuids = df['track_uuid'].unique()
    elif category == 'VEHICLE':

        uuids = []
        vehicle_superclass = ["EGO_VEHICLE","ARTICULATED_BUS","BOX_TRUCK","BUS","LARGE_VEHICLE", "CAR",
                              "MOTORCYCLE","RAILED_VEHICLE","REGULAR_VEHICLE","SCHOOL_BUS","TRUCK","TRUCK_CAB"]
        
        for vehicle_category in vehicle_superclass:
            category_df = df[df['category'] == vehicle_category]
            uuids.extend(category_df['track_uuid'].unique())
    else:
        category_df = df[df['category'] == category]
        uuids = category_df['track_uuid'].unique()

    return uuids


@cache_manager.create_cache('has_free_will')
def has_free_will(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    category = df[df['track_uuid'] == track_uuid]['category'].iloc[0]
    if category in ['ANIMAL','OFFICIAL_SIGNALER','RAILED_VEHICLE','ARTICULATED_BUS','WHEELED_RIDER','SCHOOL_BUS',
                    'MOTORCYCLIST','TRUCK_CAB','VEHICULAR_TRAILER','BICYCLIST','MOTORCYCLE','TRUCK','BOX_TRUCK','BUS',
                    'LARGE_VEHICLE','PEDESTRIAN','REGULAR_VEHICLE', 'EGO_VEHICLE']:
        return True
    else:
        return False


@composable
def get_object(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = df[df['track_uuid'] == track_uuid]

    if track_df.empty:
        print(f'Given track_uuid {track_uuid} not in log annotations.')
        return []
    else:
        timestamps = track_df['timestamp_ns']
        return sorted(timestamps)


def get_eval_timestamps(log_dir:Path):
    """
    Return the timestamps of the driving log used for evaluation.
    For competitions based on the AV2 sensor dataset, this is log_timesetamps[::5] (converting from from 10hz to 2hz).
    """
    log_timestamps = get_log_timestamps(log_dir)

    try:
        with open('run/experiment_configs/eval_timestamps.json', 'rb') as file:
            eval_timestamps_by_log_id = json.load(file)
        eval_timestamps = eval_timestamps_by_log_id[log_dir.stem]
    except:
        # This assumes that your input has predictions for all of the timestamps
        # This is valid assumption for the RefProg code, but not for the baselines
        MAX_NUM_EVAL_TIMESTAMPS = 50
        if len(log_timestamps) > MAX_NUM_EVAL_TIMESTAMPS:
            eval_timestamps = log_timestamps[::5]
        else:
            eval_timestamps = log_timestamps

    return eval_timestamps

def get_camera_names(log_dir):

    try:
        intrinsics = read_feather(log_dir/'calibration/intrinsics.feather')
    except:
        split = get_log_split(log_dir)
        intrinsics = read_feather(paths.AV2_DATA_DIR/split/log_dir.name/'calibration/intrinsics.feather')

    camera_names = list(intrinsics['sensor_name'])
    
    # Remove stereo cameras for now
    camera_names = [cam for cam in camera_names if 'stereo' not in cam.lower()]

    return camera_names


@cache_manager.create_cache('get_img_crops')
def get_img_crops(track_uuid, log_dir:Path)->dict[str,dict[int,tuple[int,int,int,int]|None]]:
    """Return bounding boxes of a track's cuboid projected into each ring camera.

    Format: ``{cam_name: {timestamp: (x1, y1, x2, y2) | None}}``.

    A box is kept when the cuboid is within ``MAX_VIEW_DIST_M`` of ego
    and at least one vertex projects inside the image.

    Disk cache: ``<log_dir>/cache/img_crops/<uuid>.json``. Since files are
    per-uuid, pathos workers writing different uuids concurrently never race.
    The cache is invalidated (recomputed) if it is older than the
    sm_annotations.feather mtime. This gives instant responses from the second
    scenario click onward, even for a log_dir that had no persistent disk cache
    such as GT.
    """
    MAX_VIEW_DIST_M = 50

    # ── disk cache fast path ──
    cache_dir = log_dir / 'cache' / 'img_crops'
    cache_file = cache_dir / f'{track_uuid}.json'
    feather = log_dir / 'sm_annotations.feather'
    if cache_file.exists() and feather.exists():
        try:
            if cache_file.stat().st_mtime >= feather.stat().st_mtime:
                with open(cache_file, 'r') as f:
                    raw = json.load(f)
                return {
                    cam: {int(ts): tuple(b) if b is not None else None
                          for ts, b in tsmap.items()}
                    for cam, tsmap in raw.items()
                }
        except Exception:
            pass  # corrupted file → fallthrough to recompute

    dataloader = EasyDataLoader(log_dir)
    camera_names = get_camera_names(log_dir)
    timestamps = get_timestamps(track_uuid, log_dir)

    img_crops = {}
    for timestamp in timestamps:

        cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp)
        points = cuboid.vertices_m
        cuboid_in_range = np.linalg.norm(cuboid.xyz_center_m) <= MAX_VIEW_DIST_M

        for cam_name in camera_names:
            if cam_name not in img_crops:
                img_crops[cam_name] = {}
            elif timestamp not in img_crops[cam_name]:
                img_crops[cam_name][timestamp] = None

            if not cuboid_in_range:
                continue

            uv, points_cam, is_valid = dataloader.project_ego_to_img_motion_compensated(points, cam_name, timestamp, log_dir.name)

            camera = dataloader.get_log_pinhole_camera(log_dir.name, cam_name)
            W = camera.width_px
            H = camera.height_px

            in_frame = (
                is_valid
                & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            )
            if in_frame.sum() < 1:
                continue

            uv_in_frame = uv[in_frame]
            x_min = np.min(uv_in_frame[:, 0])
            x_max = np.max(uv_in_frame[:, 0])
            y_min = np.min(uv_in_frame[:, 1])
            y_max = np.max(uv_in_frame[:, 1])

            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(W, int(x_max))
            y2 = min(H, int(y_max))

            if x2 > x1 and y2 > y1:
                box = (x1, y1, x2, y2)
                img_crops[cam_name][timestamp] = box

    # ── persist to disk (atomic rename, JSON serializable form) ──
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        serializable = {
            cam: {str(ts): list(b) if b is not None else None
                  for ts, b in tsmap.items()}
            for cam, tsmap in img_crops.items()
        }
        tmp = cache_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(serializable, f)
        os.replace(tmp, cache_file)
    except Exception:
        pass  # even if the cache write fails, the evaluation still returns normally

    return img_crops


@cache_manager.create_cache('get_context_annotations')
def get_context_annotations(log_dir: Path) -> dict[int, dict[str, dict]]:
    """Load per-timestamp per-camera Scene Context VLM annotations for one log.

    Returns ``{timestamp_ns: {cam_name: {"infra": {...}, "weather": {...},
    "time_of_day": {...}, ...}}}``.

    Auto-detects the JSON schema:
      - v4.1 (current): top-level ``per_camera`` key. Per-camera block is
        returned; ego items are not included here (use a dedicated ego loader
        when needed).
      - v3 (legacy): top-level ``cameras`` key. Returned as-is.

    Reads from the postprocessed split (``{split}_processed``) produced by
    ``tools/scene_context_extraction/src/postprocess_runner.py``
    """
    context_dir = paths.CONTEXT_ANNOTATIONS_DIR / f"{get_log_split(log_dir)}_processed" / Path(log_dir).name
    if not context_dir.exists():
        return {}

    annotations: dict[int, dict[str, dict]] = {}
    for json_file in context_dir.glob("*.json"):
        try:
            ts = int(json_file.stem)
        except ValueError:
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        annotations[ts] = data.get("per_camera", data.get("cameras", {}))
    return annotations


@cache_manager.create_cache('get_ego_annotations')
def get_ego_annotations(log_dir: Path) -> dict[int, dict[str, bool]]:
    """Return per-timestamp ego-block annotations as ``{ts_ns: {key: bool}}``."""
    context_dir = paths.CONTEXT_ANNOTATIONS_DIR / f"{get_log_split(log_dir)}_processed" / Path(log_dir).name
    if not context_dir.exists():
        return {}

    annotations: dict[int, dict[str, bool]] = {}
    for json_file in context_dir.glob("*.json"):
        try:
            ts = int(json_file.stem)
        except ValueError:
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        annotations[ts] = data.get("ego", {}) or {}
    return annotations


@cache_manager.create_cache('get_all_crops')
def get_all_crops(log_dir:Path, timestamps=None, track_uuids=None)->dict[str,dict[int,tuple[int,int,int,int]|None]]:
    """Returns all of the image bounding boxes for a given track. This is in the format

        img_crops[timestamp][cam_name][track_uuid] = {
            'category': categories[i],
            'percent_in_cam': percent_in_cam,
            'crop_area': crop_area,
            'cam_H':H,
            'cam_W':W,
            'bbox': (x_min, x_max, y_min, y_max),
            'crop': (x1, y1, x2, y2),
            'cam_z': camera_depths[i]
        }
        
    """
    cache_path = log_dir/'cache/track_crop_information.json'

    if cache_path.exists():
        with open(cache_path, 'rb') as file:
            img_crops = json.load(file)
        return img_crops

    dataloader = EasyDataLoader(log_dir)
    camera_names = get_camera_names(log_dir)

    if timestamps is None:
        timestamps = get_log_timestamps(log_dir)
    if track_uuids is None:
        track_uuids = get_uuids_of_category(log_dir, 'ANY')

    ego_uuid = get_ego_uuid(log_dir)

    img_crops = {}
    for timestamp in tqdm(timestamps, desc='Getting track crop information by timestamp.'):
        
        timestamp = int(timestamp)

        for cam_name in camera_names:
            
            camera = dataloader.get_log_pinhole_camera(log_dir.name, cam_name)
            W = camera.width_px
            H = camera.height_px
            
            if timestamp not in img_crops:
                img_crops[timestamp] = {}
            if cam_name not in img_crops[timestamp]:
                img_crops[timestamp][cam_name] = {}

            if cam_name == 'ring_front_center' or cam_name == 'CAM_FRONT':
                img_crops[timestamp][cam_name][str(ego_uuid)] = {
                    'category': 'EGO_VEHICLE',
                    'percent_in_cam': 1.00,
                    'crop_area': W*H,
                    'cam_H':H,
                    'cam_W':W,
                    'bbox': (0, 0, W, H),
                    'crop': (0, 0, W, H),
                    'cam_z': 0.5 # Actually will be negative, dummy value to not get filtered out in later code
                }

            cuboid_vertices = []
            cuboid_centroids = []
            categories = []
            valid_track_mask = np.zeros(len(track_uuids), dtype=bool)
            for i, track_uuid in enumerate(track_uuids):

                cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp)
                if cuboid is not None:
                    valid_track_mask[i] = True
                    cuboid_vertices.append(cuboid.vertices_m)
                    cuboid_centroids.append(cuboid.xyz_center_m[np.newaxis,:])
                    categories.append(cuboid.category)
                else:
                    categories.append('filler')
                    cuboid_vertices.append(np.zeros((8,3)))
                    cuboid_centroids.append(np.zeros((1,3)))
            
            # Concatenating centroids and vertices for more efficient computation
            points_ego = np.concat([np.concat(cuboid_centroids, axis=0), np.concat(cuboid_vertices, axis=0)])
            uv, points_cam, is_valid = dataloader.project_ego_to_img_motion_compensated(points_ego, cam_name, timestamp, log_dir.name)

            # Unstacking the centroids and vertices
            camera_depths = points_cam[:len(track_uuids), 2]
            uv = uv[len(track_uuids):].reshape((len(track_uuids), 8, 2))
            is_valid = np.sum(is_valid[len(track_uuids):].reshape(len(track_uuids), 8), axis=1) > 2 # must have at least three vertices within view of the camera
            valid_track_mask = valid_track_mask & is_valid

            for i, track_uuid in enumerate(track_uuids):
                track_uuid = str(track_uuid)
                if track_uuid in img_crops[timestamp][cam_name] or not valid_track_mask[i] or camera_depths[i] < 0:
                    continue
                
                x_min = np.min(uv[i,:,0])
                x_max = np.max(uv[i,:,0])
                y_min = np.min(uv[i,:,1])
                y_max = np.max(uv[i,:,1])
                
                x1 = max(0, int(x_min))
                y1 = max(0, int(y_min))
                x2 = min(W, int(x_max))
                y2 = min(H, int(y_max))

                if x2 > x1 and y2 > y1:
                    crop_area= (x2-x1)*(y2-y1)
                    bbox_area = ((x_max-x_min)*(y_max-y_min))
                    percent_in_cam = crop_area / bbox_area

                    img_crops[timestamp][cam_name][track_uuid] = {
                        'category': categories[i],
                        'percent_in_cam': percent_in_cam,
                        'crop_area': crop_area,
                        'cam_H':H,
                        'cam_W':W,
                        'bbox': (x_min, x_max, y_min, y_max),
                        'crop': (x1, y1, x2, y2),
                        'cam_z': camera_depths[i]
                    }

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, 'w') as file:
        json.dump(img_crops, file, indent=4)
    print(f'Log id crop information stored in {cache_path}')

    return img_crops   


def get_best_crop(track_uuid, log_dir)->dict:
    """ Returns the timestamp, camera, and image bounding box
    according to the maximum area of the track bounding box in the format.
    
    {'timestamp': timestamp, 'cam': cam, 'crop': crop, 'score': score, 'category': object_crops[timestamp][cam][track_uuid]['category']}
    """
    object_crops = get_all_crops(log_dir)

    timestamps_and_cams = []
    for timestamp, crops_by_camera in object_crops.items():
        for camera, crops_by_uuid in crops_by_camera.items():
            if track_uuid in crops_by_uuid:
                timestamps_and_cams.append((timestamp, camera))

    best_score = 0
    best_crop = None
    for timestamp, cam in timestamps_and_cams:
        
        track_crop_dict = object_crops[timestamp][cam][track_uuid]
        visibility_mask = np.zeros((track_crop_dict['cam_H'], track_crop_dict['cam_W']))

        track_x1, track_y1, track_x2, track_y2 = track_crop_dict['crop']
        visibility_mask[track_y1:track_y2, track_x1:track_x2] = True
        percent_in_cam = track_crop_dict['percent_in_cam']
        track_depth = track_crop_dict['cam_z']

        for uuid, crop_dict in object_crops[timestamp][cam].items():
            # Skip self and objects behind the track. Also skip the ego vehicle:
            # it is stored as a full-frame (0,0,W,H) box, so treating it as an
            # occluder would blank the whole mask and zero out every front-camera score.
            if (uuid == track_uuid or crop_dict['category'] == 'EGO_VEHICLE'
                    or crop_dict['cam_z'] < 0 or crop_dict['cam_z'] > track_depth):
                continue
            #else the object is located between the camera and the track, figure out which pixels are occluded

            object_x1, object_y1, object_x2, object_y2 = object_crops[timestamp][cam][uuid]['crop']
            visibility_mask[object_y1:object_y2, object_x1:object_x2] = False

        visible_area = np.sum(visibility_mask)
        percent_unoccluded = visible_area / track_crop_dict['crop_area']
        score = percent_in_cam * percent_unoccluded *  visible_area / 100

        if score >= best_score:
            best_score = score

            pad_x = .2 * (track_x2 - track_x1)
            pad_y = .2 * (track_y2 - track_y1)
            crop = (
                max(0, int(track_x1 - pad_x)),
                max(0, int(track_y1 - pad_y)),
                min(track_crop_dict['cam_W'], int(track_x2 + pad_x)),
                min(track_crop_dict['cam_H'], int(track_y2 + pad_y))
            )

            best_crop = {'timestamp': timestamp, 'cam': cam, 'crop': crop, 'score': score, 'category': object_crops[timestamp][cam][track_uuid]['category']}

    return best_crop


@cache_manager.create_cache('get_img_crop')
def get_img_crop(camera, timestamp, log_dir:Path, box=None):

    dataloader = EasyDataLoader(log_dir)
    img_path = dataloader.get_closest_img_fpath(log_dir.name, camera, timestamp)

    if img_path is None:
        return None

    img = Image.open(img_path)

    if box is not None:
        img = img.crop(box)

    return img


# ---------------------------------------------------------------------------
# VLM visual-filter plumbing (Qwen3.6-35B-A3B via vLLM). Used by the atomic
# functions get_visual_actor / get_visual_behavior (refAV/atomic_functions.py),
# which call _visual_filter with mode='actor' | 'behavior'. Endpoints come from
# REFAV_VLM_ENDPOINTS, model from REFAV_VLM_MODEL.
# ---------------------------------------------------------------------------

_VLM_SYSTEM = (
    "You are a precise visual classifier for an autonomous-driving perception dataset. "
    "Each image is a tight crop from a vehicle's ring camera showing ONE tracked road object; "
    "the object of interest is at the CENTER of the crop (edge content is context, not the target). "
    "Judge ONLY the centered object. Be STRICT: answer true ONLY when there is clear, unambiguous "
    "visual evidence. If it is occluded, too small/blurry, ambiguous, or merely a generic object "
    "lacking the specific described feature, answer false. Output only a compact JSON object."
)

class VlmServerError(RuntimeError):
    """A VLM request could not produce a verdict — server down/unreachable, an
    HTTP/timeout error, or a reply without a parseable {"match": ...}. The visual
    atoms raise it to abort the run loudly rather than pass every track as
    "no match". Health-check first with tools/vlm_server/check_connection.py."""

def _vlm_endpoints():
    raw = os.environ.get(
        "REFAV_VLM_ENDPOINTS",
        "http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003")
    return [u.strip().rstrip("/") + "/v1/chat/completions" for u in raw.split(",") if u.strip()]

def _vlm_crop_b64(uuid, log_dir):
    best = get_best_crop(str(uuid), log_dir)
    if best is None:
        return None
    try:
        img = get_img_crop(best["cam"], int(best["timestamp"]), log_dir, box=tuple(best["crop"]))
    except Exception:
        return None
    if img is None:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return _b64.b64encode(buf.getvalue()).decode("ascii")

def _vlm_call(b64, user_text, endpoints, start_idx=0):
    
    _VLM_TIMEOUT = 30          # per-request seconds (a 20-token classify is fast)

    payload = {
        "model": os.environ.get("REFAV_VLM_MODEL", "qwen3.6-35b"),
        "temperature": 0.0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
        # Force a parseable {"match": bool} verdict. Some served checkpoints
        # (e.g. Qwen3.6-A3B on images) ignore the JSON-only instruction and
        # reply in prose; json_schema guided decoding constrains the format
        # without changing the (temperature-0) verdict.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "schema": {
                    "type": "object",
                    "properties": {"match": {"type": "boolean"}},
                    "required": ["match"],
                },
            },
        },
        "messages": [
            {"role": "system", "content": _VLM_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
    }
    data = json.dumps(payload).encode()
    n = len(endpoints)
    transport_errors = []
    for k in range(n):
        endpoint = endpoints[(start_idx + k) % n]
        req = _urlreq.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
        try:
            with _urlreq.urlopen(req, timeout=_VLM_TIMEOUT) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            transport_errors.append(f"{endpoint}: {e!r}")
            continue        # fail over to the next replica
        a, b = txt.find("{"), txt.rfind("}")
        if a != -1 and b > a:
            try:
                return bool(json.loads(txt[a:b + 1]).get("match"))
            except Exception as e:
                raise VlmServerError(
                    f"VLM reply from {endpoint} had no parseable JSON verdict: {txt!r}. "
                    f"Is thinking mode disabled (chat_template_kwargs enable_thinking=false)? "
                    f"See tools/vlm_server/README.md 'Troubleshooting'."
                ) from e
        raise VlmServerError(
            f"VLM reply from {endpoint} carried no {{\"match\": ...}} JSON: {txt!r}. "
            f"Is thinking mode disabled (chat_template_kwargs enable_thinking=false)? "
            f"See tools/vlm_server/README.md 'Troubleshooting'."
        )
    # Every endpoint failed at the transport level -> the whole fleet is down.
    raise VlmServerError(
        f"VLM request failed on all {n} endpoint(s): {'; '.join(transport_errors)}. "
        f"Is the vLLM fleet up and reachable? Verify with tools/vlm_server/check_connection.py "
        f"(REFAV_VLM_ENDPOINTS={os.environ.get('REFAV_VLM_ENDPOINTS', 'http://localhost:8000..8003')})."
    )

def _visual_filter(track_candidates: dict, log_dir: Path, description: str,
                   mode: str, max_workers: int = 16) -> dict:
    if not track_candidates:
        return {}
    if mode == "actor":
        user_text = (f"Does the centered object in this crop clearly show {description}? "
                     "Answer true ONLY if there is definite, clearly visible evidence. "
                     'Respond with ONLY a JSON object: {"match": true} or {"match": false}.')
    else:  # behavior
        user_text = (f'Does the centered object in this crop clearly match this appearance/behavior: '
                     f'"{description}"? Answer true ONLY if it is definitely, clearly visible in this '
                     'single frame. Respond with ONLY a JSON object: {"match": true} or {"match": false}.')
    candidates = list(track_candidates)
    get_best_crop(str(candidates[0]), log_dir)        # warm per-log crop cache before threads
    endpoints = _vlm_endpoints()
    with _ThreadPool(max_workers=max_workers) as ex:
        crops = dict(ex.map(lambda u: (u, _vlm_crop_b64(u, log_dir)), candidates))
    items = [(u, b64) for u, b64 in crops.items() if b64]
    with _ThreadPool(max_workers=max_workers) as ex:
        verdicts = ex.map(
            lambda iu: (iu[1][0], _vlm_call(iu[1][1], user_text, endpoints, iu[0] % len(endpoints))),
            enumerate(items))
        matched = {u for u, m in verdicts if m}
    return {u: track_candidates[u] for u in matched}


def get_clip_colors(images:list, possible_colors:list[str], pipe=None):
    
    texts = [f'a {color} object' for color in possible_colors]
    
    # Initialize pipeline with device_map for multi-GPU
    if pipe is None:
        pipe = pipeline(
            model="google/siglip2-so400m-patch16-naflex",
            task="zero-shot-image-classification",
            device_map="auto",  # Automatically distributes across available GPUs
            dtype="auto",
            batch_size=16
        )
    
    outputs = pipe(images, candidate_labels=texts)
    
    # Process outputs same as before
    best_labels = []
    for output in outputs:
        best_label = max(output, key=lambda x: x['score'])['label'].split()[1]
        best_labels.append(best_label)

    return best_labels


@lru_cache(maxsize=2)
def _load_siglip_model(device: str):
    """Lazy-load the SigLIP2 model + processor on the given device (one entry per device).

    Used to embed crops at cache-build time (GPU) and to encode subcategory text
    at runtime (CPU). use_fast=False because torchvision is absent in this env.
    """
    import torch
    from transformers import AutoModel, AutoProcessor
    ckpt = "google/siglip2-so400m-patch16-naflex"
    model = AutoModel.from_pretrained(ckpt).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt, use_fast=False)
    return model, processor


def _gpu_auto_batch_size(device):
    """Pick an initial batch size from free VRAM headroom (CUDA only).

    Rough heuristic (~8 crops per GB of free memory, clamped to [8, 512]); the
    embed loop backs off automatically on OOM, so this only needs to be a sane
    starting point. CPU / unknown -> a modest fixed batch.
    """
    import torch
    if device.type != "cuda":
        return 16
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    return int(max(8, min(512, free_gb * 8)))


def _embed_images(model, processor, image_paths, batch_size: int = None):
    """Return L2-normalized SigLIP2 image features (N, D) as a float32 numpy array.

    Same logic as the feasibility notebook (naflex processor keeps aspect ratio,
    get_image_features -> pooler_output -> L2 norm), but the batch size is dynamic:
    it starts from free-VRAM headroom (batch_size=None) and halves on CUDA OOM,
    retrying the same chunk, so it uses the available memory without crashing.
    """
    import torch
    device = next(model.parameters()).device
    if batch_size is None:
        batch_size = _gpu_auto_batch_size(device)

    embs = []
    i = 0
    n = len(image_paths)
    while i < n:
        bs = min(batch_size, n - i)
        try:
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i + bs]]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            pooled = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = pooled.float().cpu()
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            embs.append(emb)
            i += bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, bs // 2)   # back off and retry this same chunk smaller
    return torch.cat(embs, dim=0).numpy()


@lru_cache(maxsize=256)
def get_subcategory_text_embedding(prompt: str) -> np.ndarray:
    """Encode a subcategory prompt into an L2-normalized SigLIP2 text feature (D,).

    Runs on CPU (text is short and infrequent, so this avoids competing with the
    tracker for GPU memory). Cached per prompt. Template matches the notebook.
    """
    import torch
    model, processor = _load_siglip_model("cpu")
    article = "an" if prompt[:1].lower() in "aeiou" else "a"
    inputs = processor(text=[f"a photo of {article} {prompt}"],
                       padding="max_length", max_length=64, truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    pooled = out.pooler_output if hasattr(out, "pooler_output") else out
    emb = pooled.float()
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze(0).numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_siglip_logit_params():
    """SigLIP2 calibration (exp(logit_scale), logit_bias), about (113.73, -15.94).

    prob = sigmoid(cos * scale + bias). Read from the model so it stays correct
    if the checkpoint ever changes.
    """
    model, _ = _load_siglip_model("cpu")
    return float(model.logit_scale.detach().exp()), float(model.logit_bias.detach())


@cache_manager.create_cache('category_score_maps')
def get_category_score_maps(log_dir):
    """Per-log {uuid: category} and {uuid: max detection score} from the feather.

    Reads the feather once (cached) and builds both maps so callers can look up a
    track's class and confidence without scanning per uuid. A missing 'score'
    column (e.g. ground-truth feather) defaults every score to 1.0.
    """
    df = read_feather(log_dir / 'sm_annotations.feather')
    cat_map = {str(u): c for u, c in zip(df['track_uuid'], df['category'])}
    if 'score' in df.columns:
        score_map = {str(u): float(s) for u, s in df.groupby('track_uuid')['score'].max().items()}
    else:
        score_map = {u: 1.0 for u in cat_map}
    return cat_map, score_map


def _build_map_caches_for_log(log_dir):
    """Build semantic_lane_cache and road_side_cache for a single log and save to disk.

    Saves to paths.GLOBAL_CACHE_PATH/{log_id}/ since these are tracker-independent.
    """
    log_dir = Path(log_dir)
    log_id = log_dir.name
    global_cache_dir = paths.GLOBAL_CACHE_PATH / log_id
    global_cache_dir.mkdir(parents=True, exist_ok=True)
    avm = None

    # --- Semantic lane cache ---
    semantic_path = global_cache_dir / 'semantic_lane_cache.json'
    if not semantic_path.exists():
        avm = get_map(log_dir)
        semantic_cache = {}
        for ls_id, ls in avm.vector_lane_segments.items():
            lanes = get_semantic_lane(ls, log_dir, avm=avm)
            semantic_cache[str(ls_id)] = [l.id for l in lanes]
        with open(semantic_path, 'w') as f:
            json.dump(semantic_cache, f)
    else:
        with open(semantic_path, 'r') as f:
            semantic_cache = json.load(f)

    # Set so get_road_side -> get_semantic_lane can use it within this process
    cache_manager.semantic_lane_cache = semantic_cache

    # --- Road side cache ---
    road_side_path = global_cache_dir / 'road_side_cache.json'
    if not road_side_path.exists():
        if avm is None:
            avm = get_map(log_dir)
        rs_cache = {}
        for ls_id, ls in avm.vector_lane_segments.items():
            same = get_road_side(ls, log_dir, 'same', avm=avm)
            opp = get_road_side(ls, log_dir, 'opposite', avm=avm)
            rs_cache[str(ls_id)] = {
                'same': [s.id for s in same],
                'opposite': [o.id for o in opp]
            }
        with open(road_side_path, 'w') as f:
            json.dump(rs_cache, f)


# ---------------------------------------------------------------------------
# Tracker cuboid z-fix (per-split crop correction)
#
# Some tracker feathers store tz_m as the cuboid TOP rather than its center, so
# the boxes float about height/2 above the ground and the projected crop grabs
# the empty space above the object. The fix lowers the av2 Cuboid geometry by
# height/2 at crop time only (an in-process monkey patch); the feather is never
# modified, and scenario-mining eval matches in BEV xy, so eval is unaffected.
# Whether a (tracker, split) needs the fix is decided once by measuring how high
# parked cars sit, then cached under output/cache/z_offset/.
# ---------------------------------------------------------------------------
CROP_SCORE_THRESHOLD = 0.05      # tracks below this confidence get no crop / embedding
_ZFIX_GROUND_TOLERANCE_M = 0.2   # parked cars above this height count as floating
_zfix_applied = False


def _apply_cuboid_zfix_patch():
    """Lower av2 Cuboid vertices and center by height/2. Idempotent per process."""
    global _zfix_applied
    if _zfix_applied:
        return
    original_vertices = Cuboid.vertices_m.func     # cached_property -> .func
    original_center = Cuboid.xyz_center_m.fget      # property -> .fget

    def vertices_lowered(self):
        v = original_vertices(self).copy()
        v[:, 2] -= self.height_m / 2
        return v

    def center_lowered(self):
        c = original_center(self).copy()
        c[2] -= self.height_m / 2
        return c

    Cuboid.vertices_m = property(vertices_lowered)
    Cuboid.xyz_center_m = property(center_lowered)
    _zfix_applied = True


def tracker_needs_zfix(log_dir) -> bool:
    """Whether crops for this (tracker, split) need the cuboid z-fix.

    Decided by the median ground-vehicle bottom height across the split: parked
    cars sitting well above z=0 mean tz_m is a top, not a center. The verdict is
    cached to output/cache/z_offset/<tracker>__<split>.json so the measurement
    runs only once per tracker/split.
    """
    log_dir = Path(log_dir)
    split = get_log_split(log_dir)
    tracker = log_dir.parents[1].name if len(log_dir.parents) >= 2 else 'unknown'
    cache_path = paths.GLOBAL_CACHE_PATH / 'z_offset' / f'{tracker}__{split}.json'
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return bool(json.load(f)['needs_zfix'])
        except Exception:
            pass

    bottoms = []
    for ld in sorted(log_dir.parent.iterdir()):
        feather_path = ld / 'sm_annotations.feather'
        if not feather_path.exists():
            continue
        try:
            df = read_feather(feather_path)
        except Exception:
            continue
        cars = df[df['category'] == 'REGULAR_VEHICLE']
        if len(cars):
            bottoms.extend((cars['tz_m'] - cars['height_m'] / 2).tolist())

    median_bottom = float(np.median(bottoms)) if bottoms else 0.0
    needs_zfix = median_bottom > _ZFIX_GROUND_TOLERANCE_M
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({'needs_zfix': needs_zfix, 'median_bottom_z': median_bottom,
                   'tolerance_m': _ZFIX_GROUND_TOLERANCE_M}, f)
    return needs_zfix


def _collect_crops_for_log(args):
    """Collect best crop images for tracks in a log, saving to disk.

    args is a tuple (log_dir, score_threshold, z_fix, force):
      score_threshold  None -> all tracks; float -> only tracks whose max score >= it.
      z_fix            apply the cuboid z-fix patch before projecting crops.
      force            overwrite existing crop PNGs and drop the stale bbox cache
                       (which would otherwise shortcut get_all_crops past the z-fix).
    A bare log_dir (str/Path) is also accepted for backward compatibility.

    Computes get_all_crops (expensive), then for each track finds the best crop
    and saves the cropped image to log_dir/cache/crops/{uuid}.png.
    Returns (log_dir_str, [(uuid, crop_path_or_None), ...]).
    """
    if isinstance(args, (str, Path)):
        log_dir, score_threshold, z_fix, force = args, None, False, False
    else:
        log_dir, score_threshold, z_fix, force = args
    log_dir = Path(log_dir)

    if z_fix:
        _apply_cuboid_zfix_patch()

    crop_save_dir = log_dir / 'cache' / 'crops'
    if force:
        stale = log_dir / 'cache' / 'track_crop_information.json'
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass

    results = []
    try:
        uuids = get_uuids_of_category(log_dir, 'ANY')
    except Exception:
        return str(log_dir), results

    if score_threshold is not None:
        _, score_map = get_category_score_maps(log_dir)
        uuids = [u for u in uuids if score_map.get(str(u), 0.0) >= score_threshold]

    for uuid in uuids:
        uuid_str = str(uuid)
        crop_path = crop_save_dir / f'{uuid_str}.png'

        if crop_path.exists() and not force:
            results.append((uuid_str, str(crop_path)))
            continue

        try:
            best = get_best_crop(uuid_str, log_dir)
            if best is not None:
                img = get_img_crop(
                    best['cam'], int(best['timestamp']),
                    log_dir, box=best['crop']
                )
                if img is not None:
                    crop_save_dir.mkdir(parents=True, exist_ok=True)
                    if crop_path.exists():
                        # Save to a fresh inode to avoid a permission clash with
                        # a PNG another user may have written on shared storage.
                        try:
                            crop_path.unlink()
                        except OSError:
                            pass
                    img.save(crop_path)
                    results.append((uuid_str, str(crop_path)))
                    continue
        except Exception:
            pass
        results.append((uuid_str, None))

    return str(log_dir), results


def construct_caches(log_dirs: list[Path], num_processes: int = None):
    """Construct semantic_lane_cache, road_side_cache, and color_cache for all log_dirs.

    Builds map-based caches (semantic_lane, road_side) in parallel across logs.
    Builds color_cache by collecting crops in parallel, then running a single
    SigLIP pipeline on batched images.
    Skips any cache that already exists on disk.

    Call this before launching parallel eval processes.
    """
    if num_processes is None:
        num_processes = max(int(.9*os.cpu_count()), 1)

    # --- Phase 1: Map caches in parallel (saved to GLOBAL_CACHE_PATH) ---
    logs_needing_map = [
        ld for ld in log_dirs
        if not (paths.GLOBAL_CACHE_PATH / Path(ld).name / 'semantic_lane_cache.json').exists()
        or not (paths.GLOBAL_CACHE_PATH / Path(ld).name / 'road_side_cache.json').exists()
    ]
    if logs_needing_map:
        print(f"Building map caches for {len(logs_needing_map)} logs using {num_processes} processes...")
        pool = Pool(num_processes)
        pool.map(_build_map_caches_for_log, logs_needing_map)
        print("Map cache construction complete.")

    # --- Phase 2: Color caches ---
    logs_needing_color = [
        ld for ld in log_dirs
        if not (Path(ld) / 'cache' / 'color_cache.json').exists()
    ]
    if logs_needing_color:
        print(f"Building color caches for {len(logs_needing_color)} logs...")

        # Phase 2a: Collect and save crop images in parallel across logs (non-GPU).
        # z-fix the crops (per split) so the PNGs shared with the embedding phase
        # are geometrically correct; no score floor here so color sees all tracks.
        print(f"Collecting track crops in parallel using {num_processes} processes...")
        pool = Pool(num_processes)
        color_tasks = [(str(ld), None, tracker_needs_zfix(ld), False) for ld in logs_needing_color]
        crop_results = pool.map(_collect_crops_for_log, color_tasks)

        # Phase 2b: Organize saved crop paths into batches for SigLIP
        possible_colors = ["white", "silver", "black", "red", "yellow", "blue"]
        batch_size = 256
        image_batches = []
        info_batches = []
        current_batch = []
        current_infos = []
        color_caches = {}

        for log_dir_str, track_results in crop_results:
            color_caches[log_dir_str] = {}
            for uuid, crop_path in track_results:
                if crop_path is not None:
                    current_infos.append((log_dir_str, uuid))
                    current_batch.append(crop_path)
                    if len(current_batch) >= batch_size:
                        image_batches.append(current_batch)
                        info_batches.append(current_infos)
                        current_batch = []
                        current_infos = []
                else:
                    color_caches[log_dir_str][uuid] = None

        if current_batch:
            image_batches.append(current_batch)
            info_batches.append(current_infos)

        # Phase 2c: Single SigLIP pipeline on batched crop file paths
        if image_batches:
            pipe = pipeline(
                model="google/siglip2-so400m-patch16-naflex",
                task="zero-shot-image-classification",
                device_map="auto",
                dtype="auto",
                batch_size=256
            )
            for image_batch, batch_info in tqdm(
                zip(image_batches, info_batches),
                total=len(image_batches),
                desc="Running color classification"
            ):
                colors = get_clip_colors(image_batch, possible_colors, pipe=pipe)
                for color, (log_dir_str, track_uuid) in zip(colors, batch_info):
                    color_caches[log_dir_str][track_uuid] = color

        for log_dir_str, color_cache in color_caches.items():
            cache_dir = Path(log_dir_str) / 'cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / 'color_cache.json', 'w') as f:
                json.dump(color_cache, f)

        print("Color cache construction complete.")

    # --- Phase 3: SigLIP2 crop-embedding caches (GPU) ---
    # Embed the crops at the confidence floor with the per-split z-fix; idempotent
    # (skips logs that already have crop_embeddings.npz).
    build_crop_embedding_caches(log_dirs, num_processes=num_processes,
                                score_threshold=CROP_SCORE_THRESHOLD)


def build_crop_embedding_caches(log_dirs: list[Path], num_processes: int = None,
                                collect_crops: bool = True, batch_size: int = None,
                                score_threshold: float = CROP_SCORE_THRESHOLD,
                                z_fix='auto', force: bool = False):
    """Build SigLIP2 crop-embedding caches (cache/crop_embeddings.npz) for the given logs.

    For each log: (optionally) extract the best crop of every track whose max detection
    score >= score_threshold, applying a per-split cuboid z-fix when needed, then embed
    those crops and save {uuids, embs (N, D) fp16 L2-normalized} to crop_embeddings.npz.
    This is the single entry point for crop + embedding builds; construct_caches calls it
    and tools/build_crop_embeddings.py wraps it for multi-GPU runs.

    score_threshold: confidence floor for which tracks get a crop / embedding (None = all).
        Building at a low floor is non-lossy: at eval time the feather is independently
        re-filtered to a higher threshold (run_experiment score_threshold), so the extra
        low-score embeddings simply go unused.
    z_fix: 'auto' decides per (tracker, split) via tracker_needs_zfix; True/False forces it.
    collect_crops: if True (default) ensure crops exist first; False embeds existing crops.
    force: rebuild even if crop_embeddings.npz exists (re-extracts crops, drops the stale
        bbox cache so the z-fix takes effect, and overwrites the npz).
    """
    import torch

    if num_processes is None:
        num_processes = max(int(.9 * os.cpu_count()), 1)

    if force:
        target_logs = [Path(ld) for ld in log_dirs]
    else:
        target_logs = [Path(ld) for ld in log_dirs
                       if not (Path(ld) / 'cache' / 'crop_embeddings.npz').exists()]
    if not target_logs:
        return

    print(f"Building crop-embedding caches for {len(target_logs)} logs "
          f"(score>={score_threshold}, z_fix={z_fix}, force={force})...")
    if collect_crops:
        # z-fix is resolved here in the parent (measured once per tracker/split and
        # cached) so workers just receive a bool.
        tasks = [(str(ld), score_threshold,
                  (tracker_needs_zfix(ld) if z_fix == 'auto' else bool(z_fix)),
                  force)
                 for ld in target_logs]
        pool = Pool(num_processes)
        pool.map(_collect_crops_for_log, tasks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_siglip_model(device)
    for ld in tqdm(target_logs, desc="Embedding crops"):
        crop_dir = Path(ld) / 'cache' / 'crops'
        # Embed only crops above the floor; the dir may hold extra crops from the
        # color phase (which keeps all tracks).
        score_map = None
        if score_threshold is not None:
            _, score_map = get_category_score_maps(ld)
        image_paths, uuids = [], []
        for p in sorted(crop_dir.glob('*.png')):
            if score_map is not None and score_map.get(p.stem, 0.0) < score_threshold:
                continue
            image_paths.append(p)
            uuids.append(p.stem)
        if not image_paths:
            continue
        embs = _embed_images(model, processor, image_paths, batch_size=batch_size)
        np.savez_compressed(
            Path(ld) / 'cache' / 'crop_embeddings.npz',
            uuids=np.array(uuids), embs=embs.astype(np.float16)
        )
    print("Crop-embedding cache construction complete.")


@cache_manager.create_cache('get_timestamps')
def get_timestamps(track_uuid, log_dir):

    df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = df[df['track_uuid'] == track_uuid]

    if track_df.empty:
        print(f'Given track_uuid {track_uuid} not in log annotations.')
        return []
    else:
        timestamps = track_df['timestamp_ns']
        return sorted(timestamps)


def get_log_timestamps(log_dir):
    df = read_feather(log_dir / 'sm_annotations.feather')
    timestamps = df['timestamp_ns'].unique()
    return sorted(timestamps)

@cache_manager.create_cache('_nearby_lane_candidates')
def _nearby_lane_candidates(avm: ArgoverseStaticMap, qx: int, qy: int, qz: int) -> list[LaneSegment]:
    """Stage-1 nearby-lane search keyed by 1m-quantized position.

    Trajectories sample positions at ~10 Hz with sub-meter motion between
    consecutive timestamps; the 5m radius makes the *candidate set* invariant
    to <1m perturbations. Searching with radius 6m here adds a 1m buffer so
    quantization can never miss a lane that the original radius-5 query would
    have returned. The downstream point-in-polygon test (Stage 2) still uses
    the exact position, so the final result is bit-exact with the legacy form.
    """
    return list(avm.get_nearby_lane_segments(np.array([float(qx), float(qy), float(qz)]), 6))


@cache_manager.create_cache('get_lane_segments')
def get_lane_segments(avm: ArgoverseStaticMap, position) -> list[LaneSegment]:
    "Get lane segments object is currently in from city coordinate location"
    qx, qy, qz = round(float(position[0])), round(float(position[1])), round(float(position[2]))
    candidates = _nearby_lane_candidates(avm, qx, qy, qz)

    lane_segments = []
    for ls in candidates:
        if is_point_in_polygon(position[:2], ls.polygon_boundary[:,:2]):  # exact position, parity-preserving
            lane_segments.append(ls)
    return lane_segments


def get_lane_segments_batch(avm: ArgoverseStaticMap, positions) -> list[list[LaneSegment]]:
    """Batched get_lane_segments for many positions.

    Returns a list of length T where entry t is the same as
    get_lane_segments(avm, positions[t]).

    Per-position get_lane_segments performs two costs that compound on long
    trajectories: (1) a Stage-1 nearby-lane search (`_nearby_lane_candidates`)
    keyed by 1m-quantized cell, and (2) a per-candidate Stage-2
    point-in-polygon check. This batch version groups positions by their
    quantized cell so the Stage-1 lookup is amortized (one call per unique
    cell instead of T), and runs the Stage-2 check with the vectorized
    `_is_point_in_polygon_batch` (M points × one polygon per call) instead of
    M individual `is_point_in_polygon` calls per cell. Same algorithm, same
    final lane-segment sets per position — only the work-sharing changes.
    """
    positions = np.asarray(positions, dtype=np.float64)
    T = len(positions)
    if T == 0:
        return []

    # Stage 1: 1m-quantize positions and look up candidates once per unique cell.
    qx = np.round(positions[:, 0]).astype(int)
    qy = np.round(positions[:, 1]).astype(int)
    qz = np.round(positions[:, 2]).astype(int)

    cell_to_indices: dict[tuple, list[int]] = {}
    for i in range(T):
        cell = (int(qx[i]), int(qy[i]), int(qz[i]))
        cell_to_indices.setdefault(cell, []).append(i)

    results: list[list[LaneSegment]] = [[] for _ in range(T)]

    # Stage 2: for each unique cell, run a single batched polygon check
    # across all the positions that fall into that cell.
    for cell, indices in cell_to_indices.items():
        candidates = _nearby_lane_candidates(avm, cell[0], cell[1], cell[2])
        if not candidates:
            continue
        cell_pts_xy = positions[indices, :2]  # (M, 2)
        for ls in candidates:
            polygon = ls.polygon_boundary[:, :2]
            mask = _is_point_in_polygon_batch(cell_pts_xy, polygon)
            for k, inside in enumerate(mask):
                if inside:
                    results[indices[k]].append(ls)

    return results


@cache_manager.create_cache('get_pedestrian_crossings')
def get_pedestrian_crossings(avm: ArgoverseStaticMap, track_polygon) -> list[PedestrianCrossing]:
    "Get pedestrian crossing that object is currently in from city coordinate location"
    ped_crossings = []

    scenario_crossings = avm.get_scenario_ped_crossings()
    for i, pc in enumerate(scenario_crossings):
        if polygons_overlap(pc.polygon[:,:2], track_polygon[:,:2]):
            ped_crossings.append(pc)

    return ped_crossings


# Per-log on-disk store keyed in-memory by log_id, populated lazily from
# `output/cache/{log_id}/scenario_lanes.json`. Stores only lane ids (int),
# so the on-disk format is tracker-agnostic and human-inspectable. Same shape
# as the existing semantic_lane_cache.json (utils.py:128 load_custom_caches).
_scenario_lanes_disk_in_memory: dict = {}


def _scenario_lanes_disk_path(log_dir: Path) -> Path:
    return paths.GLOBAL_CACHE_PATH / Path(log_dir).name / 'scenario_lanes.json'


def _scenario_lanes_disk_load(log_dir: Path) -> dict:
    """Lazily load the per-log cache file into _scenario_lanes_disk_in_memory."""
    log_id = Path(log_dir).name
    if log_id not in _scenario_lanes_disk_in_memory:
        path = _scenario_lanes_disk_path(log_dir)
        if path.exists():
            try:
                with open(path) as f:
                    _scenario_lanes_disk_in_memory[log_id] = json.load(f)
            except Exception:
                _scenario_lanes_disk_in_memory[log_id] = {}
        else:
            _scenario_lanes_disk_in_memory[log_id] = {}
    return _scenario_lanes_disk_in_memory[log_id]


def _scenario_lanes_disk_put(track_uuid: str, log_dir: Path, scenario_lanes: dict) -> None:
    """Write {uuid: {ts: lane_id}} entry for this track_uuid back to disk atomically.
    Multi-process safe enough: same (uuid, log) inputs produce the same output, so
    the last writer winning does not corrupt semantics; tmpfile + rename gives atomicity.
    """
    log_id = Path(log_dir).name
    store = _scenario_lanes_disk_load(log_dir)
    store[track_uuid] = {str(ts): (ls.id if ls is not None else None)
                         for ts, ls in scenario_lanes.items()}
    cache_dir = paths.GLOBAL_CACHE_PATH / log_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _scenario_lanes_disk_path(log_dir)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(store, f)
    tmp.replace(path)


@cache_manager.create_cache('get_scenario_lanes')
def get_scenario_lanes(track_uuid:str, log_dir:Path, avm=None)->dict[int,LaneSegment]:
    """Returns: scenario_lanes as a dict giving lane the object is in keyed by timestamp"""

    if not avm:
        avm = get_map(log_dir)

    # Disk cache hit: rebuild dict[ts -> LaneSegment] from stored lane ids in O(T) lookups.
    # Cross-process / cross-prompt: same (uuid, log_dir) -> immediate reuse.
    map_lane_dict = avm.vector_lane_segments
    cached = _scenario_lanes_disk_load(log_dir).get(track_uuid)
    if cached is not None:
        return {int(ts): (map_lane_dict[lid] if lid is not None else None)
                for ts, lid in cached.items()}

    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    angular_velocities, _ = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self')
    
    #Key lane segment id, value list of timestamps (associated with trajectory)
    lane_buckets:dict[int, list[int]] = {}

    #Put all points in lane buckets
    #While there exist unassigned points
        #Pop the bucket with the most points
        #Assign timestamps within bucket to popped lane 
        #Remove all points in popped bucket from other buckets

    for i in range(len(timestamps)):

        lane_segments = get_lane_segments(avm, traj[i])

        for ls in lane_segments:
            if ls.id not in lane_buckets:
                lane_buckets[ls.id] = [timestamps[i]]
            else:
                lane_buckets[ls.id].append(timestamps[i])
        
    scenario_lanes:dict[int, LaneSegment] = {}

    while len(lane_buckets) > 0:
        
        most_points = 0
        best_lane_id = None
        for lane_id, lane_timestamps in lane_buckets.items():

            if len(lane_timestamps) > most_points:
                most_points = len(lane_timestamps)
                best_lane_id = lane_id
            elif len(lane_timestamps) == most_points:
                # This often occurs if the objects starts or ends a log
                # at the end or start respectively of an intersection LaneSegment
                ls = map_lane_dict[lane_id]
                turn_direction = get_turn_direction(ls)
                angular_velocity = np.mean(angular_velocities[np.isin(timestamps, lane_timestamps)])

                if (turn_direction == 'left' and angular_velocity > 0.15) \
                or (turn_direction == 'right' and angular_velocity < -0.15) \
                or (turn_direction == 'straight' and -0.15 < angular_velocity < 0.15):
                    most_points = len(lane_timestamps)
                    best_lane_id = lane_id
            
        removed_timestamps = lane_buckets.pop(best_lane_id)
        for timestamp in removed_timestamps:
            scenario_lanes[timestamp] = map_lane_dict[best_lane_id]    

        for lane_id, lane_timestamps in list(lane_buckets.items()):
            remaining_timestamps = list(set(lane_timestamps).difference(removed_timestamps))
            if len(remaining_timestamps) == 0:
                lane_buckets.pop(lane_id)
            else:
                lane_buckets[lane_id] = remaining_timestamps

    for timestamp in timestamps:
        if timestamp not in scenario_lanes:
            scenario_lanes[timestamp] = None

    # Persist for cross-process / cross-prompt reuse. lane_id-only on-disk format
    # makes this independent of av2 SDK object versions.
    try:
        _scenario_lanes_disk_put(track_uuid, log_dir, scenario_lanes)
    except Exception:
        pass  # disk-cache failures are not fatal; in-memory cache still works

    return scenario_lanes


def get_road_side(ls:LaneSegment, log_dir, side:Literal['same','opposite'], avm=None) -> list[LaneSegment]:

    if not ls:
        return []

    if not avm:
        avm = get_map(log_dir)
    lane_dict = avm.vector_lane_segments
    map_lane_ids = set([ls.id for ls in lane_dict.values()])

    if ls.id not in map_lane_ids:
        return []

    try:
        road_side_cache = cache_manager.road_side_cache
        road_side_ids = road_side_cache[str(ls.id)][side]
        return [lane_dict[id] for id in road_side_ids]
    except: pass

    same_side_frontier = get_semantic_lane(ls, log_dir, avm=avm)

    same_side = []
    opposite_side = []

    while same_side_frontier:
        lane_segment = same_side_frontier.pop(0)
        same_side.append(lane_segment.id)

        if lane_segment.left_neighbor_id and lane_segment.left_neighbor_id in map_lane_ids:
            left_neighbor = lane_dict[lane_segment.left_neighbor_id]
            left_edge = lane_segment.left_lane_boundary.xyz[:,:2]
            right_edge = left_neighbor.right_lane_boundary.xyz[:,:2]
            edge_distance = np.linalg.norm(left_edge[0]-right_edge[0]) + np.linalg.norm(left_edge[-1]-right_edge[-1])
            if (left_neighbor.id not in opposite_side and left_neighbor.id not in same_side
            and edge_distance < .1):
                same_side_frontier.append(left_neighbor)
            elif left_neighbor.id not in opposite_side and left_neighbor.id not in same_side:
                opposite_side.append(left_neighbor.id)

        if lane_segment.right_neighbor_id and lane_segment.right_neighbor_id in map_lane_ids:
            right_neighbor = lane_dict[lane_segment.right_neighbor_id]
            right_edge = lane_segment.right_lane_boundary.xyz[:,:2]
            left_edge = right_neighbor.left_lane_boundary.xyz[:,:2]
            edge_distance = np.linalg.norm(left_edge[0]-right_edge[0]) + np.linalg.norm(left_edge[-1]-right_edge[-1])

            if (right_neighbor.id not in opposite_side and right_neighbor.id not in same_side
            and edge_distance < .1):
                same_side_frontier.append(right_neighbor)
            elif right_neighbor.id not in opposite_side and right_neighbor.id not in same_side:
                opposite_side.append(right_neighbor.id)

    if side == 'same':
        road_side = [lane_dict[lane_id] for lane_id in same_side]
    elif side == 'opposite':
        if opposite_side:
            road_side = get_road_side(lane_dict[opposite_side[0]], log_dir, side='same')
        else:
            road_side = []
    
    return road_side


# Geometric fallback for on_relative_side_of_road on divided (median) roads, where the opposing
# lanes are not map neighbors and lane topology (get_road_side) links nothing.
_OPPOSITE_DIR_COS_MAX = -0.5            # lane travel directions point opposite ways (>~120 deg apart)
_SAME_ROAD_MAX_DIST_M = 50.0           # the two agents are on the same stretch of road
_ACROSS_ROAD_GAP_RANGE_M = (1.5, 30.0)  # lateral gap: across a lane/median, yet still the same road


def _lane_travel_direction(lane_segment):
    """Unit 2D travel direction of a lane segment (boundary points run along travel)."""
    boundary = lane_segment.left_lane_boundary.xyz[:, :2]
    direction = boundary[-1] - boundary[0]
    length = np.linalg.norm(direction)
    return direction / length if length > 1e-6 else None


def _opposite_across_road(track_lane, related_lane, track_xy, related_xy):
    """True if related_lane runs opposite to track_lane and sits laterally across the road.

    Used on divided roads (median): the lanes must run anti-parallel, and the two agents must be
    on the same stretch of road but separated laterally (wide enough to be across a lane/median,
    close enough to still be the same road).
    """
    if track_lane is None or track_xy is None or related_xy is None:
        return False
    track_dir = _lane_travel_direction(track_lane)
    related_dir = _lane_travel_direction(related_lane)
    if track_dir is None or related_dir is None:
        return False
    if np.dot(track_dir, related_dir) > _OPPOSITE_DIR_COS_MAX:   # not anti-parallel
        return False

    offset = related_xy - track_xy
    if np.linalg.norm(offset) > _SAME_ROAD_MAX_DIST_M:
        return False
    lane_normal = np.array([-track_dir[1], track_dir[0]])        # perpendicular to lane travel
    lateral_gap = abs(float(np.dot(offset, lane_normal)))
    return _ACROSS_ROAD_GAP_RANGE_M[0] <= lateral_gap <= _ACROSS_ROAD_GAP_RANGE_M[1]


def get_semantic_lane(ls: LaneSegment, log_dir, avm=None) -> list[LaneSegment]:
    """Returns a list of lane segments that would make up a single 'lane' coloquailly.
    Finds all lane segments that are directionally forward and backward to the given lane
    segment."""

    if not ls:
        return []

    if not avm:
        avm = get_map(log_dir)
    lane_segments = avm.vector_lane_segments

    try:
        semantic_lanes = cache_manager.semantic_lane_cache[str(ls.id)]
        all_lanes = avm.vector_lane_segments
        return [all_lanes[ls_id] for ls_id in semantic_lanes]
    except:
        pass

    semantic_lane = [ls]

    if not ls.is_intersection or get_turn_direction(ls) == 'straight':
        predecessors = [ls]
        sucessors = [ls]
    else:
        return semantic_lane

    while predecessors:
        pred_ls = predecessors.pop()
        pred_direction = get_lane_orientation(pred_ls, avm)
        ppred_ids = pred_ls.predecessors
        
        most_likely_pred = None
        best_similarity = 0
        for ppred_id in ppred_ids:
            if ppred_id in lane_segments:
                ppred_ls = lane_segments[ppred_id]
                ppred_direction = get_lane_orientation(ppred_ls, avm)
                similarity = np.dot(ppred_direction, pred_direction)/(np.linalg.norm(ppred_direction)*np.linalg.norm(pred_direction))

                if ((not ppred_ls.is_intersection
                or get_turn_direction(lane_segments[ppred_id]) == 'straight') 
                and similarity > best_similarity):
                    best_similarity = similarity
                    most_likely_pred = ppred_ls

        if most_likely_pred and most_likely_pred not in semantic_lane:
            semantic_lane.append(most_likely_pred)
            predecessors.append(most_likely_pred)

    while sucessors:
        pred_ls = sucessors.pop()
        pred_direction = get_lane_orientation(pred_ls, avm)
        ppred_ids = pred_ls.successors
        
        most_likely_pred = None
        best_similarity = -np.inf
        for ppred_id in ppred_ids:
            if ppred_id in lane_segments:
                ppred_ls = lane_segments[ppred_id]
                ppred_direction = get_lane_orientation(ppred_ls, avm)
                similarity = np.dot(ppred_direction, pred_direction)/(np.linalg.norm(ppred_direction)*np.linalg.norm(pred_direction))

                if ((not ppred_ls.is_intersection
                or get_turn_direction(lane_segments[ppred_id]) == 'straight') 
                and similarity > best_similarity):
                    best_similarity = similarity
                    most_likely_pred = ppred_ls

        if most_likely_pred and most_likely_pred not in semantic_lane:
            semantic_lane.append(most_likely_pred)
            sucessors.append(most_likely_pred)
    
    return semantic_lane


def get_turn_direction(ls: LaneSegment):

    if not ls or not ls.is_intersection:
        return None

    start_direction = ls.right_lane_boundary.xyz[0,:2] - ls.left_lane_boundary.xyz[0,:2]
    end_direction = ls.right_lane_boundary.xyz[-1,:2] - ls.left_lane_boundary.xyz[-1,:2]

    start_angle = np.arctan2(start_direction[0], start_direction[1])
    end_angle = np.arctan2(end_direction[0], end_direction[1])

    angle_change = end_angle - start_angle

    if abs(angle_change) > np.pi:
        if angle_change > 0:
            angle_change -= 2*np.pi
        else:
            angle_change += 2*np.pi

    if angle_change > np.pi/6:
        return 'right'
    elif angle_change < -np.pi/6:
        return 'left'
    else:
        return 'straight'
    

def get_lane_orientation(ls: LaneSegment, avm: ArgoverseStaticMap) -> np.ndarray:
    "Returns orientation (as unit direction vectors) at the start and end of the LaneSegment"
    centerline = avm.get_lane_segment_centerline(ls.id)
    orientation  = centerline[-1] - centerline[0]
    orientation /= np.linalg.norm(orientation + 1e-8)
    return orientation   
        

def unwrap_func(decorated_func: Callable, n=1) -> Callable:
    """Get the original function from a decorated function."""

    unwrapped_func = decorated_func
    for _ in range(n):
        if hasattr(unwrapped_func, '__wrapped__'):
            unwrapped_func = unwrapped_func.__wrapped__
        else:
            break
    
    return unwrapped_func


def parallelize_uuids(
    func: Callable,
    all_uuids: list[str],
    *args,
    **kwargs
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parallelize UUID processing using Pathos ProcessingPool.
    
    Notes:
        - Pathos provides better serialization than standard multiprocessing
        - ProcessingPool.map() is already synchronous and will wait for completion
        - Pathos handles class methods and nested functions better than multiprocessing
    """
    func = unwrap_func(func)

    def worker_func(uuid: str) -> tuple[str, Any, Any]:
        """
        Worker function wrapper that maintains closure over func and its arguments.
        Pathos handles this closure better than standard multiprocessing.
        """
        result = func(uuid, *args, **kwargs)
        if not isinstance(result, tuple):
            result = (result, None)
        timestamps = result[0]
        related = result[1]
        
        return uuid, timestamps, related

    # Initialize the pool — shrink the pool size to match the workload to minimize worker startup cost.
    # And if UUID count is at or below _SEQ_THRESHOLD, bypass ProcessPool entirely (serial). This avoids
    # tiny-workload inefficiency such as 23s → 8s. cache_manager.num_processes acts as the upper bound.
    _SEQ_THRESHOLD = 4   # run serially if uuids are at or below this
    n_uuids = len(all_uuids)

    if n_uuids <= _SEQ_THRESHOLD:
        results = [worker_func(u) for u in all_uuids]
    else:
        num_processes = max(1, min(int(cache_manager.num_processes), n_uuids))
        with Pool(nodes=num_processes) as pool:
            results = pool.map(worker_func, all_uuids)

    # Process results
    uuid_dict = {}
    related_dict = {}

    for uuid, timestamps, related in results:
        if timestamps is not None:
            uuid_dict[uuid] = timestamps
            related_dict[uuid] = related

    return uuid_dict, related_dict
        

def _is_point_in_polygon_batch(points, polygon):
    """Vectorized ray-casting: returns (M,) bool for points (M, 2) vs polygon (N, 2).

    Same algorithm as `is_point_in_polygon`, just evaluated for all M points
    against each polygon edge in numpy. Used by atomic functions that test many
    points against the same polygon (e.g. near_intersection per-track trajectory).
    """
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    px1, py1 = float(polygon[0, 0]), float(polygon[0, 1])
    for i in range(1, n + 1):
        px2, py2 = float(polygon[i % n, 0]), float(polygon[i % n, 1])
        cond1 = y > min(py1, py2)
        cond2 = y <= max(py1, py2)
        cond3 = x <= max(px1, px2)
        if py1 != py2:
            xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
            cond4 = (px1 == px2) | (x <= xinters)
        else:
            # cond1 ^ cond2 is always False here (y > py and y <= py impossible),
            # so cond4's value is irrelevant — choose False to avoid arbitrary nan.
            cond4 = np.zeros_like(x, dtype=bool)
        inside = inside ^ (cond1 & cond2 & cond3 & cond4)
        px1, py1 = px2, py2
    return inside


def is_point_in_polygon(point, polygon):
    """
    Determine if a point is inside a polygon using the ray-casting algorithm.

    :param point: (x, y) coordinates of the point.
    :param polygon: List of (x, y) coordinates defining the polygon vertices.
    :return: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2

    return inside


def polygons_overlap(poly1, poly2):
    """
    Determine if two polygons overlap using the Separating Axis Theorem (SAT).

    Parameters:
    poly1, poly2: Nx2 numpy arrays where each row is a vertex (x,y)
                 First and last vertices should be the same
    visualize: bool, whether to show a visualization of the polygons

    Returns:
    bool: True if polygons overlap, False otherwise
    """
    def get_edges(polygon):
        # Get all edges of the polygon as vectors
        return [polygon[i+1] - polygon[i] for i in range(len(polygon)-1)]

    def get_normal(edge):
        # Get the normal vector to an edge
        return np.array([-edge[1], edge[0]])

    def project_polygon(polygon, axis):
        # Project all vertices onto an axis
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)

    def overlap_on_axis(min1, max1, min2, max2):
        # Check if projections overlap
        return (min1 <= max2 and min2 <= max1) \
                or (min1<=min2 and max1>=max2) \
                or (min2<=min1 and max2>=max1)

    # Get all edges from both polygons
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)

    # Test all normal vectors as potential separating axes
    for edge in edges1 + edges2:
        # Get the normal to the edge
        normal = get_normal(edge)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Project both polygons onto the normal
        min1, max1 = project_polygon(poly1, normal)
        min2, max2 = project_polygon(poly2, normal)

        # If we find a separating axis, the polygons don't overlap
        if not overlap_on_axis(min1, max1, min2, max2):
            return False

    # If we get here, no separating axis was found, so the polygons overlap
    return True


@cache_manager.create_cache('get_nth_pos_deriv')
def get_nth_pos_deriv(
    track_uuid, 
    n, 
    log_dir, 
    coordinate_frame=None,
    direction='forward') -> tuple[np.ndarray, list[int]]:

    """Returns the nth positional derivative of the track at all timestamps 
    with respect to city coordinates. """

    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_poses = get_ego_SE3(log_dir)

    # Filter the DataFrame
    cuboid_df = df[df['track_uuid'] == track_uuid]
    ego_coords = cuboid_df[['tx_m', 'ty_m', 'tz_m']].to_numpy()

    timestamps = cuboid_df['timestamp_ns'].to_numpy()
    # Vectorized ego -> city transform across all timestamps.
    # transform_from(p) = rotation @ p + translation
    rotations    = np.stack([ego_poses[int(ts)].rotation    for ts in timestamps]) if len(timestamps) else np.zeros((0,3,3))
    translations = np.stack([ego_poses[int(ts)].translation for ts in timestamps]) if len(timestamps) else np.zeros((0,3))
    city_coords  = np.einsum('tij,tj->ti', rotations, ego_coords) + translations

    #Very often, different cuboids are not seen by the ego vehicle at the same time.
    #Only the timestamps where both cuboids are observed are calculated.
    if type(coordinate_frame) != SE3 and coordinate_frame is not None and coordinate_frame != get_ego_uuid(log_dir):
        if coordinate_frame == 'self':
            coordinate_frame = track_uuid

        cf_df = df[df['track_uuid'] == coordinate_frame]
        cf_timestamps = cf_df['timestamp_ns'].to_numpy()

        new_timestamps = np.array(list(set(cf_timestamps).intersection(set(timestamps))))
        new_timestamps.sort(axis=0)

        city_coords = city_coords[np.isin(timestamps, new_timestamps)]
        timestamps = new_timestamps
        cf_df = cf_df[cf_df['timestamp_ns'].isin(timestamps)]
    
    INTERPOLATION_RATE = 1
    prev_deriv = np.copy(city_coords)
    next_deriv = np.zeros(prev_deriv.shape)
    for _ in range(n):
        next_deriv = np.zeros(prev_deriv.shape)
        if len(timestamps) == 1:
            break
        # Vectorized central difference. past_index/future_index per-row,
        # then broadcast over xyz.
        idx = np.arange(len(prev_deriv))
        past_index   = np.maximum(0, idx - INTERPOLATION_RATE)
        future_index = np.minimum(len(timestamps) - 1, idx + INTERPOLATION_RATE)
        dt = (timestamps[future_index] - timestamps[past_index]).astype(np.float64)
        next_deriv = 1e9 * (prev_deriv[future_index] - prev_deriv[past_index]) / dt[:, None]
        prev_deriv = next_deriv
    
    if len(timestamps) == 1:
        if n == 0:
            pos_deriv = prev_deriv
        else:
            pos_deriv = np.array([[0,0,0]], dtype=np.float64)
    elif len(timestamps) == 0:
        return prev_deriv, [int(timestamp) for timestamp in timestamps]
    else:
        pos_deriv = scipy.ndimage.median_filter(prev_deriv, size=min(7,len(prev_deriv)), mode='nearest', axes=0) 

    if type(coordinate_frame) == SE3:
        pos_deriv = (coordinate_frame.transform_from(pos_deriv.T)).T
    elif coordinate_frame == get_ego_uuid(log_dir):
        # Vectorized city -> ego: city_to_ego = ego_pose.inverse(),
        # so inv_R = R^T, inv_t = -R^T @ t. For n != 0, only the rotation matters.
        ego_R = np.stack([ego_poses[int(ts)].rotation    for ts in timestamps])
        ego_t = np.stack([ego_poses[int(ts)].translation for ts in timestamps])
        inv_R = np.transpose(ego_R, (0, 2, 1))
        if n == 0:
            inv_t = -np.einsum('tij,tj->ti', inv_R, ego_t)
            pos_deriv = np.einsum('tij,tj->ti', inv_R, pos_deriv) + inv_t
        else:
            #Velocity/acceleration/jerk vectors only need to be rotated
            pos_deriv = np.einsum('tij,tj->ti', inv_R, pos_deriv)
    elif coordinate_frame is not None:
        cf_df = df[df['track_uuid'] == coordinate_frame]
        if cf_df.empty:
            print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')
            print('Returning answer in city coordinates')
            return pos_deriv, [int(timestamp) for timestamp in timestamps]

        cf_df = cf_df[cf_df['timestamp_ns'].isin(timestamps)]
        cf_list = CuboidList.from_dataframe(cf_df)

        # Vectorized city -> self via city_to_self = (cf.dst_SE3_object).inverse().compose(ego_pose.inverse())
        # i.e. city_to_self.rotation    = inv_cf_R @ inv_ego_R
        #      city_to_self.translation = inv_cf_R @ inv_ego_t + inv_cf_t
        ego_R = np.stack([ego_poses[int(ts)].rotation    for ts in timestamps])
        ego_t = np.stack([ego_poses[int(ts)].translation for ts in timestamps])
        cf_R  = np.stack([cf_list[i].dst_SE3_object.rotation    for i in range(len(cf_list))])
        cf_t  = np.stack([cf_list[i].dst_SE3_object.translation for i in range(len(cf_list))])

        inv_ego_R = np.transpose(ego_R, (0, 2, 1))
        inv_ego_t = -np.einsum('tij,tj->ti', inv_ego_R, ego_t)
        inv_cf_R  = np.transpose(cf_R, (0, 2, 1))
        inv_cf_t  = -np.einsum('tij,tj->ti', inv_cf_R, cf_t)

        cs_R = np.einsum('tij,tjk->tik', inv_cf_R, inv_ego_R)
        if n == 0:
            cs_t = np.einsum('tij,tj->ti', inv_cf_R, inv_ego_t) + inv_cf_t
            pos_deriv = np.einsum('tij,tj->ti', cs_R, pos_deriv) + cs_t
        else:
            #Velocity/acceleration/jerk vectors only need to be rotated
            pos_deriv = np.einsum('tij,tj->ti', cs_R, pos_deriv)

    if direction == 'left':
        rot_mat = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    elif direction == 'right':
        rot_mat = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    elif direction == 'backward':
        rot_mat = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    else:
        rot_mat = np.eye(3)

    pos_deriv = (rot_mat @ pos_deriv.T).T

    return pos_deriv, [int(timestamp) for timestamp in timestamps]


def get_nth_radial_deriv(track_uuid, n, log_dir, 
    coordinate_frame=None)->tuple[np.ndarray, np.ndarray]:

    relative_pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=coordinate_frame)
    
    distance = np.linalg.norm(relative_pos, axis=1)
    radial_deriv = distance
    for i in range(n):
        if len(radial_deriv) > 1:
            radial_deriv = np.gradient(radial_deriv)
        else:
            radial_deriv = np.array([0])

    return radial_deriv, timestamps


@cache_manager.create_cache('get_nth_yaw_deriv')
def get_nth_yaw_deriv(track_uuid, n, log_dir, coordinate_frame=None, in_degrees=False):
    """Returns the nth angular derivative of the track at all timestamps 
    with respect to the given coordinate frame. The default coordinate frame is city.
    The returned angle is yaw measured from the x-axis of the track coordinate frame to the x-axis
    of the source coordinate frame"""

    df = read_feather(log_dir / 'sm_annotations.feather')
    ego_poses = get_ego_SE3(log_dir)

    # Filter the DataFrame
    cuboid_df = df[df['track_uuid'] == track_uuid]
    timestamps = cuboid_df['timestamp_ns'].to_numpy()

    # Vectorized self -> city rotation: ego_pose.compose(self_to_ego) only needs rotation
    # (yaw = rotvec).z is invariant to translation. compose.R = ego_R @ self_to_ego_R.
    # Bypass CuboidList.from_dataframe (heavy) by reading quaternions directly.
    quats = cuboid_df[['qw','qx','qy','qz']].to_numpy() if len(cuboid_df) else np.zeros((0,4))
    self_to_ego_R = quat_to_mat(quats) if len(quats) else np.zeros((0,3,3))
    ego_R         = np.stack([ego_poses[int(ts)].rotation for ts in timestamps]) if len(timestamps)  else np.zeros((0,3,3))
    self_to_city_R = np.einsum('tij,tjk->tik', ego_R, self_to_ego_R)

    #Very often, different cuboids are not seen by the ego vehicle at the same time.
    #Only the timestamps where both cuboids are observed are calculated.
    if type(coordinate_frame) != SE3 and coordinate_frame is not None and coordinate_frame != get_ego_uuid(log_dir):
        if coordinate_frame == 'self':
            coordinate_frame = track_uuid

        cf_df = df[df['track_uuid'] == coordinate_frame]
        cf_timestamps = cf_df['timestamp_ns'].to_numpy()

        if cf_df.empty:
            print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')
            print('Returning answer in city coordinates')
        else:
            new_timestamps = np.array(list(set(cf_timestamps).intersection(set(timestamps))))
            new_timestamps.sort(axis=0)

            filtered_timestamps = np.isin(timestamps, new_timestamps)

            # Convert mask to indices
            filtered_indices = np.where(filtered_timestamps)[0]

            # Index the rotation stack and ego_R together so downstream branches stay aligned
            self_to_city_R = self_to_city_R[filtered_indices]
            ego_R          = ego_R[filtered_indices]
            timestamps = new_timestamps

    # Vectorized: Rotation.from_matrix supports batched (N, 3, 3) input.
    if len(self_to_city_R):
        city_yaws = Rotation.from_matrix(self_to_city_R).as_rotvec()
    else:
        city_yaws = np.zeros((0, 3))

    INTERPOLATION_RATE = 1
    prev_deriv = np.copy(city_yaws)
    next_deriv = np.zeros(prev_deriv.shape)
    for j in range(n):
        next_deriv = np.zeros(prev_deriv.shape)
        if len(timestamps) == 1:
            break
        # Vectorized central difference with j==0 angle-wrap (|diff| > pi -> ±2pi).
        idx = np.arange(len(prev_deriv))
        past_index   = np.maximum(0, idx - INTERPOLATION_RATE)
        future_index = np.minimum(len(prev_deriv) - 1, idx + INTERPOLATION_RATE)
        difference = prev_deriv[future_index] - prev_deriv[past_index]
        if j == 0:
            difference = np.where(difference >  np.pi, difference - 2*np.pi, difference)
            difference = np.where(difference < -np.pi, difference + 2*np.pi, difference)
        dt = (timestamps[future_index] - timestamps[past_index]).astype(np.float64)
        next_deriv = 1e9 * difference / dt[:, None]
        prev_deriv = next_deriv

    cf_angles = np.copy(prev_deriv)

    if n == 0 and coordinate_frame == get_ego_uuid(log_dir):
        # Vectorized: rotvec -> matrix -> apply city_to_ego.R = ego_R^T -> back to rotvec
        inv_ego_R = np.transpose(ego_R, (0, 2, 1))
        prev_R    = Rotation.from_rotvec(prev_deriv).as_matrix() if len(prev_deriv) else np.zeros((0,3,3))
        new_R     = np.einsum('tij,tjk->tik', inv_ego_R, prev_R)
        cf_angles = Rotation.from_matrix(new_R).as_rotvec() if len(new_R) else np.zeros((0,3))
    elif n == 0 and coordinate_frame is not None and type(coordinate_frame) != SE3:
        cf_df = df[df['track_uuid'] == coordinate_frame]
        if not cf_df.empty:
            # Bypass CuboidList.from_dataframe via direct quaternion -> matrix.
            # Match the legacy slicing `cf_list[i] for i in range(len(prev_deriv))`:
            # original only indexed the first len(prev_deriv) rows of cf_list, so we
            # slice the quaternion matrix the same way (cf_df may have more rows than
            # the post-intersection `timestamps`/`ego_R`).
            cf_quats = cf_df[['qw','qx','qy','qz']].to_numpy()
            cf_R = quat_to_mat(cf_quats)[:len(prev_deriv)]
            inv_ego_R = np.transpose(ego_R, (0, 2, 1))
            inv_cf_R  = np.transpose(cf_R, (0, 2, 1))
            # city_to_obj = ego_to_obj.compose(city_to_ego); .rotation = inv_cf_R @ inv_ego_R
            city_to_obj_R = np.einsum('tij,tjk->tik', inv_cf_R, inv_ego_R)
            prev_R = Rotation.from_rotvec(prev_deriv).as_matrix() if len(prev_deriv) else np.zeros((0,3,3))
            new_R  = np.einsum('tij,tjk->tik', city_to_obj_R, prev_R)
            cf_angles = Rotation.from_matrix(new_R).as_rotvec() if len(new_R) else np.zeros((0,3))
    elif n == 0 and type(coordinate_frame) == SE3:
        prev_R = Rotation.from_rotvec(prev_deriv).as_matrix() if len(prev_deriv) else np.zeros((0,3,3))
        new_R  = np.einsum('ij,tjk->tik', coordinate_frame.rotation, prev_R)
        cf_angles = Rotation.from_matrix(new_R).as_rotvec() if len(new_R) else np.zeros((0,3))
    elif n==0 and coordinate_frame is not None:
        print('Coordinate frame must be None, \'ego\', \'self\', track_uuid, or city to coordinate frame SE3 object.')

    if in_degrees: 
        cf_angles = np.rad2deg(cf_angles)

    return cf_angles[:,2], [int(timestamp) for timestamp in timestamps]


def get_dataset(log_dir):
    """"""

    log_dir = Path(log_dir)
    if log_dir.stem in TRAIN+VAL+TEST:
        return 'AV2'
    #TODO: Add checking to make sure log_id is in NuScenes training or val split
    else:
        return 'NUSCENES'

def get_log_split(log_dir:Union[str,Path]):
    """Returns the AV2 sensor split for the given log_id or log_dir"""

    log_dir = Path(log_dir)
    if log_dir.stem in VAL:
        split = 'val'
    elif log_dir.stem in TEST:
        split = 'test'
    elif log_dir.stem in TRAIN:
        split = 'train'
    #TODO: Add better checking
    else:
        split = 'nuprompt_val'

    return split


@cache_manager.create_cache('get_map')
def get_map(log_dir: Path):

    log_dir = Path(log_dir)
    try:
        avm = ArgoverseStaticMap.from_map_dir(log_dir / 'map', build_raster=True)
    except:
        split = get_log_split(log_dir)
        avm = ArgoverseStaticMap.from_map_dir(paths.AV2_DATA_DIR / split / log_dir.name / 'map', build_raster=True)

    return avm


# Median classifier thresholds — validated on 17 holes across val logs
# 91aa/cae5/f6cc/96dd with 100% accuracy. Hardcoded; not LLM-tunable.
MEDIAN_NARROW_INSCRIBED_M = 2.5   # a narrow median has width ≤ 5m
MEDIAN_WIDE_INSCRIBED_M   = 8.0   # a wide boulevard median has width ≤ 16m
MEDIAN_WIDE_ASPECT        = 3.0   # in the wide case it must be elongated (length/width)


def _is_median_hole(hole: _ShPolygon) -> bool:
    """Treat a hole as a median if it is narrow, or if it is wide but sufficiently elongated."""
    inscribed = _sh_polylabel(hole, tolerance=0.3).distance(hole.boundary)
    if inscribed <= MEDIAN_NARROW_INSCRIBED_M:
        return True
    if inscribed <= MEDIAN_WIDE_INSCRIBED_M:
        mbr = hole.minimum_rotated_rectangle
        cs = list(mbr.exterior.coords)
        edges = sorted(_math.dist(cs[i], cs[i + 1]) for i in range(4))
        if edges[0] > 0 and edges[2] / edges[0] >= MEDIAN_WIDE_ASPECT:
            return True
    return False


@cache_manager.create_cache('get_median_polygons')
def get_median_polygons(log_dir: Path) -> list:
    """Return all median polygons (shapely Polygon) of the log. Computed only once per log and cached.

    Takes the unary_union of the drivable areas, then extracts interior holes, keeping only narrow or elongated holes.
    """
    avm = get_map(log_dir)
    polys = []
    for da in avm.vector_drivable_areas.values():
        try:
            xy = da.xyz[:, :2]
        except Exception:
            continue
        if len(xy) >= 3:
            polys.append(_ShPolygon(xy))
    if not polys:
        return []
    union = _sh_unary_union(polys)
    parts = list(union.geoms) if union.geom_type == 'MultiPolygon' else [union]
    holes = [_ShPolygon(r) for part in parts for r in part.interiors]
    return [h for h in holes if _is_median_hole(h)]


@cache_manager.create_cache('get_ego_SE3')
def get_ego_SE3(log_dir:Path):
    """Returns list of ego_to_city SE3 transformation matrices"""

    log_dir = Path(log_dir)
    try:
        ego_poses = read_city_SE3_ego(log_dir)
    except:
        split = get_log_split(log_dir)
        ego_poses = read_city_SE3_ego(paths.AV2_DATA_DIR / split / log_dir.name)

    return ego_poses


def dilate_convex_polygon(points, distance):
    """
    Dilates the perimeter of a convex polygon specified in clockwise order by a given distance.
    
    Args:
        points (numpy.ndarray): Nx2 array of (x, y) coordinates representing the vertices of the convex polygon
                                in counterclockwise order. The first and last points are identical.
        distance (float): Distance to dilate the polygon perimeter. Positive for outward, negative for inward.

    Returns:
        numpy.ndarray: Nx2 array of (x, y) coordinates representing the dilated polygon vertices.
                       The first and last points will also be identical.
    """
    def normalize(v):
        """Normalize a vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    # Ensure counterclockwise winding for outward dilation
    shoelace = sum((points[(i+1)%len(points)][0] - points[i][0]) * (points[(i+1)%len(points)][1] + points[i][1]) for i in range(len(points)-1))
    if shoelace > 0:  # clockwise, flip to counterclockwise
        points = points[::-1]

    n = len(points)  # Account for duplicate closing point
    dilated_points = []

    for i in range(1,n):
        # Current, previous, and next points
        prev_point = points[i - 1]  # Previous vertex (wrap around for first vertex)
        curr_point = points[i]     # Current vertex
        next_point = points[(i + 1) % (n-1)]  # Next vertex (wrap around for last vertex)

        # Edge vectors
        edge1 = normalize(curr_point - prev_point)  # Edge vector from prev to curr
        edge2 = normalize(next_point - curr_point)  # Edge vector from curr to next

        # Perpendicular vectors to edges (flipped for clockwise order)
        perp1 = np.array([edge1[1], -edge1[0]])  # Rotate -90 degrees
        perp2 = np.array([edge2[1], -edge2[0]])  # Rotate -90 degrees

        # Average of perpendiculars (to find outward bisector direction)
        bisector = normalize(perp1 + perp2)

        # Avoid division by zero or near-zero cases
        dot_product = np.dot(bisector, perp1)
        if abs(dot_product) < 1e-10:  # Small threshold for numerical stability
            displacement = distance * bisector  # Fallback: scale bisector direction
        else:
            displacement = distance / dot_product * bisector

        # Compute the new vertex
        new_point = curr_point + displacement
        dilated_points.append(new_point)

    # Add the first point to the end to close the polygon
    dilated_points.append(dilated_points[0])
    return np.array(dilated_points)
    

@cache_manager.create_cache('get_cuboid_from_uuid')
def get_cuboid_from_uuid(track_uuid, log_dir, timestamp = None):
    df = read_feather(log_dir / 'sm_annotations.feather')
    
    track_df = df[df["track_uuid"] == track_uuid]

    if timestamp:
        track_df = track_df[track_df["timestamp_ns"] == timestamp]
        if track_df.empty:
            return None

    track_cuboids = CuboidList.from_dataframe(track_df)
    
    return track_cuboids[0]


@cache_manager.create_cache('to_scenario_dict')
def to_scenario_dict(object_datastructure, log_dir)->dict:

    if isinstance(object_datastructure, dict):
        object_dict = deepcopy(object_datastructure)
    elif isinstance(object_datastructure, list) or isinstance(object_datastructure, np.ndarray):
        object_dict = {uuid: unwrap_func(get_object)(uuid, log_dir) for uuid in object_datastructure}
    elif isinstance(object_datastructure, str):
        object_dict = {object_datastructure: unwrap_func(get_object)(object_datastructure, log_dir)}
    elif isinstance(object_datastructure, int):
        timestamp = object_datastructure
        df = read_feather(log_dir / 'sm_annotations.feather')
        timestamp_df = df[df['timestamp_ns'] == timestamp]

        if timestamp_df.empty:
            print(f'Timestamp {timestamp} not found in annotations')

        object_dict = {track_uuid: [timestamp] for track_uuid in timestamp_df['track_uuid'].unique()}
    else:
        print(f'Provided object, {object_datastructure}, of type {type(object_datastructure)}, must be a track_uuid, list[track_uuid], \
              timestamp, or dict[timestamp:list[timestamp]]')
        print('Comparing to all objects in the log.')

        df = read_feather(log_dir / 'sm_annotations.feather')
        all_uuids = df['track_uuid'].unique()
        object_dict, _ = parallelize_uuids(get_object, all_uuids, log_dir)
    
    return object_dict


def cuboid_distance(cuboid1:Union[str, Cuboid], cuboid2:Union[str, Cuboid], log_dir, timestamp=None) -> float:
    """Returns the minimum distance between two objects at the given timestamp. Timestamp is not required
    if the given objects are single cuboids."""

    if not isinstance(cuboid1, Cuboid):
        cuboid1 = get_cuboid_from_uuid(cuboid1, log_dir, timestamp=timestamp)
    if not isinstance(cuboid2, Cuboid):
        cuboid2 = get_cuboid_from_uuid(cuboid2, log_dir, timestamp=timestamp)

    c1_verts = cuboid1.vertices_m
    c2_verts = cuboid2.vertices_m

    rect1 = np.array([c1_verts[2],c1_verts[6],c1_verts[7],c1_verts[3],c1_verts[2]])[:,:2]
    rect2 = np.array([c2_verts[2],c2_verts[6],c2_verts[7],c2_verts[3],c2_verts[2]])[:,:2]

    distance = min_distance_between_rectangles(rect1, rect2)

    return distance


@cache_manager.create_cache('_yaw_fallback_posx_in_city')
def _yaw_fallback_posx_in_city(track_uuid: str, log_dir, timestamps) -> np.ndarray:
    """City-frame position of the cuboid's local +x tip at each timestamp.

    Equivalent to:
        for ts in timestamps:
            cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=ts)
            ego_to_city[ts].compose(cuboid.dst_SE3_object).transform_from([1,0,0])

    Pulled out of `yaw_fallback_dirs` so the cache key is (uuid, log_dir,
    timestamps) only — independent of the caller's `track_pos_in_city` array,
    which made the original signature uncacheable in practice.
    """
    df = read_feather(log_dir / 'sm_annotations.feather')
    ts_list = [int(t) for t in timestamps]
    sub = (df[df['track_uuid'] == track_uuid]
              .set_index('timestamp_ns')
              .reindex(ts_list)
              .reset_index())
    self_R = quat_to_mat(sub[['qw','qx','qy','qz']].to_numpy())   # (T, 3, 3): self -> ego
    self_t = sub[['tx_m','ty_m','tz_m']].to_numpy()               # (T, 3)
    ego_poses = get_ego_SE3(log_dir)
    ego_R = np.stack([ego_poses[ts].rotation    for ts in ts_list])  # (T, 3, 3): ego -> city
    ego_t = np.stack([ego_poses[ts].translation for ts in ts_list])  # (T, 3)
    # compose.R @ [1,0,0] + compose.t == (ego_R @ self_R)[:,:,0] + ego_R @ self_t + ego_t
    composed_R_col0 = np.einsum('tij,tjk->tik', ego_R, self_R)[:, :, 0]
    composed_t      = np.einsum('tij,tj->ti', ego_R, self_t) + ego_t
    return composed_R_col0 + composed_t


def yaw_fallback_dirs(track_uuid: str, log_dir, track_pos_in_city, timestamps) -> np.ndarray:
    """City-frame forward direction at each timestamp via the cuboid-yaw fallback.

    Equivalent to running the original per-ts fallback block in
    heading_in_relative_direction_to:
        for ts in timestamps:
            cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=ts)
            posx = ego_to_city[ts].compose(cuboid.dst_SE3_object).transform_from([1,0,0])
            dir  = posx - track_pos_in_city[ts]

    Vectorized via direct quaternion -> rotation + batched SE3 compose; one
    DataFrame filter total instead of one per timestamp.

    Args:
        track_uuid: track to look up cuboid yaws for.
        log_dir: scenario log dir (read_feather is cached).
        track_pos_in_city: (T, 3) city-frame positions of `track_uuid` at the
            given timestamps (already produced by get_nth_pos_deriv(...,0,log_dir)).
        timestamps: iterable of T timestamp_ns values matching track_pos_in_city.

    Returns:
        (T, 3) city-frame forward direction vectors.
    """
    posx = _yaw_fallback_posx_in_city(track_uuid, log_dir, timestamps)
    return posx - np.asarray(track_pos_in_city)


def get_cuboid_vertices_batch(track_uuid: str, log_dir, timestamps) -> np.ndarray:
    """Return (T, 8, 3) vertices_m for a track across many timestamps.

    Replaces per-ts get_cuboid_from_uuid (one DataFrame filter + CuboidList construction
    per call) with one DataFrame filter + one CuboidList for all timestamps. Used by
    atomic functions that need each per-timestamp cuboid (vertices_m or rect).

    Row order matches `timestamps` via set_index('timestamp_ns').reindex(...).
    """
    ts_list = [int(t) for t in timestamps]
    if not ts_list:
        return np.zeros((0, 8, 3))

    df = read_feather(log_dir / 'sm_annotations.feather')
    sub = (df[(df['track_uuid'] == track_uuid) & (df['timestamp_ns'].isin(set(ts_list)))]
              .set_index('timestamp_ns')
              .reindex(ts_list)
              .reset_index())
    cl = CuboidList.from_dataframe(sub)
    verts = np.empty((len(cl), 8, 3))
    for i in range(len(cl)):
        verts[i] = cl[i].vertices_m
    return verts


def cuboid_distance_batch(track_uuid: str, candidate_uuid: str, log_dir, timestamps) -> np.ndarray:
    """Batched cuboid_distance for many timestamps.

    Returns a (T,) float array where entry t is the same as
    cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp=timestamps[t]).

    Per-timestamp `cuboid_distance` performs two DataFrame filters + two
    CuboidList constructions every call. Here we filter the annotation DataFrame
    once per uuid and build vertices in a single batch, then loop only the cheap
    rect/min_distance step. Used by atomic functions to replace inner loops that
    call cuboid_distance once per timestamp.
    """
    timestamps_list = [int(ts) for ts in timestamps]
    if not timestamps_list:
        return np.zeros(0)

    df = read_feather(log_dir / 'sm_annotations.feather')
    ts_set = set(timestamps_list)

    def _rects_for(uuid):
        sub = df[(df['track_uuid'] == uuid) & (df['timestamp_ns'].isin(ts_set))]
        # Reorder rows to match `timestamps_list`. Missing timestamps -> NaN row,
        # which will produce NaN distance later (matches per-call behaviour where
        # get_cuboid_from_uuid returns None and crashes; callers must pass valid ts).
        sub = sub.set_index('timestamp_ns').reindex(timestamps_list).reset_index()
        cl = CuboidList.from_dataframe(sub)
        rects = np.empty((len(cl), 5, 2))
        for i in range(len(cl)):
            v = cl[i].vertices_m  # (8, 3)
            rects[i] = np.array([v[2], v[6], v[7], v[3], v[2]])[:, :2]
        return rects

    r1 = _rects_for(track_uuid)
    r2 = _rects_for(candidate_uuid)
    out = np.empty(len(timestamps_list))
    for i in range(len(timestamps_list)):
        out[i] = min_distance_between_rectangles(r1[i], r2[i])
    return out


def min_distance_between_rectangles(rect1, rect2):
    """
    Calculate the minimum distance between two rectangles.

    Args:
        rect1: np.array shape (5, 2) - first rectangle (counter-clockwise)
        rect2: np.array shape (5, 2) - second rectangle (counter-clockwise)

    Returns:
        float: Minimum distance between rectangles. Returns 0 if overlapping.
    """
    rect1 = np.asarray(rect1, dtype=np.float64)
    rect2 = np.asarray(rect2, dtype=np.float64)

    # Check for overlap
    if polygons_overlap(rect1, rect2):
        return 0.0

    # Vectorized: compute every (vertex of one rect) -> (edge of the other rect) distance
    # in two broadcast calls. The original double for-loop computes the same 32 unique
    # vertex-edge pairs (twice each as a1/a2), then takes the min.
    v1 = rect1[:-1]                # (4, 2)  rect1 vertices
    v2 = rect2[:-1]                # (4, 2)  rect2 vertices
    e1_a, e1_b = rect1[:-1], rect1[1:]   # (4, 2) each, rect1 edges
    e2_a, e2_b = rect2[:-1], rect2[1:]

    # broadcast: (4, 1, 2) vs (1, 4, 2) -> (4, 4) distance grids
    d_v1_e2 = _point_to_segment_distance_batch(v1[:, None, :], e2_a[None, :, :], e2_b[None, :, :])
    d_v2_e1 = _point_to_segment_distance_batch(v2[:, None, :], e1_a[None, :, :], e1_b[None, :, :])
    return float(min(d_v1_e2.min(), d_v2_e1.min()))


def point_to_segment_distance(p, a, b):
    """Compute distance from point p to segment ab."""
    ap = p - a
    ab = b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _point_to_segment_distance_batch(p, a, b):
    """Broadcast version of point_to_segment_distance for shapes (..., 2) -> (...).

    Used by min_distance_between_rectangles to compute every (vertex, edge) pair
    in a single numpy call. Kept private because the broadcast wrapper adds
    overhead in scalar-only callsites that the public function still serves.
    """
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ap = p - a
    ab = b - a
    ab_dot_ab = np.sum(ab * ab, axis=-1)
    ap_dot_ab = np.sum(ap * ab, axis=-1)
    # degenerate (zero-length) segment: treat the segment as point a.
    safe_ab = np.where(ab_dot_ab == 0.0, 1.0, ab_dot_ab)
    t = np.clip(ap_dot_ab / safe_ab, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)


@composable
def near_ego(
    track_uuid:Union[list,dict], 
    log_dir:Path,
    distance_thresh:float=50)->dict:
    """
    Returns timestamps where the object is near the ego vehicle
    """

    pos, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=get_ego_uuid(log_dir))
    near_ego_timestamps = timestamps[np.linalg.norm(pos) < distance_thresh]

    return near_ego_timestamps


def filter_by_ego_distance(scenario, log_dir, max_distance=50):
    
    ego_uuid = get_ego_uuid(log_dir)

    for track_uuid, related_objects in list(scenario.items()):

        pos, log_timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=ego_uuid)
        within_distance = np.linalg.norm(pos, axis=1) < max_distance
        valid_timestamps = np.array(log_timestamps)[within_distance]
        
        if isinstance(related_objects, dict):
            related_objects = scenario_at_timestamps(related_objects, valid_timestamps)
        else:
            referred_timestamps = []
            for timestamp in related_objects:
                if timestamp in valid_timestamps:
                    referred_timestamps.append(timestamp)

            scenario[track_uuid] = referred_timestamps


@cache_manager.create_cache('post_process_scenario')
def post_process_scenario(scenario, log_dir) -> dict:
    """
    1. Filter out referred objects that are only referred for 1 timestamp (likely noise)
    2. Filter out relationships (referred and related objects) with a relative distance of over 50m. 
    3. If a referred object is referred for less than 1.5s, expand the referred timestamps symmetrically in both directions to hit 1.5s.

    Return False if scenario was removed or filtered down to an empty set. Return true if there still exist referred objects with timestamps.
    """

    remove_empty_branches(scenario)
    if dict_empty(scenario):
        return True

    filter_by_relationship_distance(scenario, log_dir, max_distance=50)
    # dilate FIRST: expands single-timestamp predictions to 1.5s so they survive
    # the subsequent length filter (otherwise filter_by_length removes them before
    # dilate can rescue them).
    dilate_timestamps(scenario, log_dir, min_timespan_s=1.5)
    filter_by_length(scenario, min_timesteps=2)

    if dict_empty(scenario):
        return False
    else:
        return True


def filter_by_length(scenario, min_timesteps=2):

    for track_uuid, related_objects in list(scenario.items()):
        if isinstance(related_objects, list) or isinstance(related_objects, set):
            if len(related_objects) < min_timesteps:
                scenario.pop(track_uuid)
        else:
            filter_by_length(related_objects, min_timesteps)


def filter_by_relationship_distance(scenario, log_dir, max_distance=50):
    
    for track_uuid, related_objects in list(scenario.items()):
        if isinstance(related_objects, dict):

            for related_uuid, related_grandchildren in list(related_objects.items()):

                if isinstance(related_grandchildren, dict):
                    filter_by_relationship_distance(related_objects, log_dir, max_distance)

                traj, timestamps = get_nth_pos_deriv(related_uuid, 0, log_dir, coordinate_frame=track_uuid)
                related_timestamps = get_scenario_timestamps(related_grandchildren)
                related_position = traj[np.isin(timestamps, related_timestamps)]
                related_distance = np.linalg.norm(related_position, axis=1)

                if not np.any(related_distance < max_distance):
                    scenario[track_uuid].pop(related_uuid)


def dilate_timestamps(scenario, log_dir, min_timespan_s:float=1.5, log_df = None):
    """Adds additional timestamps (symetrically) to any referred tracks that are under 1.5s seconds long to match RefAV annotation procedure."""


    if log_df is None:
        log_df = read_feather(log_dir / 'sm_annotations.feather')

    timestamps = sorted(log_df['timestamp_ns'].unique())
    timestep_s = 1E-9*(timestamps[1]-timestamps[0])
    min_length = round(min_timespan_s/timestep_s)

    for track_uuid, related_objects in scenario.items():
        if isinstance(related_objects, dict):
            dilate_timestamps(related_objects, log_dir, min_timespan_s, log_df=log_df)

        elif isinstance(related_objects, list):
            referred_timestamps = sorted(related_objects)
            track_av2_timestamps =  np.array(sorted(log_df.loc[log_df['track_uuid'] == track_uuid, 'timestamp_ns'].unique()))

            referred_indices = np.isin(track_av2_timestamps, referred_timestamps)

            index = 0
            while index < len(track_av2_timestamps):
                #traverse the array from left to right
                #if a 1 is reached, the left pointer stops and the right keeps going until it hits a 0
                    #if the right reaches a 0, calculate the distance between left and right
                    #if this distance < 15, update the left and right pointer indices to 1

                if referred_indices[index] == 0:
                    index += 1
                else:
                    left = index - 1
                    right = index + 1

                    while right < len(referred_indices) and referred_indices[right] == 1:
                        right += 1

                    len_time_seg = (right-left) - 1
                    dilation_size = (min_length - len_time_seg)//2

                    for _ in range(dilation_size):
                        if left >= 0:
                            referred_indices[left] = 1
                            left -= 1
                        if right < len(referred_indices):
                            referred_indices[right] = 1
                            right += 1

                    index = right

            scenario[track_uuid] = list(track_av2_timestamps[referred_indices])


def filter_by_roi(scenario, log_dir):
    """
    Remove scenarios that never have a referred object in side the region of interest.
    Keep scenarios that ever have a referred object in the region of interest as-is.
    """

    filtered_scenario = in_region_of_interest(scenario, log_dir)

    if dict_empty(filtered_scenario):
        if not dict_empty(scenario):
            print('Scenario has referred objects, but none within the region of interest.')

        return filtered_scenario
    else:
        return scenario


def swap_keys_and_listed_values(dict:dict[float,list])->dict[float,list]:
    
    swapped_dict = {}
    for key, timestamp_list in dict.items():
        for timestamp in timestamp_list:
            if timestamp not in swapped_dict:
                swapped_dict[timestamp] = []
            swapped_dict[timestamp].append(key)

    return swapped_dict



def dict_empty(d:dict):
    if len(d) == 0:
        return True

    for value in d.values():
        if isinstance(value, list) and len(value) > 0:
            return False

        if isinstance(value, dict) and not dict_empty(value):
            return False
        
    return True


@composable_relational
def at_stop_sign_(track_uuid, stop_sign_uuids, log_dir, forward_thresh=10) -> tuple[list, dict[str,list]]:
    RIGHT_THRESH = 7 #m

    stop_sign_timestamps = []
    stop_signs = {}
    
    track_lanes = get_scenario_lanes(track_uuid, log_dir)

    for stop_sign_id in stop_sign_uuids:
        pos, _ = get_nth_pos_deriv(track_uuid, 0, log_dir, coordinate_frame=stop_sign_id)
        yaws, timestamps = get_nth_yaw_deriv(track_uuid, 0, log_dir, coordinate_frame=stop_sign_id, in_degrees=True)
        for i in range(len(timestamps)):
            if (-1<pos[i,0]<forward_thresh and -RIGHT_THRESH<pos[i,1]<0 
            and track_lanes.get(timestamps[i],None) 
            and stop_sign_lane(stop_sign_id, log_dir) 
            and track_lanes[timestamps[i]].id == stop_sign_lane(stop_sign_id, log_dir).id
            and (yaws[i] >= 90 or yaws[i] <= -90)):

                if stop_sign_id not in stop_signs:
                    stop_signs[stop_sign_id] = []
                stop_signs[stop_sign_id].append(timestamps[i])
            
                if timestamps[i] not in stop_sign_timestamps:
                    stop_sign_timestamps.append(timestamps[i])

    return stop_sign_timestamps, stop_signs


@composable
def occluded(track_uuid, log_dir):

    annotations_df = read_feather(log_dir / 'sm_annotations.feather')
    track_df = annotations_df[annotations_df['track_uuid'] == track_uuid]
    track_when_occluded = track_df[track_df['num_interior_pts'] == 0]

    if track_when_occluded.empty:
        return []
    else:
        return sorted(track_when_occluded['timestamp_ns'])


def stop_sign_lane(stop_sign_id, log_dir) -> LaneSegment:
    avm = get_map(log_dir)
    pos, _ = get_nth_pos_deriv(stop_sign_id, 0, log_dir)

    ls_list = avm.get_nearby_lane_segments(pos[0,:2], 10)
    best_ls = None
    best_dist = np.inf
    for ls in ls_list:
        dist = np.linalg.norm(pos[0]-ls.right_lane_boundary.xyz[-1])

        if not ls.is_intersection and dist < best_dist:
            best_ls = ls
            best_dist = dist

    if best_ls == None:
        for ls in ls_list:
            dist = np.linalg.norm(pos[0]-ls.right_lane_boundary.xyz[-1])

            if dist < best_dist:
                best_ls = ls
                best_dist = dist
    
    return best_ls


def get_pos_within_lane(pos, ls: LaneSegment) -> tuple:

    if not ls or not is_point_in_polygon(pos[:2], ls.polygon_boundary[:,:2]):
        return None, None

    #Projecting to 2D for BEV
    pos = pos[:2]
    left_line = ls.left_lane_boundary.xyz[:,:2]
    right_line = ls.right_lane_boundary.xyz[:,:2]

    left_dist = 0
    left_point = None
    left_total_length = 0
    min_dist = np.inf
    for i in range(1, len(left_line)):
        segment_start = left_line[i-1]
        segment_end = left_line[i]

        segment_length = np.linalg.norm(segment_end-segment_start)
        segment_direction = (segment_end-segment_start)/segment_length
        segment_proj = np.dot((pos-segment_start), segment_direction)*segment_direction
        proj_length = np.linalg.norm(segment_proj)

        if 0 <= proj_length <= segment_length:
            proj_point = segment_start + segment_proj
        elif proj_length < 0:
            proj_point = segment_start
        else:
            proj_point = segment_end

        proj_dist = np.linalg.norm(pos-proj_point)

        if proj_dist < min_dist:
            min_dist = proj_dist
            left_point = segment_start + segment_proj
            left_dist = left_total_length + proj_length

        left_total_length += segment_length

    right_dist = 0
    right_point = None
    right_total_length = 0
    min_dist = np.inf
    for i in range(1, len(right_line)):
        segment_start = right_line[i-1]
        segment_end = right_line[i]

        segment_length = np.linalg.norm(segment_end-segment_start)
        segment_direction = (segment_end-segment_start)/segment_length
        segment_proj = np.dot((pos-segment_start), segment_direction)*segment_direction
        proj_length = np.linalg.norm(segment_proj)

        if 0 <= proj_length <= segment_length:
            proj_point = segment_start + segment_proj
        elif proj_length < 0:
            proj_point = segment_start
        else:
            proj_point = segment_end

        proj_dist = np.linalg.norm(pos-proj_point)

        if proj_dist < min_dist:
            min_dist = proj_dist
            right_point = segment_start + segment_proj
            right_dist = right_total_length + proj_length

        right_total_length += segment_length

    if left_point is not None and right_point is not None:
        total_length = (left_total_length + right_total_length)/2
        distance = (left_dist + right_dist)/2
        pos_along_length = distance/total_length

        total_width = np.linalg.norm(left_point - right_point)
        lateral_dir_vec = (left_point - right_point)/total_width
        lateral_proj = np.dot((pos-left_point), lateral_dir_vec)*lateral_dir_vec
        pos_along_width = np.linalg.norm(lateral_proj)/total_width
        return pos_along_length, pos_along_width
    
    else:
        print("Position not found within lane_segment. Debug function further.")
        return None, None


@composable
def in_region_of_interest(track_uuid, log_dir):

    in_roi_timestamps = []

    avm = get_map(log_dir)
    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)

    for timestamp in timestamps:
        cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
        ego_to_city = ego_poses[timestamp]
        city_cuboid = cuboid.transform(ego_to_city)
        city_vertices = city_cuboid.vertices_m
        city_vertices = city_vertices.reshape(-1, 3)[:,:2]
        is_within_roi = avm.get_raster_layer_points_boolean(city_vertices, layer_name="ROI")
        if is_within_roi.any():
            in_roi_timestamps.append(timestamp)

    return in_roi_timestamps


def remove_empty_branches(scenario_dict):
    
    if isinstance(scenario_dict, dict):
        track_uuids = list(scenario_dict.keys())
        for track_uuid in track_uuids:
            children = scenario_dict[track_uuid]
            timestamps = get_scenario_timestamps(children)
            if len(timestamps) == 0:
                scenario_dict.pop(track_uuid)
            else:
                remove_empty_branches(children)


def get_scenario_timestamps(scenario_dict:dict) -> list:
    if not isinstance(scenario_dict, dict):
        #Scenario dict is a list of timestamps
        return scenario_dict

    timestamps = []
    for relationship in scenario_dict.values():
        timestamps.extend(get_scenario_timestamps(relationship))

    return sorted(list(set(timestamps)))


def get_scenario_uuids(scenario_dict:dict) -> list:
    if get_scenario_timestamps(scenario_dict):
        scenario_uuids = list(scenario_dict.keys())
        for child in scenario_dict.items():
            if isinstance(child, dict):
                scenario_uuids.extend(get_scenario_uuids(child))
        return list(set(scenario_uuids))
    else:
        return []


def reconstruct_track_dict(scenario_dict):
    track_dict = {}

    for track_uuid, related_objects in scenario_dict.items():
        if isinstance(related_objects, dict):
            timestamps = get_scenario_timestamps(related_objects)
            if len(timestamps) > 0:
                track_dict[track_uuid] = get_scenario_timestamps(related_objects)
        else:
            if len(related_objects) > 0:
                track_dict[track_uuid] = related_objects

    return track_dict


def reconstruct_relationship_dict(scenario_dict):
    #Reconstructing legacy relationship dict

    relationship_dict = {track_uuid: {} for track_uuid in scenario_dict.keys()}

    for track_uuid, child in scenario_dict.items():
        if not isinstance(child, dict):
            continue
        
        descendants = get_objects_and_timestamps(scenario_dict[track_uuid])
        for related_uuid, timestamps in descendants.items():
            relationship_dict[track_uuid][related_uuid] = timestamps

    return relationship_dict


def get_objects_and_timestamps(scenario_dict: dict) -> dict:
    track_dict = {}

    for uuid, related_children in scenario_dict.items():

        if isinstance(related_children, dict):
            track_dict[uuid] = get_scenario_timestamps(related_children)
            temp_dict = get_objects_and_timestamps(related_children)

            for child_uuid, timestamps in temp_dict.items():
                if child_uuid not in track_dict:
                    track_dict[child_uuid] = timestamps
                else:
                    track_dict[child_uuid] = sorted(list(track_dict[child_uuid]) + list(timestamps))
        else:
            if uuid not in track_dict:
                track_dict[uuid] = related_children
            else:
                track_dict[uuid] = sorted(list(set(track_dict[uuid])) + list(related_children))

    return track_dict


def print_indented_dict(d:dict, indent=0):
    """
    Recursively prints a dictionary with indentation.

    Args:
        d (dict): The dictionary to print.
        indent (int): The current indentation level (number of spaces).
    """
    for key, value in d.items():
        print(" " * indent + str(key) + ":")
        if isinstance(value, dict):
            print_indented_dict(value, indent=indent + 4)
        else:
            print(" " * (indent + 4) + str(value))


def extract_pkl_log(filename, log_id, output_dir='output', is_gt=False):
    sequences = load(filename)
    extracted_sequence = {log_id: sequences[log_id]}

    if is_gt:
        save(extracted_sequence, output_dir / f'{log_id}_gt_annotations.pkl')
    else:
        save(extracted_sequence, output_dir / f'{log_id}_extracted.pkl')


def get_related_objects(relationship_dict):
    track_dict = reconstruct_track_dict(relationship_dict)

    all_related_objects = {}

    for track_uuid, related_objects in relationship_dict.items():
        for related_uuid, timestamps in related_objects.items():
            if timestamps and related_uuid not in track_dict and related_uuid not in all_related_objects:
                all_related_objects[related_uuid] = timestamps
            elif timestamps and related_uuid not in track_dict and related_uuid in all_related_objects:
                all_related_objects[related_uuid] = sorted(set(all_related_objects[related_uuid]).union(timestamps))
            elif timestamps and related_uuid in track_dict and related_uuid not in all_related_objects:
                non_track_timestamps = sorted(set(track_dict[related_uuid]).difference(timestamps))
                if non_track_timestamps:
                    all_related_objects[related_uuid] = non_track_timestamps
            elif timestamps and related_uuid in track_dict and related_uuid in all_related_objects:
                non_track_timestamps = set(track_dict[related_uuid]).difference(timestamps)
                if non_track_timestamps:
                    all_related_objects[related_uuid] = sorted(set(all_related_objects[related_uuid]).union(non_track_timestamps))

    return all_related_objects


def get_objects_of_prompt(log_dir, prompt):
    return to_scenario_dict(get_uuids_of_prompt(log_dir, prompt), log_dir)

def get_uuids_of_prompt(log_dir, prompt):
    df = read_feather(log_dir / 'sm_annotations.feather')

    if prompt == 'ANY':
        uuids = df['track_uuid'].unique()
    else:
        category_df = df[df['prompt'] == prompt]
        uuids = category_df['track_uuid'].unique()

    return uuids


def create_mining_pkl(description, scenario, log_dir:Path, output_dir:Path):
    """
    Generates both a pkl file for evaluation and annotations for the scenario mining challenge.
    """

    log_id = log_dir.name
    frames = []
    (output_dir / log_id).mkdir(exist_ok=True)
    
    annotations = read_feather(log_dir / 'sm_annotations.feather') # all objects detected by the tracker
    all_uuids = list(annotations['track_uuid'].unique()) # the full list of objects in this log
    ego_poses = get_ego_SE3(log_dir)

    eval_timestamps = get_eval_timestamps(log_dir) # the list of timestamps to be evaluated

    # input:  { 'uuid_V1': [ts_3, ts_4], 'uuid_V2': [ts_5, ts_6] }
    # output: { ts_3: ['uuid_V1'], ts_4: ['uuid_V1'], ts_5: ['uuid_V2'], ts_6: ['uuid_V2'] }
    referred_objects = swap_keys_and_listed_values(reconstruct_track_dict(scenario))
    
    # input:
    #     {
    #         'uuid_V1': {'uuid_P1': [ts_3, ts_4], 'uuid_P2': [ts_3]},
    #         'uuid_V2': {'uuid_P3': [ts_5, ts_6, ts_7]},
    #     }

    #     output:
    #     {
    #         'uuid_V1': {'uuid_P1': [ts_3, ts_4], 'uuid_P2': [ts_3]},
    #         'uuid_V2': {'uuid_P3': [ts_5, ts_6, ts_7]},
    #     }
    relationships = reconstruct_relationship_dict(scenario)
    
    # output: { 'uuid_P1': [ts_3, ts_4], 'uuid_P2': [ts_3], 'uuid_P3': [ts_5, ts_6, ts_7] }
    related_objects = swap_keys_and_listed_values(get_related_objects(relationships))

    for timestamp in eval_timestamps:
        frame = {}
        timestamp_annotations = annotations[annotations['timestamp_ns'] == timestamp]

        timestamp_uuids = list(timestamp_annotations['track_uuid'].unique())
        ego_to_city = ego_poses[timestamp]

        frame['seq_id'] = (log_id, description)
        frame['timestamp_ns'] = timestamp
        frame['ego_translation_m'] = list(ego_to_city.translation)
        frame['description'] = description

        n = len(timestamp_uuids)
        frame['translation_m'] = np.zeros((n, 3))
        frame['size'] = np.zeros((n,3), dtype=np.float32)
        frame['yaw'] = np.zeros(n, dtype=np.float32)
        frame['label'] = np.zeros(n, dtype=np.int32)
        frame['name'] = np.zeros(n, dtype='<U31')
        frame['track_id'] = np.zeros(n, dtype=np.int32)
        frame['score'] = np.zeros(n, dtype=np.float32)

        for i, track_uuid in enumerate(timestamp_uuids):
            track_df = timestamp_annotations[timestamp_annotations['track_uuid'] == track_uuid]
            if track_df.empty:
                continue
            
            cuboid = CuboidList.from_dataframe(track_df)[0]
            translation_m = ego_to_city.transform_from(cuboid.xyz_center_m)
            size = np.array([cuboid.length_m, cuboid.width_m, cuboid.height_m], dtype=np.float32)
            yaw = Rotation.from_matrix(ego_to_city.compose(cuboid.dst_SE3_object).rotation).as_euler('zxy')[0]

            if timestamp in referred_objects and track_uuid in referred_objects[timestamp]:
                category = "REFERRED_OBJECT"                                                                                                                            
                label = 0
            elif timestamp in related_objects and track_uuid in related_objects[timestamp]:
                category = "RELATED_OBJECT"
                label = 1
            else:
                category = "OTHER_OBJECT"
                label = 2

            frame['translation_m'][i,:] = translation_m
            frame['size'][i,:] = size
            frame['yaw'][i] = yaw
            frame['label'][i] = label
            frame['name'][i] = category
            frame['track_id'][i] = all_uuids.index(track_uuid)

            # Assign a score of 1 to tracker predictions that do not have an associated confidence value
            try:
                frame['score'][i] = float(track_df['score'].iloc[0])
            except:
                frame['score'][i] = 1.0

        frames.append(frame)

    sequences = {(log_id, description): frames}
    save(sequences, output_dir / log_id / f'{description}_predictions.pkl')
    print(f'Scenario pkl file for {description}_{log_id[:8]} saved successfully.')

    return True


def fix_pred_pkl(prediction_pkl:Path, label_pkl:Path, output_filename:Path) -> None:
    """
    Aligns the sequences and timestamps between a prediction PKL file with the label PKL file. 
    Pads the prediction pkl with a default prediction for timestamps and log-prompt pairs that are in the annotations
    PKL but not the prediction PKL. Remove timestamps found within the prediction PKL that are not within the label PKL
    """

    with open(prediction_pkl, 'rb') as file:
        predictions:dict = pickle.load(file)

    with open(label_pkl, 'rb') as file:
        labels:dict = pickle.load(file)   

    #Remove sequences and timestamps from the predictions that are not in the labels 
    filtered_predictions = {}

    for seq_id, pred_frames in predictions.items():
        if seq_id not in labels:
            continue

        label_frames = labels[seq_id]
        label_timestamps = []
        for frame in label_frames:
            label_timestamps.append(frame['timestamp_ns'])

        filtered_frames = []
        for frame in pred_frames:
            if frame['timestamp_ns'] in label_timestamps:
                filtered_frames.append(frame)

        filtered_predictions[seq_id] = filtered_frames

    if not filtered_predictions:
        print('Supplied prediction pkl and label pkl have no overlap! Make sure you are supplying the correct combination' \
        'of predictions and labels.')
        return

    #Add default sequences and timestamps that are in the labels but not in the timestamps
    fixed_predictions = {}

    for seq_id, label_frames in labels.items():
        frame_infos_dict = {}
        for frame in label_frames:
            timestamp = frame['timestamp_ns']
            frame_infos_dict[timestamp] = {
                'timestamp_ns': timestamp,
                'seq_id': frame['seq_id'],
                'ego_translation_m': frame['ego_translation_m']
            }
            if 'description' in frame:
                frame_infos_dict[timestamp]['description'] = frame['description']


        if seq_id not in filtered_predictions:
            default_sequence = create_default_sequence(frame_infos_dict)
            fixed_predictions[seq_id] = default_sequence
            continue

        pred_frames = filtered_predictions[seq_id]
        pred_timestamps = []
        for frame in pred_frames:
            if len(frame['track_id'] == 0):
                print('Zero-length frame changed')
                frame = create_default_frame(frame_infos_dict[frame['timestamp_ns']])
            pred_timestamps.append(frame['timestamp_ns'])

        for frame in label_frames:
            timestamp = frame['timestamp_ns']
            if timestamp not in pred_timestamps:
                print(f'Timestamp {timestamp} appended')
                pred_frames.append(create_default_frame(frame_infos_dict[timestamp]))

        print(len(label_frames))
        print(len(pred_frames))
        assert len(pred_frames) == len(label_frames)
        fixed_predictions[seq_id] = pred_frames
    assert len(fixed_predictions) == len(labels)

    with open(output_filename, 'wb') as file:
        pickle.dump(fixed_predictions, file)


def create_default_frame(frame_infos) -> dict:

    frame = {}
    frame['seq_id'] = frame_infos['seq_id']
    frame['timestamp_ns'] = frame_infos['timestamp_ns']
    frame['ego_translation_m'] = frame_infos['ego_translation_m']
    if 'description' in frame_infos:
        frame['description'] = frame_infos['description']

    frame['translation_m'] = np.zeros((1, 3))
    frame['translation_m'][0] = frame['ego_translation_m']
    frame['size'] = np.zeros((1,3), dtype=np.float32)
    frame['yaw'] = np.zeros(1, dtype=np.float32)
    frame['label'] = np.array([2], dtype=np.int32)
    frame['name'] = np.array(['OTHER_OBJECT'], dtype='<U31')
    frame['track_id'] = np.zeros(1, dtype=np.int32)
    frame['score'] = np.zeros(1, dtype=np.float32)

    return frame


def create_default_sequence(frame_infos_dict:dict) -> list:
    sequence = []
    for frame_infos in frame_infos_dict.values():
        sequence.append(create_default_frame(frame_infos))

    return sequence

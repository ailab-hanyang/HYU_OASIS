"""multi_class_tracking — rule-based re-tracker for RefAV.

To fix the ID-switching problem of Le3DE2E detection results, it keeps the
detection / class information as-is and only assigns new track_uuid values.

Usage:
    from tools.multi_class_tracking.src import (
        TrackerParams, MultiClassTracker, PostProcessParams,
        build_output_df, load_frames, save_output_feather,
    )
"""

from .track import Track, FrameRecord, Measurement, Frame
from .kalman import KFParams, predict, update, build_F, build_H
from .association import (
    AssociationParams, build_cost_matrix,
    hungarian_match, greedy_match, match_pairs,
)
from .tracker import TrackerParams, MultiClassTracker
from .post_process import PostProcessParams, build_output_df, summarize_tracks, classify_static, jitter_freeze_yaw, iou_clean_removed, dynamic_velocity_yaw
from .imm_filter import IMMFilter, IMMParams
from .imm_config import imm_params_from_config, default_imm_params, SUB_MODELS
from .imm_debug import DebugRecorder, SmoothInputRecorder
from .motion_models import (
    CVModel, CAModel, CTRVModel, CTRAModel,
    MotionModelParams, BaseMotionModel, init_from_xy_yaw,
)
from .io_utils import (
    load_frames, save_output_feather,
    get_log_dir, get_dst_log_dir, PROJECT_ROOT,
)

__all__ = [
    "Track", "FrameRecord", "Measurement", "Frame",
    "KFParams", "predict", "update", "build_F", "build_H",
    "AssociationParams", "build_cost_matrix",
    "hungarian_match", "greedy_match", "match_pairs",
    "TrackerParams", "MultiClassTracker",
    "PostProcessParams", "build_output_df", "summarize_tracks", "classify_static", "jitter_freeze_yaw", "iou_clean_removed", "dynamic_velocity_yaw",
    "load_frames", "save_output_feather",
    "get_log_dir", "get_dst_log_dir", "PROJECT_ROOT",
    "IMMFilter", "IMMParams", "imm_params_from_config", "default_imm_params",
    "SUB_MODELS", "DebugRecorder", "SmoothInputRecorder",
    "CVModel", "CAModel", "CTRVModel", "CTRAModel",
    "MotionModelParams", "BaseMotionModel", "init_from_xy_yaw",
]

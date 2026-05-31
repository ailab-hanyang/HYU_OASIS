"""Script to reproduce RefProg results. It may take several hours to complete an experiment, depending on the number of tracks coming from the tracker."""
from dotenv import load_dotenv; load_dotenv()
import argparse
import json
from pathlib import Path
import os
import yaml

import refAV.paths as paths
from refAV.dataset_conversion import (
    separate_scenario_mining_annotations,
    pickle_to_feather,
    create_gt_mining_pkls_parallel,
    create_gt_pkl_file,
    mirror_tracker_caches,
    filter_tracker_feathers_by_score,
    thr_tag,
)
from refAV.parallel_scenario_prediction import run_parallel_eval
from refAV.eval import evaluate_pkls, combine_pkls
from refAV.utils import construct_caches

parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument(
    "--exp_name",
    type=str,
    default="exp63",
    help="Enter the name of the experiment from experiments.yml you would like to run.",
)
parser.add_argument(
    "--procs_per_task",
    type=int,
    default=3,
    help="The number of processes your eval script should launch with",
)
parser.add_argument(
    "--score_threshold",
    type=float,
    default=None,
    help="Confidence threshold (e.g. 0.11). 설정 시 sm_annotations 의 score>=thr 행만 사용. "
         "yml 의 score_threshold 가 있으면 그것이 우선.",
)
args = parser.parse_args()

with open(paths.EXPERIMENTS, "rb") as file:
    config = yaml.safe_load(file)

exp_name = config[args.exp_name]["name"]
llm = config[args.exp_name]["LLM"]
tracker = config[args.exp_name]["tracker"]
split = config[args.exp_name]["split"]
multi_agent = config[args.exp_name].get("multi_agent", False)

if llm not in config["LLM"]:
    print("Experiment uses an invalid LLM")
if tracker not in config["tracker"]:
    print("Experiment uses invalid tracking results")
if split not in ["train", "test", "val"]:
    print("Experiment must use split train, test, or val")

log_prompts_path = paths.SM_DOWNLOAD_DIR / f"log_prompt_pairs_{split}.json"

if split in ["train", "val"]:

    # GT 처리
    sm_feather = paths.SM_DOWNLOAD_DIR / f"scenario_mining_{split}_annotations.feather" # 모든 로그의 모든 프롬프트에 대한 GT 어노테이션이 한 파일에 통합
    sm_data_split_path = paths.SM_DATA_DIR / split
    combined_gt_path = sm_data_split_path / f"combined_gt_{split}.pkl"

    if not combined_gt_path.exists():
        separate_scenario_mining_annotations(sm_feather, sm_data_split_path)
        create_gt_mining_pkls_parallel(
            sm_feather,
            sm_data_split_path,
            num_processes=max(1, int(0.9 * os.cpu_count())),
        )
        create_gt_pkl_file(
            sm_data_split_path, 
            log_prompts_path, 
            output_path = combined_gt_path
        )

# 3D Perception prediction (log당) feather 저장 경로
tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split

# Tracker 변형 처리 — 두 종류의 suffix 가 chain 으로 적용될 수 있다.
#   _ego    : ego offset 보정. base tracker (suffix 없음) 의 pkl 에서 feather 자동 생성.
#   _yawfix : stage-1 yaw flip 보정. _ego 변형의 feather 에서 별도 스크립트로 생성
#             (tools/rts_smoothing/apply_yaw_correction.py). pkl 은 없음.
# 두 변형 모두 부모 tracker 의 color/crop cache 를 symlink 로 재사용한다 — yaw flip 은
# 박스의 길이·너비·높이·중심을 보존하므로 카메라 투영 crop 영역이 동일.

# 한 단계 위 부모 tracker (cache mirror 의 source) 결정.
if tracker.endswith("_yawfix"):
    parent_tracker = tracker[:-len("_yawfix")]      # 예: Le3DE2E_Tracking_ego
else:
    parent_tracker = tracker

# pkl 의 base 이름 — _ego 까지 마저 떼어낸 형태.
src_tracker = parent_tracker[:-4] if parent_tracker.endswith("_ego") else parent_tracker
tracker_predictions_pkl = Path(f"tracker_downloads/{src_tracker}_{split}.pkl")

if not tracker_predictions_dest.exists():
    if tracker.endswith("_yawfix"):
        # yawfix feather 는 별도 스크립트로만 생성 가능 — 없으면 명확히 안내.
        raise FileNotFoundError(
            f"yawfix tracker prediction 디렉토리 없음: {tracker_predictions_dest}\n"
            f"  → 먼저 yaw flip 보정을 적용해서 feather 를 생성하세요:\n"
            f"     python -m tools.rts_smoothing.apply_yaw_correction "
            f"--src_tracker {parent_tracker} --dst_tracker {tracker} --split {split}"
        )
    av2_data_split = paths.AV2_DATA_DIR
    pickle_to_feather(av2_data_split, tracker_predictions_pkl, tracker_predictions_dest)

# 변형 tracker (_ego 또는 _yawfix) 면 부모의 color/crop cache 를 symlink 로 미러링.
# construct_caches 가 per-log 단위로 cache 존재를 체크해 자동 스킵하므로 추가 분기 불필요.
if tracker.endswith("_yawfix") or tracker.endswith("_ego"):
    cache_src_name = parent_tracker if tracker.endswith("_yawfix") else src_tracker
    src_tracker_dir = paths.TRACKER_PRED_DIR / cache_src_name
    dst_tracker_dir = paths.TRACKER_PRED_DIR / tracker
    n = mirror_tracker_caches(src_tracker_dir, dst_tracker_dir, split)
    if n > 0:
        print(f"  [Cache] mirrored {n} log caches: {cache_src_name} -> {tracker}")

# Confidence threshold 필터 — 켜져있으면 새 tracker dir(<base>_thr<NNN>) 에 score 필터된 feather 저장.
# 우선순위: yml score_threshold > CLI --score_threshold > exp_name suffix 자동추론
import re as _re
yml_score_thr = config[args.exp_name].get("score_threshold")
score_threshold = yml_score_thr if yml_score_thr is not None else args.score_threshold

# 자동 추론 — exp_name 에 "_thr<NNN>" 이 들어있으면 그 값으로 추정
if score_threshold is None:
    _m = _re.match(r"^(.+?)_thr(\d{3})$", exp_name)
    if _m:
        score_threshold = int(_m.group(2)) / 100.0
        print(f"  [Filter] auto-inferred score_threshold={score_threshold} from exp_name suffix")

if score_threshold is not None and score_threshold > 0:
    suffix = thr_tag(score_threshold)              # 예: "thr011"
    base_tracker = tracker[:-(len(suffix)+1)] if tracker.endswith(f"_{suffix}") else tracker
    new_tracker = base_tracker if tracker.endswith(f"_{suffix}") else f"{tracker}_{suffix}"
    base_for_filter = paths.TRACKER_PRED_DIR / base_tracker
    new_tracker_dir = paths.TRACKER_PRED_DIR / new_tracker

    n_fea = filter_tracker_feathers_by_score(base_for_filter, new_tracker_dir, split, score_threshold)
    if n_fea > 0:
        print(f"  [Filter] score>={score_threshold} → {n_fea} feathers: {base_tracker} -> {new_tracker}")
    n_cache = mirror_tracker_caches(base_for_filter, new_tracker_dir, split)
    if n_cache > 0:
        print(f"  [Cache] mirrored {n_cache} caches into {new_tracker}")

    # tracker 변수 갱신 (필터된 dir 사용). exp_name 은 이미 suffix 가 있으면 유지, 없으면 추가.
    tracker = new_tracker
    tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split
    if not exp_name.endswith(f"_{suffix}"):
        exp_name = f"{exp_name}_{suffix}"
    print(f"  [Filter] tracker → {tracker}, exp_name → {exp_name}")

# Build caches before parallel eval so subprocesses just load from disk
with open(log_prompts_path, 'rb') as f:
    log_prompts = json.load(f)
all_log_dirs = [paths.TRACKER_PRED_DIR / tracker / split / log_id for log_id in log_prompts.keys()]
construct_caches(all_log_dirs)

run_parallel_eval(exp_name, log_prompts_path, args.procs_per_task, multi_agent=multi_agent)

experiment_dir = paths.SM_PRED_DIR / exp_name 
combined_preds_path = combine_pkls(experiment_dir / "scenario_predictions", log_prompts_path, suffix="_predictions")

# Only train and val splits will be evaluated
if split in ["train", "val"]:
    metrics = evaluate_pkls(combined_preds_path, combined_gt_path, experiment_dir)
    print(metrics)

if split == "val":
    print("Nice work! A submission to EvalAI can be made with")
    print(f"evalai challenge 2662 phase 5283 submit --file {combined_preds_path} --large")
elif split == "test":
    print("Only train and val splits can be evaluated. Please make a submission to EvalAI!")
    print(f"evalai challenge 2662 phase 5282 submit --file {combined_preds_path} --large")

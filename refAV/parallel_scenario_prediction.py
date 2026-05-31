"""Script to run eval.py in parallel."""

import subprocess
import multiprocessing
import argparse
import os
import sys
import time
import json
from collections import deque
from pathlib import Path
import tempfile
import refAV.paths as paths


def _write_temp_json(lpp_dict: dict, task_idx: int) -> Path:
    """Write log_prompt mapping to a temp JSON file and return its path.

    use delete=False so the file persists after closing, allowing subprocess access.
    """
    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', delete=False, suffix='.json', prefix=f'task_{task_idx}_'
    )
    temp_file_path = Path(temp_file.name)
    json.dump(lpp_dict, temp_file, indent=4)
    temp_file.close()
    return temp_file_path


def run_parallel_eval(exp_name: str, log_prompts_path: Path, procs_per_task: int = 2, multi_agent: bool = False):
    """
    Launches multiple eval.py processes in parallel.
    It determines which log_id/prompt pairs still need processing,
    divides them among tasks, and allocates available CPUs dynamically.
    """
    log_prompts_path = Path(log_prompts_path)

    print(f"Starting parallel evaluation for experiment: {exp_name}")
    print(f"Reading log prompts from: {log_prompts_path}")

    # Read the full log_prompts mapping
    with open(log_prompts_path, 'rb') as file:
        lpp = json.load(file)

    # Build work units grouped by log_id (one work unit = one log_id + its pending prompts).
    # log_id 단위로 묶는 이유: cache_manager.load_custom_caches() 가 log_id 별 호출이라
    # 같은 log 의 prompts 는 한 subprocess 에서 처리해야 cache reload 안 함.
    print("Checking which log_id/prompt pairs need evaluation...")
    work_units = []                              # [(log_id, [prompts...]), ...]
    total_work_items = 0
    for log_id, prompts in lpp.items():
        pending = []
        for prompt in prompts:
            pred_path = paths.SM_PRED_DIR / exp_name / 'scenario_predictions' / log_id / f'{prompt}_predictions.pkl'
            if not pred_path.exists():
                pending.append(prompt)
        if pending:
            work_units.append((log_id, pending))
            total_work_items += len(pending)

    total_units = len(work_units)
    print(f"Total log_id/prompt pairs requiring evaluation: {total_work_items}")
    print(f"Total log_id work units: {total_units}")

    if total_units == 0:
        print("No evaluation needed. All predictions found.")
        return

    # Get available CPU count
    cpu_count = multiprocessing.cpu_count()
    # Leave one CPU free for the parent script and other system processes
    cpus_available_for_tasks = max(1, int(.95*(cpu_count)))

    print(f"System has {cpu_count} CPUs. {cpus_available_for_tasks} available for tasks.")
    print(f"Each task requests {procs_per_task} processes.")

    # Maximum number of subprocesses running at the same time.
    max_concurrent = max(1, cpus_available_for_tasks // procs_per_task)
    # Don't spawn more concurrent workers than there are work units.
    max_concurrent = min(max_concurrent, total_units)

    print(f"Max concurrent subprocesses: {max_concurrent}")

    # --- Dynamic Dispatch Loop ---
    work_queue = deque(work_units)               # 각 원소: (log_id, [prompts...])
    running = []                                 # [(Popen, temp_file, idx, started_at, log_id, n_prompts), ...]
    finished = 0
    launched = 0
    completed_prompts = 0
    queued_prompts = total_work_items            # prompts still in work_queue
    t_start = time.time()

    print("\nStarting parallel tasks...")

    try:
        while work_queue or running:
            # 1) 빈 슬롯 채움
            while work_queue and len(running) < max_concurrent:
                log_id, prompts = work_queue.popleft()
                launched += 1
                idx = launched

                task_lpp_dict = {log_id: prompts}
                try:
                    temp_file_path = _write_temp_json(task_lpp_dict, idx)
                except Exception as e:
                    print(f"Error creating temp file for task {idx}: {e}", file=sys.stderr)
                    raise

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(procs_per_task)
                env["PYTHONWARNINGS"] = "ignore::FutureWarning"

                command = [
                    sys.executable,
                    str(Path("refAV/eval.py")),
                    "--exp_name", exp_name,
                    "--log_prompt_pairs", str(temp_file_path),
                    "--num_processes", str(procs_per_task),
                    "--log_index", str(idx),
                    "--total_logs", str(total_units),
                ]
                if multi_agent:
                    command.append("--multi_agent")

                try:
                    process = subprocess.Popen(command, env=env)
                except FileNotFoundError:
                    print(
                        f"Error: Python interpreter '{sys.executable}' or script 'refAV/eval.py' "
                        f"not found. Make sure you are running from the project root.",
                        file=sys.stderr,
                    )
                    if temp_file_path.exists():
                        try: os.remove(temp_file_path)
                        except OSError: pass
                    raise
                except Exception as e:
                    print(f"An error occurred launching task {idx}: {e}", file=sys.stderr)
                    if temp_file_path.exists():
                        try: os.remove(temp_file_path)
                        except OSError: pass
                    raise

                n_prompts = len(prompts)
                queued_prompts -= n_prompts
                running.append((process, temp_file_path, idx, time.time(), log_id, n_prompts))
                print(
                    f"  [launch {idx}/{total_units}] log_id={log_id} "
                    f"prompts={n_prompts} running={len(running)}/{max_concurrent} "
                    f"| queue: {len(work_queue)} logs / {queued_prompts} prompts"
                )

            # 2) 끝난 것 회수
            still_running = []
            any_finished = False
            for proc, temp_file_path, idx, t0, log_id, n_prompts in running:
                rc = proc.poll()
                if rc is None:
                    still_running.append((proc, temp_file_path, idx, t0, log_id, n_prompts))
                else:
                    finished += 1
                    completed_prompts += n_prompts
                    any_finished = True
                    elapsed = time.time() - t0
                    if temp_file_path.exists():
                        try:
                            os.remove(temp_file_path)
                        except Exception as e:
                            print(f"  Error removing temp file {temp_file_path}: {e}", file=sys.stderr)
                    # Progress + ETA
                    wall = time.time() - t_start
                    pct = 100.0 * completed_prompts / total_work_items if total_work_items else 100.0
                    if completed_prompts > 0 and completed_prompts < total_work_items:
                        eta_s = wall * (total_work_items - completed_prompts) / completed_prompts
                        eta_str = f"ETA {eta_s/60:.1f}m"
                    else:
                        eta_str = "ETA -"
                    if rc != 0:
                        print(
                            f"  [FAIL  {idx}/{total_units}] log_id={log_id} "
                            f"rc={rc} after {elapsed:.1f}s "
                            f"| progress: {finished}/{total_units} logs, "
                            f"{completed_prompts}/{total_work_items} prompts ({pct:.1f}%) {eta_str}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  [done  {idx}/{total_units}] log_id={log_id} "
                            f"elapsed={elapsed:.1f}s "
                            f"| progress: {finished}/{total_units} logs, "
                            f"{completed_prompts}/{total_work_items} prompts ({pct:.1f}%) {eta_str}"
                        )
            running = still_running

            # 3) 아무도 안 끝났으면 짧게 sleep (busy loop 방지)
            if not any_finished and running:
                time.sleep(0.3)

        print(f"\nAll parallel tasks finished. ({finished}/{total_units} units)")

    finally:
        # KeyboardInterrupt 등 비정상 종료 시 cleanup
        if running:
            print("\nTerminating remaining subprocesses...", file=sys.stderr)
            for proc, _, _, _, _, _ in running:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            for proc, _, _, _, _, _ in running:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            for _, temp_file_path, _, _, _, _ in running:
                if temp_file_path.exists():
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel eval.py tasks.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Name of the experiment from the exp.yml file")
    parser.add_argument("--log_prompts_path", type=str, required=True, help="Path to the JSON file containing log_id to prompt list mapping.")
    parser.add_argument("--procs_per_task", type=int, default=2, help="Base number of processes to request for each eval.py task. Extra available CPUs will be distributed.")
    args = parser.parse_args()


    run_parallel_eval(
        exp_name=args.exp_name,
        log_prompts_path=args.log_prompts_path,
        procs_per_task=args.procs_per_task
    )

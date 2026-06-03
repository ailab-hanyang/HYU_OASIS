# HYU_OASIS

HYU's entry for the **Argoverse2 Scenario Mining Challenge (CVPR Workshop 2026)**, built on
[RefAV](https://arxiv.org/pdf/2505.20981). Given a natural-language prompt, an LLM composes
hand-crafted *atomic functions* (e.g. `turning`, `has_objects_in_front`) into a program that
narrows a set of bounding-box track predictions down to the tracks that match the prompt.

OASIS adds a preprocessing + perception toolchain under [`tools/`](tools/): multi-class
re-tracking, IMM/RTS track smoothing, a vLLM scene-context annotator, and a per-object
vision-language classifier server used by the `get_visual_*` atomic functions.

> All commands are run **from the repository root** with `PYTHONPATH=.` set.

## Installation

[Conda](https://anaconda.org/anaconda/conda) is recommended (Python **3.10**):

```bash
conda create -n oasis python=3.10 && conda activate oasis
pip install -r requirements.txt
export PYTHONPATH=.
```

`torch`/`torchvision` are unpinned — install the build matching your CUDA (see
[pytorch.org](https://pytorch.org)). The vLLM tools run in their own container (see below), so
`vllm` is **not** installed by `requirements.txt`. Baseline-specific extras (CLIP, SAM2,
GroundingDINO, …) are listed, commented, at the bottom of `requirements.txt`.

## Environment variables

Create a `.env` file in the repo root (it is gitignored); `run/run_experiment.py` loads it
automatically via `python-dotenv`. Set only the key(s) for the LLM provider your experiment uses
(the `LLM` field of your entry in `experiments.yml`):

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY` | Google Gemini scenario code generation |
| `ANTHROPIC_API_KEY` | Anthropic Claude scenario code generation |
| `OPENAI_API_KEY` | OpenAI scenario code generation |
| `HF_TOKEN` | Hugging Face dataset / gated weights (also the docker VLM server) |

(`baselines/black_box` additionally reads `RefAV_OPENAI_API_KEY`.)

## Dataset

Download the Argoverse2 **sensor** dataset and the RefAV scenario-mining annotations. Paths are
configured in [refAV/paths.py](refAV/paths.py) (defaults shown); download into those locations or
symlink them.

```bash
conda install s5cmd -c conda-forge

# AV2 sensor dataset  ->  paths.AV2_DATA_DIR (data/datasets/sensor)
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/*" data/datasets/sensor

# Scenario-mining annotations  ->  paths.SM_DOWNLOAD_DIR (scenario_mining_downloads)
hf auth login   # or: s5cmd --no-sign-request cp "s3://argoverse/tasks/scenario_mining/*" ...
hf download CainanD/RefAV --repo-type dataset --local-dir scenario_mining_downloads
```

Detector/tracker outputs go in `paths.TRACKER_DOWNLOAD_DIR` (`tracker_downloads`); see the
[LT3D repo](https://github.com/neeharperi/LT3D) or use pre-computed tracks. Outputs are written
under `output/` (predictions, caches, scene-context annotations — see `refAV/paths.py`).

## Quickstart

```bash
# Reproduce an experiment defined in run/experiment_configs/experiments.yml
python run/run_experiment.py --exp_name exp63

# Or step through unpacking / prediction / evaluation / visualization interactively:
jupyter notebook run/tutorial.ipynb
```

Each `--exp_name` selects an entry (LLM, tracker, split, …) from
[run/experiment_configs/experiments.yml](run/experiment_configs/experiments.yml).

## Repository layout

| Path | Contents |
|---|---|
| [refAV/](refAV/) | core library: `atomic_functions.py`, `code_generation.py` (LLM → scenario program), `eval.py`, `utils.py`, `paths.py` (edit data/output paths here) |
| [run/](run/) | `run_experiment.py` (entry point), `tutorial.ipynb`, `experiment_configs/`, `llm_prompting/`, `nuprompt_conversion/` |
| [tools/multi_class_tracking/](tools/multi_class_tracking/) | IMM multi-class re-tracking |
| [tools/rts_smoothing/](tools/rts_smoothing/) | IMM/RTS track smoothing + ego/yaw correction |
| [tools/scene_context_extraction/](tools/scene_context_extraction/) | vLLM scene-context annotator (runs in a container) |
| [tools/vlm_server/](tools/vlm_server/) | per-object VL classifier server for `get_visual_actor` / `get_visual_behavior` |
| [tools/scripts/](tools/scripts/) | one-command pipeline orchestrators (`run_*.sh`) |
| [baselines/](baselines/) | reference methods (black-box, GroundingSAM, CLIP, ReferGPT) |

Each `tools/<name>/` has its own README with setup and usage.

## License & citation

MIT (see [LICENSE](LICENSE)). This project builds on RefAV:

```bibtex
@article{refav2025,
  title  = {RefAV: Towards Planning-Centric Scenario Mining},
  author = {Davidson, Cainan and others},
  journal= {arXiv preprint arXiv:2505.20981},
  year   = {2025}
}
```

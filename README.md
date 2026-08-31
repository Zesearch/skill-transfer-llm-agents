# Induce from Subtasks, Store as Text

Code, data, and induced skill libraries for the anonymous submission *"Induce from Subtasks, Store as Text: Skill Utility Explains Cross-Task Skill Transfer in LLM Agents"*.

The paper studies how LLM agents should induce and store reusable skills. It crosses two **induction levels** (task-level vs subtask-level trajectory spans) with two **skill formats** (text workflow notes vs Python functions), plus a no-memory baseline, giving six conditions: `Task`, `Task+Text`, `Task+Code`, `Subtask`, `Subtask+Text`, `Subtask+Code`. All conditions share identical induction prompts, so each contrast isolates one axis. Experiments cover three long-horizon benchmarks (AppWorld test-challenge, OfficeBench, KramaBench) and eleven models (Qwen3-235B-A22B, GPT-OSS-120B, Nemotron-3-Super-120B via Amazon Bedrock; Qwen3-4B/8B/14B/32B and Gemma-3-4B/12B/27B via local vLLM; Gemini-3.1-Pro via Google Vertex AI).

## Repository layout

```
skill_induction/            the experiment package
  agents/                   task-level (linear*) and subtask-level (subtask*) agents
  envs/                     AppWorld / OfficeBench / KramaBench adapters
  memory/                   text (awm) and code (asi) skill memories + retrieval
  models/                   Bedrock, Vertex Gemini, and local-vLLM chat clients
  prompts/                  every prompt: ReAct cores, role blocks, induction prompts
  caching/                  disk-backed LLM cache and per-task call ledger
  experiments/run.py        the experiment CLI
  paths.py                  all configurable paths (env-var overridable)
data/
  results/                  all run pickles: dialogues, skill libraries, caches
  results/inspector_tasks/  one HTML fragment per task, used by the viewer
  AAA_final_paper_plot.ipynb  reproduces every notebook-generated paper asset
  inspector.html            trajectory viewer shell
scripts/
  exp.sh                    run agents x models x benchmarks on the current node
  submit.sh                 one sbatch job per (model x benchmark)
  build_inspector.py        regenerate the trajectory viewer from the pickles
  gemma3_pythonic.jinja     vLLM chat template for Gemma-3 tool calling
api_key/                    credentials (fill in the template; never commit real keys)
figures/                    notebook output directory
```

## Released data

`data/results/` is not tracked in git (16 GB). Download the archive from the link in `data/RESULTS_DOWNLOAD.txt` and extract it inside `data/` before running the notebook or the viewer.

`data/results/` contains one pickle per experiment cell, named `<benchmark>_<agent>_<model>_<stage>.pkl`:

- `agent`: `linear`, `linear_awm`, `linear_asi`, `subtask`, `subtask_awm`, `subtask_asi` (awm = text skills, asi = code skills).
- `stage`: `final_v1` (main experiments), `prompt_abl2_L1` / `prompt_abl2_L2` (induction-prompt ablation), `subskill_v1` (memory-swap arm: the task-level agent replaying the subtask-level agent's frozen skill library).
- Each dialogue pickle is a list of per-task records: task metadata, outcome, full message history, per-call LLM ledger, and skills induced/retrieved.
- `*_memory.pkl` files are the induced skill libraries (skills + provenance log) for every memory condition.
- `_embcache_final_v1.pkl`, `_judge_relevance_v1.pkl`, `_oracle_selfretr_final_v1.pkl` are analysis caches (sentence embeddings, LLM-judge relevance labels, self-retrieval rows).
- `shuffle_order_<benchmark>_seed0.json` fix the task-stream order shared by all conditions.

## Setup

Python 3.11+ with:

```
pip install numpy pandas tqdm filelock requests boto3 google-genai sentence-transformers matplotlib
```

`vllm` is additionally needed to serve the `*-local` models; `torch` + `sentence-transformers` are needed at run time for skill retrieval (and only on embedding-cache misses in the analysis notebook).

### Credentials

Copy the template and fill in your own keys:

```
cp api_key/config.example.json api_key/config.json
```

- `bedrock_api_key` / `bedrock_region` for the Bedrock-served MoE models (or standard AWS credentials via the environment).
- `gcp_project` / `gcp_location` for Gemini on Vertex AI (auth via `gcloud auth application-default login`).
- `huggingface_api_key` for downloading gated model weights for vLLM.

### Benchmark assets

The benchmark environments live outside the repo under `SKILL_INDUCTION_EXTERNAL` (default `external/`); see `skill_induction/paths.py` for every path and its env override:

- **AppWorld**: install the `appworld` package in a dedicated conda env and place its data under `external/appworld-data/` (`APPWORLD_DATA_ROOT`).
- **OfficeBench**: vendor the upstream benchmark under `external/_port/officebench/` (`OFFICEBENCH_PORT_ROOT`) and build an apptainer image with LibreOffice + Tesseract at `external/officebench-sandbox/officebench.sif` (`OFFICEBENCH_SANDBOX_SIF`).
- **KramaBench**: vendor the upstream repo (task workloads + data lakes) under `external/_port/kramabench/` (`KRAMABENCH_PORT_ROOT`).

## Running experiments

Single cell:

```
python -m skill_induction.experiments.run \
    --agent subtask_asi --benchmark appworld \
    --model qwen-3-235b-a22b-amazon --stage final_v1 --max-iter 50
```

Useful flags: `--num-tasks/-N`, `--override` (ignore the resume checkpoint), `--induction-level {1,2,3}` (induction-prompt ablation ladder), `--shuffle-seed` (fixed task-stream order), `--subtask-skills` + `--subtask-skills-stage` (memory-swap arm). Results append to `data/results/` and runs resume from the result pickle.

Whole matrices on one node (starts and reuses a local vLLM automatically for `*-local` models):

```
AGENTS="subtask_awm subtask_asi" MODELS="qwen-3-32b-local" BENCHES="appworld" bash scripts/exp.sh
```

Slurm submission (one job per model x benchmark): fill in `YOUR_ACCOUNT` / `YOUR_PARTITION` at the top of `scripts/submit.sh` (or pass `ACCOUNT=... PART=...` as env vars), then:

```
MODELS="qwen-3-32b-local" BENCHES="appworld officebench kramabench" bash scripts/submit.sh
```

## Reproducing the paper figures

Open `data/AAA_final_paper_plot.ipynb` and run it top to bottom from `data/`. It reads the pickles from `data/results/`, prints the Table 2 numbers, and writes each figure PDF (in the paper's order of appearance) to `figures/`. Only `numpy` + `matplotlib` are required; the skill-utility cells reuse `data/results/_embcache_final_v1.pkl`, so no GPU or embedding model is needed. The difficulty figure additionally needs the AppWorld task metadata (`APPWORLD_TASKS_DIR`, default `external/appworld-data/data/tasks`).

## Browsing trajectories

Open `data/inspector.html` directly in a browser (works over `file://`; per-task content loads lazily into an iframe), or serve it:

```
python -m http.server -d data
# then visit http://localhost:8000/inspector.html
```

Filter by benchmark, agent, model, stage, or outcome and click a task to see its full dialogue, subtask plans and summaries, and the skills it retrieved and induced. `scripts/build_inspector.py` regenerates the viewer from the pickles.

## License

Code and released artifacts are under the MIT License (see `LICENSE`).

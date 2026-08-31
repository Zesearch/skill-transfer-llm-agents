#!/usr/bin/env bash
# Run agents x models x benchmarks on the current node.
#
# Usage: bash scripts/exp.sh, with knobs passed as env vars, e.g.
#   AGENTS="subtask_asi" MODELS="qwen-3-32b-local" BENCHES="appworld" N=50 bash scripts/exp.sh
#
# Knobs: AGENTS MODELS BENCHES STAGE N PILOT PILOT_FILE OVERRIDE MAXJOBS ENV_NAME
#   EXP_TP            tensor-parallel size when this script starts a vLLM (default 1)
#   APPWORLD_STAGGER  seconds between appworld cell launches (default 25)
#   APPWORLD_CLEANUP  pkill stray 'appworld serve' after the bench (default true; submit.sh sets false so co-located jobs don't clobber)
#
# For *-local models: if no vLLM is already serving the model at VLLM_BASE_URL (default http://localhost:8000/v1), this script starts one and tears it down on exit; a vLLM you serve yourself is detected and reused.
# Bedrock (*-amazon) and Vertex (*-gcp) models are called through their APIs.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

# Locate conda's profile script so activation works under sbatch and nohup.
_csh="${CONDA_PROFILE:-}"
[ -z "$_csh" ] && [ -n "${CONDA_EXE:-}" ] && _csh="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
[ -z "$_csh" ] && _csh="$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f "$_csh" ] && source "$_csh"

export PYTHONUNBUFFERED=1
export SKILL_INDUCTION_NO_CACHE="${SKILL_INDUCTION_NO_CACHE:-1}"
export VLLM_REPETITION_PENALTY="${VLLM_REPETITION_PENALTY:-1.1}"
export SKILL_INDUCTION_VLLM_THINK="${SKILL_INDUCTION_VLLM_THINK:-0}"   # local: no-think

AGENTS="${AGENTS:-linear linear_awm linear_asi subtask subtask_awm subtask_asi}"
MODELS="${MODELS:-qwen-3-235b-a22b-amazon nemotron-3-super-120b-amazon gpt-oss-120b-amazon}"
BENCHES="${BENCHES:-appworld kramabench officebench}"
STAGE="${STAGE:-final_v1}"
N="${N:-}"
PILOT="${PILOT:-false}"
PILOT_FILE="${PILOT_FILE:-}"
OVERRIDE="${OVERRIDE:-false}"
SHUFFLE_SEED="${SHUFFLE_SEED:-}"
INDUCTION_LEVEL="${INDUCTION_LEVEL:-3}"   # 1=instr / 2=instr+demo / 3=+rules (deployed)
# Skill-transfer arm: the linear agent replays the matching subtask agent's skill library (linear_awm from subtask_awm, linear_asi from subtask_asi).
# Only valid with AGENTS drawn from {linear_awm, linear_asi}; the donor is the same model+benchmark at stage SUBTASK_SKILLS_STAGE.
SUBTASK_SKILLS="${SUBTASK_SKILLS:-false}"
SUBTASK_SKILLS_STAGE="${SUBTASK_SKILLS_STAGE:-final_v1}"
MAXJOBS="${MAXJOBS:-6}"
ENV_NAME="${ENV_NAME:-llm-agent-env}"
APPWORLD_CLEANUP="${APPWORLD_CLEANUP:-true}"
# Conda env with vLLM installed; point VLLM_ENV at yours.
VLLM_ENV="${VLLM_ENV:-/path/to/envs/vllm}"
mkdir -p exp_logs

# Progress subcommand: glanceable bars per (model x benchmark).
#   bash scripts/exp.sh progress         # STAGE=final_v1 by default
if [ "${1:-}" = progress ]; then
  STAGE="$STAGE" conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import pickle, glob, os, re, collections
STAGE=os.environ.get("STAGE","final_v1")
from skill_induction.experiments.run import _load_default_taskset, _BENCHMARK_ALIASES
AG=["linear","linear_awm","linear_asi","subtask","subtask_awm","subtask_asi"]
def short(m):
    import re as _r
    return _r.sub(r'-amazon$','',m).replace('qwen-3-','q').replace('gemma-3-','g').replace('nemotron-3-super-','nemo').replace('gpt-oss-','gpt').replace('-a22b','').replace('-120b','').replace('-local','L')
cells=collections.defaultdict(dict); benches=set(); models=set()
for p in glob.glob(f"data/*_{STAGE}.pkl"):
    if "memory" in p: continue
    name=os.path.basename(p)[:-(len(STAGE)+5)]
    for a in sorted(AG,key=len,reverse=True):
        mm=re.match(rf"(.+)_{a}_(.+)$",name)
        if mm:
            b,mod=mm.group(1),mm.group(2)
            r=pickle.load(open(p,"rb"))
            cells[(b,mod)][a]=(sum(1 for x in r if not x.get("errored")),
                               sum(1 for x in r if x.get("errored")))
            benches.add(b); models.add(mod); break
def full(b):
    c,s=_BENCHMARK_ALIASES.get(b,(b,None))
    try: return len(_load_default_taskset(c,None,s))
    except: return 0
def bar(x,t,w=20): f=int(w*x/t) if t else 0; return "█"*f+"░"*(w-f)
print(f"PROGRESS  stage={STAGE}   (bar = common subset = slowest agent of the 6)")
gtot=gden=0
for b in sorted(benches):
    t=full(b); print(f"\n{b}  /{t}")
    for mod in sorted(models, key=short):
        d=cells.get((b,mod))
        if not d: continue
        oks=[d[a][0] for a in AG if a in d]; ers=sum(d[a][1] for a in AG if a in d)
        common=min(oks) if len(d)==6 else 0
        gtot+=sum(oks); gden+=6*t
        cnts="/".join(str(d[a][0]) if a in d else "-" for a in AG)
        print(f"  {short(mod):7s} [{bar(common,t)}] {common:>4}/{t:<4} cells:{cnts}"+(f"  ({ers} err)" if ers else ""))
print(f"\nOVERALL scored: {gtot}/{gden}  ({100*gtot/gden:.0f}%)   [agents: {'/'.join(AG)}]")
PY
  exit 0
fi

# Ablation subcommand: induction-prompt component ladder.
#   bash scripts/exp.sh ablation
# Runs induction levels 1 (minimal instruction) and 2 (plus one worked example) over the memory agents.
# Level 3 (the full deployed prompt) is not re-run here; it is the existing final_v1 grid.
# Bare linear/subtask are memory-free and induction-level-invariant, so reuse their final_v1 baselines.
# Each level lands in its own stage pkl: <bench>_<agent>_<model>_<prefix>_L{1,2}.pkl
# Override knobs: LEVELS, MODELS, AGENTS, BENCHES, N, OVERRIDE, STAGE_PREFIX.
if [ "${1:-}" = ablation ]; then
  AB_LEVELS="${LEVELS:-1 2}"
  AB_PREFIX="${STAGE_PREFIX:-prompt_abl2}"
  AB_MODELS="${MODELS:-qwen-3-235b-a22b-amazon nemotron-3-super-120b-amazon gpt-oss-120b-amazon qwen-3-32b-local}"
  AB_AGENTS="${AGENTS:-linear_awm linear_asi subtask_awm subtask_asi}"
  for lvl in $AB_LEVELS; do
    echo "[exp $(date +%T)] ===== ablation: induction-level $lvl -> stage ${AB_PREFIX}_L${lvl} ====="
    INDUCTION_LEVEL="$lvl" STAGE="${AB_PREFIX}_L${lvl}" \
      MODELS="$AB_MODELS" AGENTS="$AB_AGENTS" \
      bash "$0"
  done
  echo "[exp $(date +%T)] ablation done"
  exit 0
fi

# Fail fast if a Bedrock (*-amazon) model is selected but the node has no outbound HTTPS.
if echo " $MODELS " | grep -q -- '-amazon '; then
  curl -s --max-time 10 -o /dev/null https://bedrock-runtime.us-west-2.amazonaws.com 2>/dev/null \
    || { echo "error: a Bedrock model is selected but $(hostname) has no outbound HTTPS." >&2; exit 2; }
fi

# Fail fast if a Vertex (*-gcp) model is selected but the node cannot reach googleapis.com.
# Gemini needs googleapis.com for both the API and ADC token refresh; compute nodes may not reach it even when AWS endpoints are allowlisted.
if echo " $MODELS " | grep -q -- '-gcp '; then
  curl -s --max-time 10 -o /dev/null https://aiplatform.googleapis.com 2>/dev/null \
    || { echo "error: a Gemini/Vertex (-gcp) model is selected but $(hostname) has no outbound HTTPS to googleapis.com." >&2; exit 2; }
fi

# Start a vLLM for the (first) local model if none is already serving.
# Under slurm the port is derived from the job id, so two jobs sharing a node do not collide; standalone runs keep 8000.
VLLM_PID=""
LOCAL_M="$(echo "$MODELS" | tr ' ' '\n' | grep -- '-local' | head -1 || true)"
if [ -n "$LOCAL_M" ]; then
  VLLM_PORT="${VLLM_PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 992 ))}"
  export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
  SERVED="$(conda run -n "$ENV_NAME" python -c \
    "from skill_induction.models.vllm_local import LOCAL_MODEL_CONFIG as C; print(C['$LOCAL_M'])" 2>/dev/null)"
  SERVE_ARGS="$(conda run -n "$ENV_NAME" python -c \
    "from skill_induction.models.vllm_local import vllm_serve_args as f; print(f('$LOCAL_M'))" 2>/dev/null)"
  if ! curl -s --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null | grep -qF "$SERVED"; then
    echo "[exp $(date +%T)] starting vLLM: $SERVED (TP=${EXP_TP:-1} port=$VLLM_PORT args: $SERVE_ARGS) ..."
    ( set +u; source "$_csh"; conda activate "$VLLM_ENV"
      exec vllm serve "$SERVED" $SERVE_ARGS \
        --tensor-parallel-size "${EXP_TP:-1}" --port "$VLLM_PORT" ) > "exp_logs/vllm_${LOCAL_M}.log" 2>&1 &
    VLLM_PID=$!
    trap '[ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null' EXIT
    for i in $(seq 1 100); do sleep 15
      curl -s --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null | grep -qF "$SERVED" && { echo "[exp $(date +%T)] vLLM ready (~$((i*15))s)"; break; }
    done
  fi
  curl -s --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null | grep -qF "$SERVED" \
    || { echo "error: vLLM for $LOCAL_M not ready; see exp_logs/vllm_${LOCAL_M}.log" >&2; exit 1; }
fi

cell(){ local b=$1 a=$2 m=$3
  local args="--benchmark $b --agent $a --model $m --stage $STAGE"
  args="$args --induction-level $INDUCTION_LEVEL"
  [ -n "$N" ] && args="$args -N $N"
  [ "$PILOT" = true ] && args="$args --pilot${PILOT_FILE:+ --pilot-file $PILOT_FILE}"
  [ -n "$SHUFFLE_SEED" ] && args="$args --shuffle-seed $SHUFFLE_SEED"
  [ "$SUBTASK_SKILLS" = true ] && args="$args --subtask-skills --subtask-skills-stage $SUBTASK_SKILLS_STAGE"
  # Auto-resume on a non-zero exit: the runner checkpoints per task, so an uncaught env crash is recovered by re-running without --override (done tasks are skipped).
  # --override applies to the first attempt only; a resume retry must never wipe partial progress.
  # Bounded by CELL_RETRIES (default 6) so a deterministic poison task cannot loop forever; clean completion returns 0 and breaks immediately.
  local log="exp_logs/${b}_${a}_${m}_${STAGE}.log"
  local ov=""; [ "$OVERRIDE" = true ] && ov="--override"
  local tries="${CELL_RETRIES:-6}" k=0
  while :; do
    if conda run --no-capture-output -n "$ENV_NAME" python -u -m skill_induction.experiments.run $args $ov \
         >> "$log" 2>&1; then
      break
    fi
    ov=""; k=$((k+1))
    if [ "$k" -ge "$tries" ]; then
      echo "[$(date +%T)] cell gave up after $tries attempts (still crashing): $b/$a/$m" | tee -a "$log"
      break
    fi
    echo "[$(date +%T)] cell runner crashed (exit != 0); resume retry $k/$tries: $b/$a/$m" | tee -a "$log"
    sleep 5
  done
  echo "[$(date +%T)] cell done: $b/$a/$m"
}

echo "[exp] agents=[$AGENTS] models=[$MODELS] benches=[$BENCHES] stage=$STAGE ind_level=$INDUCTION_LEVEL subtask_skills=$SUBTASK_SKILLS N=${N:-all} pilot=$PILOT override=$OVERRIDE maxjobs=$MAXJOBS"
for b in $BENCHES; do
  stag=1; [ "$b" = appworld ] && stag="${APPWORLD_STAGGER:-25}"
  cell_pids=()
  for m in $MODELS; do
    for a in $AGENTS; do
      # Throttle on live cell jobs only; `jobs -rp` would also count the auto-started vLLM server child and mis-throttle by one.
      while : ; do
        live=0; for p in "${cell_pids[@]}"; do kill -0 "$p" 2>/dev/null && live=$((live+1)); done
        [ "$live" -lt "$MAXJOBS" ] && break; sleep 5
      done
      echo "[$(date +%T)] launch $b/$a/$m"
      cell "$b" "$a" "$m" & cell_pids+=($!); sleep "$stag"
    done
  done
  # Wait only for this bench's cells; a bare `wait` would also wait on the never-exiting vLLM server child until the walltime limit.
  [ "${#cell_pids[@]}" -gt 0 ] && wait "${cell_pids[@]}"
  [ "$APPWORLD_CLEANUP" = true ] && [ "$b" = appworld ] && { pkill -u "$USER" -f 'appworld serve' 2>/dev/null || true; }
done
echo "[exp $(date +%T)] done"

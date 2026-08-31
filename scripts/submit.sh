#!/usr/bin/env bash
# Submit one sbatch job per (model x benchmark) for the selected matrix.
# Each job allocates GPUs per the per-model config and runs scripts/exp.sh, which auto-starts a vLLM for local models and fails fast for API models if the node has no internet.
#
# Usage: bash scripts/submit.sh, with knobs passed as env vars, e.g.
#   MODELS="qwen-3-32b-local" BENCHES="appworld" N=50 bash scripts/submit.sh
#   AGENTS="subtask_asi linear_asi" STAGE=final_v1 bash scripts/submit.sh
#
# Knobs: AGENTS MODELS BENCHES STAGE N PILOT PILOT_FILE OVERRIDE ACCOUNT PART TIME CPUS MEM MAXJOBS
# GPUs per model default to 1 for *-local and 0 for API models; override with MODEL_GPUS="qwen-3-32b-local=2 some-big-local=4" or DEFAULT_LOCAL_GPUS=2.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJ="$(cd "$S/.." && pwd)"

AGENTS="${AGENTS:-linear linear_awm linear_asi subtask subtask_awm subtask_asi}"
MODELS="${MODELS:-qwen-3-235b-a22b-amazon nemotron-3-super-120b-amazon gpt-oss-120b-amazon qwen-3-32b-local}"
BENCHES="${BENCHES:-appworld kramabench officebench}"
STAGE="${STAGE:-final_v1}"
N="${N:-}"
PILOT="${PILOT:-false}"
PILOT_FILE="${PILOT_FILE:-}"
OVERRIDE="${OVERRIDE:-false}"
SHUFFLE_SEED="${SHUFFLE_SEED:-}"
INDUCTION_LEVEL="${INDUCTION_LEVEL:-3}"   # 1=instr / 2=instr+demo / 3=+rules (deployed)
SUBTASK_SKILLS="${SUBTASK_SKILLS:-false}" # linear agent on subtask skill library
SUBTASK_SKILLS_STAGE="${SUBTASK_SKILLS_STAGE:-final_v1}"
BEDROCK_READ_TIMEOUT="${BEDROCK_READ_TIMEOUT:-300}"
# Fill in your slurm account and GPU partition here or via env.
ACCOUNT="${ACCOUNT:-YOUR_ACCOUNT}"
PART="${PART:-YOUR_PARTITION}"
TIME="${TIME:-12:00:00}"   # short walltime backfills better at low fairshare; resume by re-running without --override
CPUS="${CPUS:-16}"
mkdir -p "$PROJ/logs"

gpus_for(){ local m=$1 kv
  for kv in ${MODEL_GPUS:-}; do [ "${kv%%=*}" = "$m" ] && { echo "${kv#*=}"; return; }; done
  case "$m" in *-local) echo "${DEFAULT_LOCAL_GPUS:-1}";; *) echo "${DEFAULT_BEDROCK_GPUS:-0}";; esac
}
short(){ echo "$1" | sed -E 's/-amazon$//; s/qwen-3-([0-9]+b).*/q\1/; s/gemma-3-([0-9]+b).*/g\1/; s/nemotron.*/nemo/; s/gpt-oss.*/gpt/; s/-local$/L/'; }

for m in $MODELS; do
  g="$(gpus_for "$m")"
  mem="${MEM:-$([ "$g" -gt 0 ] && echo 128G || echo 96G)}"
  mj="${MAXJOBS:-$([ "$g" -gt 0 ] && echo 4 || echo 6)}"
  for b in $BENCHES; do
    lsfx=""; [ "$INDUCTION_LEVEL" = 3 ] || lsfx="-L${INDUCTION_LEVEL}"
    [ "$SUBTASK_SKILLS" = true ] && lsfx="${lsfx}-ss"
    jn="j-${b}-$(short "$m")${lsfx}"
    sbatch --account="$ACCOUNT" --partition="$PART" --time="$TIME" --nodes=1 --ntasks=1 \
           --cpus-per-task="$CPUS" --mem="$mem" --gpus-per-node="$g" -J "$jn" \
           --output="$PROJ/logs/slurm_%x_%j.out" --error="$PROJ/logs/slurm_%x_%j.err" <<EOF
#!/usr/bin/env bash
cd "$PROJ"
echo "[\$(date +%T)] node=\$(hostname) job=$jn bench=$b model=$m gpus=$g maxjobs=$mj"
AGENTS="$AGENTS" MODELS="$m" BENCHES="$b" STAGE="$STAGE" N="$N" \
PILOT="$PILOT" PILOT_FILE="$PILOT_FILE" OVERRIDE="$OVERRIDE" SHUFFLE_SEED="$SHUFFLE_SEED" \
INDUCTION_LEVEL="$INDUCTION_LEVEL" BEDROCK_READ_TIMEOUT="$BEDROCK_READ_TIMEOUT" \
SUBTASK_SKILLS="$SUBTASK_SKILLS" SUBTASK_SKILLS_STAGE="$SUBTASK_SKILLS_STAGE" \
EXP_TP="$g" MAXJOBS="$mj" APPWORLD_CLEANUP=false \
  bash "$PROJ/scripts/exp.sh"
echo "[\$(date +%T)] job done $jn"
EOF
    echo "submitted: $jn  (gpus=$g mem=$mem maxjobs=$mj)"
  done
done
echo "--- queue ---"
squeue -u "$USER" -o "%.10i %.20j %.9T %.5D %.10M %R" 2>/dev/null | grep -E " j-" || true

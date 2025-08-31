#!/usr/bin/env bash
# run_lmeval.sh — wrapper for lm_eval with clear "task type" groupings.
#
# Quick start (先从生成类开始，稳定、兼容 chat-completions)：
#   ./run_lmeval.sh -t gen -l 10
#
# 说明：
# - gen 组 = gsm8k, svamp, asdiv, aqua, math（生成式评测，chat-completions 可跑）
# - mc  组 = mmlu, hellaswag, arc_challenge, truthfulqa_mc2, winogrande, boolq, piqa
#           （判别/多选，需要 completions 后端支持 loglikelihood；先别跑这个）
# - 也可直接用 -t "task1,task2" 自定义列表
# - 默认使用 $OPENAI_API_KEY；也可用 -m 或 LM_MODEL_ARGS 覆盖 --model_args
# - 默认 LIMIT=20（快速冒烟）；全量请 -l 0

set -euo pipefail

# ===== 默认参数（快） =====
OUTPUT_DIR="./"
LIMIT=20
BATCH_SIZE=1
SEED=42
OVERRIDE_SHOTS=""
TASKS=""
APPLY_CHAT_TEMPLATE=1
BACKEND="chat"      # chat | comp
MAX_GEN_TOKS=128
MODEL_ARGS="${LM_MODEL_ARGS:-}"

usage() {
  cat <<'EOF'
Usage: run_lmeval.sh [options]

Options:
  -t TASKS     逗号分隔任务或分组名：gen | mc | core
               gen = gsm8k,svamp,asdiv,aqua,math
               mc  = mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,boolq,piqa
               core = gen + mc
  -M BACKEND   chat(默认，用于 gen) | comp(用于 mc，需要 completions 后端)
  -s SHOTS     覆盖所有任务的 few-shot（默认按任务启发式）
  -l LIMIT     每任务样本数（0=全量；默认20）
  -o OUTDIR    输出目录（默认 ./）
  -b BS        batch_size（默认1）
  -r SEED      随机种子（默认42）
  -m ARGS      完整 --model_args 字符串
  -G TOKS      --max_gen_toks（默认128，仅生成任务有用）
  -n           不加 --apply_chat_template
  -h           帮助

示例：
  ./run_lmeval.sh -t gen -l 10
EOF
}

# 解析参数
while getopts ":t:M:s:l:o:b:r:m:G:nh" opt; do
  case $opt in
    t) TASKS="$OPTARG" ;;
    M) BACKEND="$OPTARG" ;;
    s) OVERRIDE_SHOTS="$OPTARG" ;;
    l) LIMIT="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    r) SEED="$OPTARG" ;;
    m) MODEL_ARGS="$OPTARG" ;;
    G) MAX_GEN_TOKS="$OPTARG" ;;
    n) APPLY_CHAT_TEMPLATE=0 ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "ERROR: lm_eval not found. pip install 'lm-eval[api]' first." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

expand_group() {
  case "$1" in
    gen)  echo "gsm8k,svamp,asdiv,aqua,math" ;;
    mc)   echo "mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,boolq,piqa" ;;
    core) echo "mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,boolq,piqa,gsm8k,svamp,asdiv,aqua,math" ;;
    *)    echo "$1" ;;
  esac
}

if [[ -z "${TASKS}" ]]; then
  echo "ERROR: Please specify -t TASKS（如 -t gen）." >&2
  usage
  exit 2
fi
TASKS="$(expand_group "${TASKS}")"
IFS=',' read -r -a TASK_ARR <<< "$(echo "${TASKS}" | tr -d '[:space:]')"

# 判别类任务判断
is_mc_task() {
  case "$1" in
    mmlu|hellaswag|arc_challenge|truthfulqa_mc2|winogrande|boolq|piqa) return 0 ;;
    *) return 1 ;;
  esac
}

# 选择后端
MODEL_KIND="local-chat-completions"
if [[ "${BACKEND}" == "comp" ]]; then
  MODEL_KIND="local-completions"
fi

# 默认 --model_args
if [[ -z "${MODEL_ARGS}" ]]; then
  if [[ "${BACKEND}" == "chat" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "WARNING: \$OPENAI_API_KEY 未设置。也可用 -m 覆盖 --model_args." >&2
    fi
    MODEL_ARGS="model=deepseek-chat,base_url=https://api.deepseek.com/v1/chat/completions,api_key=${OPENAI_API_KEY:-},eos_string=<|im_end|>"
  else
    echo "ERROR: -M comp 需要你用 -m 提供 completions 端点（支持 logprobs）。" >&2
    exit 2
  fi
fi

# 防呆：chat 后端不能跑 mc 任务
if [[ "${BACKEND}" == "chat" ]]; then
  for t in "${TASK_ARR[@]}"; do
    if is_mc_task "$t"; then
      echo "ERROR: 任务 '$t' 需要 loglikelihood，不能用 chat 后端。请用 -M comp。" >&2
      exit 2
    fi
  done
fi

# few-shot 启发式
default_shots_for_task() {
  case "$1" in
    # gen
    gsm8k|svamp|asdiv|aqua) echo 0 ;;
    math) echo 4 ;;
    # mc
    mmlu) echo 5 ;;
    hellaswag) echo 10 ;;
    arc_challenge) echo 25 ;;
    truthfulqa_mc2) echo 0 ;;
    winogrande) echo 5 ;;
    boolq) echo 3 ;;
    piqa) echo 5 ;;
    *) echo 0 ;;
  esac
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

for task in "${TASK_ARR[@]}"; do
  if [[ -n "${OVERRIDE_SHOTS}" ]]; then
    SHOTS="${OVERRIDE_SHOTS}"
  else
    SHOTS="$(default_shots_for_task "${task}")"
  fi
  OUTFILE="${OUTPUT_DIR%/}/results_${task}_$(timestamp).json"

  echo ">>> Running backend=${MODEL_KIND} task=${task} shots=${SHOTS} limit=${LIMIT} -> ${OUTFILE}"

  cmd=(
    lm_eval
    --model "${MODEL_KIND}"
    --model_args "${MODEL_ARGS}"
    --tasks "${task}"
    --num_fewshot "${SHOTS}"
    --limit "${LIMIT}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --output_path "${OUTFILE}"
    --gen_kwargs "max_tokens=${MAX_GEN_TOKS}"

  )

  if [[ "${BACKEND}" == "chat" && "${APPLY_CHAT_TEMPLATE}" -eq 1 ]]; then
    cmd+=( --apply_chat_template --fewshot_as_multiturn )
  fi

  "${cmd[@]}"
done

echo "All done."

#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$(pwd)}"

# Core run selection. STAGE can be: pretrain, probe, probe_simple, probe_dino,
# probe_all, both, both_simple, both_dino, both_all.
export METHOD="${METHOD:-lejepa}"
case "${METHOD}" in
  lejepa)
    ;;
  MultiCW|multicw|multi_cw|multi-cw)
    export METHOD="multicw"
    ;;
  JointCW|jointcw|joint_cw|joint-cw)
    export METHOD="jointcw"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}; expected lejepa, multicw, or jointcw." >&2
    exit 2
    ;;
esac
export STAGE="${STAGE:-both}"
export PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${ROOT_DIR}/benchmarks/imagenet100/lejepa_pretrain.yaml}"
export PROBE_CONFIG="${PROBE_CONFIG:-${ROOT_DIR}/benchmarks/imagenet100/lejepa_linear_probe.yaml}"
export DINO_PROBE_CONFIG="${DINO_PROBE_CONFIG:-${ROOT_DIR}/benchmarks/imagenet100/lejepa_linear_probe_dino_style.yaml}"

# Dataset and output locations. DATASET selects a preset; low-level vars below
# remain overrideable for custom datasets.
export DATASET="${DATASET:-imagenet10}"
case "${DATASET}" in
  imagenet10|imagenette)
    DEFAULT_DATASET_PATH="frgfm/imagenette"
    DEFAULT_DATASET_CACHE_NAME="imagenet10"
    DEFAULT_NUM_CLASSES="10"
    DEFAULT_RUN_DATASET="inet10"
    DEFAULT_DATASET_REVISION="refs/convert/parquet"
    DEFAULT_WANDB_PROJECT="imagenet10-lejepa"
    ;;
  imagenet100)
    DEFAULT_DATASET_PATH="clane9/imagenet-100"
    DEFAULT_DATASET_CACHE_NAME="imagenet100"
    DEFAULT_NUM_CLASSES="100"
    DEFAULT_RUN_DATASET="inet100"
    DEFAULT_DATASET_REVISION="null"
    DEFAULT_WANDB_PROJECT="imagenet100-lejepa"
    ;;
  custom)
    if [[ -z "${DATASET_PATH:-}" || -z "${NUM_CLASSES:-}" ]]; then
      echo "DATASET=custom requires DATASET_PATH and NUM_CLASSES." >&2
      exit 2
    fi
    DEFAULT_DATASET_PATH="${DATASET_PATH}"
    DEFAULT_DATASET_CACHE_NAME="${DATASET_CACHE_NAME:-custom}"
    DEFAULT_NUM_CLASSES="${NUM_CLASSES}"
    DEFAULT_RUN_DATASET="custom"
    DEFAULT_DATASET_REVISION="${DATASET_REVISION:-null}"
    DEFAULT_WANDB_PROJECT="lejepa-custom"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}; expected imagenet10, imagenet100, or custom." >&2
    exit 2
    ;;
esac

export STORAGE_ROOT="${STORAGE_ROOT:-${HOME}/storage}"
export STABLE_PRETRAINING_DATA_DIR="${STABLE_PRETRAINING_DATA_DIR:-${STORAGE_ROOT}/datasets/stable-pretraining}"
export DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
export DATASET_CACHE_NAME="${DATASET_CACHE_NAME:-${DEFAULT_DATASET_CACHE_NAME}}"
export DATASET_REVISION="${DATASET_REVISION:-${DEFAULT_DATASET_REVISION}}"
export NUM_CLASSES="${NUM_CLASSES:-${DEFAULT_NUM_CLASSES}}"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export VAL_SPLIT="${VAL_SPLIT:-validation}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-${DATASET_CACHE_NAME}}"
export IN_DOMAIN_DATASET_NAME="${IN_DOMAIN_DATASET_NAME:-${DATASET_CACHE_NAME}}"

if [[ "${USE_NVME_DATASET:-0}" == "1" ]]; then
  export LOCAL_SCRATCH="${LOCAL_SCRATCH:-${TMPDIR_LOCAL:-${TMPDIR:-/tmp/${USER:-user}-${SLURM_JOB_ID:-lejepa}}}}"
  NVME_DATA_ROOT="${LOCAL_SCRATCH}/datasets/stable-pretraining"
  NVME_HF_HOME="${LOCAL_SCRATCH}/cache/huggingface"
  mkdir -p "${NVME_DATA_ROOT}" "${NVME_HF_HOME}"

  if [[ "${DATASET_BACKEND:-hf}" == "lance" ]]; then
    if [[ ! -d "${DATASET_PATH}" ]]; then
      echo "USE_NVME_DATASET=1 with DATASET_BACKEND=lance requires DATASET_PATH to be an existing directory: ${DATASET_PATH}" >&2
      exit 2
    fi
    NVME_LANCE_PATH="${NVME_DATA_ROOT}/$(basename "${DATASET_PATH}")"
    if [[ ! -e "${NVME_LANCE_PATH}/.nvme_stage_complete" ]]; then
      rm -rf "${NVME_LANCE_PATH}"
      mkdir -p "$(dirname "${NVME_LANCE_PATH}")"
      cp -a "${DATASET_PATH}" "${NVME_LANCE_PATH}"
      touch "${NVME_LANCE_PATH}/.nvme_stage_complete"
    fi
    export DATASET_PATH="${NVME_LANCE_PATH}"
  else
    STORAGE_DATASET_CACHE="${STABLE_PRETRAINING_DATA_DIR}/${DATASET_CACHE_NAME}"
    NVME_DATASET_CACHE="${NVME_DATA_ROOT}/${DATASET_CACHE_NAME}"
    if [[ ! -d "${STORAGE_DATASET_CACHE}" ]]; then
      echo "USE_NVME_DATASET=1 requires an existing storage dataset cache: ${STORAGE_DATASET_CACHE}" >&2
      exit 2
    fi
    if [[ ! -e "${NVME_DATASET_CACHE}/.nvme_stage_complete" ]]; then
      rm -rf "${NVME_DATASET_CACHE}"
      mkdir -p "$(dirname "${NVME_DATASET_CACHE}")"
      cp -a "${STORAGE_DATASET_CACHE}" "${NVME_DATASET_CACHE}"
      touch "${NVME_DATASET_CACHE}/.nvme_stage_complete"
    fi
    export STABLE_PRETRAINING_DATA_DIR="${NVME_DATA_ROOT}"
  fi

  export HF_HOME="${NVME_HF_HOME}"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  echo "NVME dataset staging enabled: LOCAL_SCRATCH=${LOCAL_SCRATCH}"
  echo "STABLE_PRETRAINING_DATA_DIR=${STABLE_PRETRAINING_DATA_DIR}"
  echo "DATASET_PATH=${DATASET_PATH}"
fi

export SEED="${SEED:-42}"
export RUN_GROUP="${RUN_GROUP:-${METHOD}-vits-${DEFAULT_RUN_DATASET}-e${EPOCHS:-20}-g${DEVICES:-1}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/results/le/stable_pretraining/${RUN_GROUP}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/seed${SEED}}"
export PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${OUTPUT_DIR}/linear_probe}"
export DINO_PROBE_OUTPUT_DIR="${DINO_PROBE_OUTPUT_DIR:-${OUTPUT_DIR}/linear_probe_dino_style}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-${STORAGE_ROOT}/results/le/stable_pretraining/.cache}"
export RUN_MANIFEST="${RUN_MANIFEST:-${OUTPUT_DIR}/pretrain_run.yaml}"
export CONFIG_ARCHIVE_DIR="${CONFIG_ARCHIVE_DIR:-${OUTPUT_DIR}/configs}"

# Model and LeJEPA loss.
export ENCODER_NAME="${ENCODER_NAME:-vit_small_patch16_224}"
export EMBED_DIM="${EMBED_DIM:-384}"
export PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-0}"
export DROP_PATH_RATE="${DROP_PATH_RATE:-0.1}"
export SIGREG="${SIGREG:-ep}"
export APPLY_SIGREG_ON="${APPLY_SIGREG_ON:-all}"
export RANDOM_CLUSTER_SEED="${RANDOM_CLUSTER_SEED:-0}"
export OVERRIDE_SR_GAMMA="${OVERRIDE_SR_GAMMA:-null}"
export LAMB="${LAMB:-0.02}"
export N_SLICES="${N_SLICES:-1024}"
export N_POINTS="${N_POINTS:-17}"
export T_MAX="${T_MAX:-3.0}"

# MultiCW loss. Select it with:
#   METHOD=multicw STAGE=pretrain ./run_lejepa.sh
# Pairwise JointCW remains available separately with ``METHOD=jointcw``.
export N_GLOBAL="${N_GLOBAL:-2}"
export N_LOCAL="${N_LOCAL:-6}"
export CW_VARIANT="${CW_VARIANT:-canonical}"
export RHO_GG="${RHO_GG:-0.88}"
export RHO_GL="${RHO_GL:-0.72}"
export RHO_LL="${RHO_LL:-0.61}"
export W_JOINT="${W_JOINT:-0.60}"
export W_COLLECTIVE_MAJOR="${W_COLLECTIVE_MAJOR:-0.25}"
export W_COLLECTIVE_MINOR="${W_COLLECTIVE_MINOR:-0.05}"
export W_GLOBAL_RESIDUAL="${W_GLOBAL_RESIDUAL:-0.03}"
export W_LOCAL_RESIDUAL="${W_LOCAL_RESIDUAL:-0.07}"

export JCW_BETA="${JCW_BETA:-${BETA:-1.0}}"
export JCW_W_PLUS="${JCW_W_PLUS:-${W_PLUS:-null}}"

# The YAML uses the standard LeJEPA 2 global + 6 local crop recipe.
# To change the number of crops, edit benchmarks/imagenet100/lejepa_pretrain.yaml.

METHOD_ARGS=()
if [[ "${METHOD}" == "multicw" ]]; then
  METHOD_ARGS=(
    "module.forward=stable_pretraining.forward.multi_cw"
    "module.model._target_=stable_pretraining.methods.MultiCW"
    "~module.model.sigreg"
    "~module.model.apply_sigreg_on"
    "~module.model.random_cluster_seed"
    "~module.model.lamb"
    "~module.model.n_slices"
    "~module.model.n_points"
    "~module.model.t_max"
    "+module.model.n_global=${N_GLOBAL}"
    "+module.model.n_local=${N_LOCAL}"
    "+module.model.cw_variant=${CW_VARIANT}"
    "+module.model.rho_gg=${RHO_GG}"
    "+module.model.rho_gl=${RHO_GL}"
    "+module.model.rho_ll=${RHO_LL}"
    "+module.model.w_joint=${W_JOINT}"
    "+module.model.w_collective_major=${W_COLLECTIVE_MAJOR}"
    "+module.model.w_collective_minor=${W_COLLECTIVE_MINOR}"
    "+module.model.w_global_residual=${W_GLOBAL_RESIDUAL}"
    "+module.model.w_local_residual=${W_LOCAL_RESIDUAL}"
  )
elif [[ "${METHOD}" == "jointcw" ]]; then
  METHOD_ARGS=(
    "module.forward=stable_pretraining.forward.joint_cw"
    "module.model._target_=stable_pretraining.methods.JointCW"
    "~module.model.sigreg"
    "~module.model.apply_sigreg_on"
    "~module.model.random_cluster_seed"
    "~module.model.lamb"
    "~module.model.n_slices"
    "~module.model.n_points"
    "~module.model.t_max"
    "+module.model.rho_gg=${RHO_GG}"
    "+module.model.rho_gl=${RHO_GL}"
    "+module.model.rho_ll=${RHO_LL}"
    "+module.model.beta=${JCW_BETA}"
    "+module.model.w_plus=${JCW_W_PLUS}"
  )
fi

# Pretraining optimization.
export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export LR="${LR:-4e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.1}"
export END_LR="${END_LR:-4e-6}"
export PRECISION="${PRECISION:-16-mixed}"
export ACCELERATOR="${ACCELERATOR:-gpu}"
if [[ "${DEVICES:-1}" -gt 1 ]]; then
  export STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"
  export SYNC_BATCHNORM="${SYNC_BATCHNORM:-1}"
else
  export STRATEGY="${STRATEGY:-auto}"
  export SYNC_BATCHNORM="${SYNC_BATCHNORM:-0}"
fi
export PROBE_STRATEGY="${PROBE_STRATEGY:-${STRATEGY}}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-20}"

# Post-training linear probe.
export PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
export PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-${BATCH_SIZE}}"
export PROBE_LR="${PROBE_LR:-0.03}"
export PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-1e-6}"
export DINO_PROBE_EPOCHS="${DINO_PROBE_EPOCHS:--1}"
export DINO_PROBE_MAX_STEPS="${DINO_PROBE_MAX_STEPS:-12500}"
export DINO_PROBE_VAL_CHECK_INTERVAL="${DINO_PROBE_VAL_CHECK_INTERVAL:-1250}"
export DINO_PROBE_LRS="${DINO_PROBE_LRS:-[1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,0.1]}"
export DINO_PROBE_N_LAST_BLOCKS="${DINO_PROBE_N_LAST_BLOCKS:-[1,2,4]}"
export DINO_PROBE_POOLS="${DINO_PROBE_POOLS:-[cls,avgpool]}"
export DINO_PROBE_OPTIMIZERS="${DINO_PROBE_OPTIMIZERS:-[sgd,adamw]}"
export DINO_PROBE_RRC_SCALE_MIN="${DINO_PROBE_RRC_SCALE_MIN:-0.08}"
export DINO_PROBE_RRC_SCALE_MAX="${DINO_PROBE_RRC_SCALE_MAX:-1.0}"
export BACKBONE_CKPT="${BACKBONE_CKPT:-${OUTPUT_DIR}/backbone.pt}"
# Set CKPT_PATH=/absolute/path/to/last.ckpt when exporting a backbone from
# an existing pretraining run without a manifest.

# Logging. LOGGER can be csv, wandb, or none.
export LOGGER="${LOGGER:-csv}"
export WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
export WANDB_PROJECT="${WANDB_PROJECT:-spt_cw_jepa}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_GROUP}}"
export WANDB_BASE_TAGS="${WANDB_BASE_TAGS:-${WANDB_TAGS:-}}"

mkdir -p "${OUTPUT_DIR}" "${PROBE_OUTPUT_DIR}" "${DINO_PROBE_OUTPUT_DIR}" "${CONFIG_ARCHIVE_DIR}"

LOGGER_ARGS=()
build_logger_args() {
  local run_stage="$1"
  local run_name="${RUN_NAME:-${RUN_GROUP}-seed${SEED}-${run_stage}}"
  local wandb_tags="${WANDB_BASE_TAGS}"
  local stage_tag="stage_${run_stage}"
  export CURRENT_RUN_NAME="${run_name}"

  wandb_tags="${wandb_tags#[}"
  wandb_tags="${wandb_tags%]}"
  if [[ -z "${wandb_tags}" ]]; then
    wandb_tags="${stage_tag}"
  else
    wandb_tags="${wandb_tags},${stage_tag}"
  fi
  export WANDB_TAGS="${wandb_tags}"

  LOGGER_ARGS=()
  if [[ "${LOGGER}" == "wandb" ]]; then
    LOGGER_ARGS=(
      "trainer.logger._target_=lightning.pytorch.loggers.WandbLogger"
      "+trainer.logger.project=${WANDB_PROJECT}"
      "+trainer.logger.entity=${WANDB_ENTITY}"
      "+trainer.logger.group=${WANDB_GROUP}"
      "trainer.logger.name=${run_name}"
      "+trainer.logger.log_model=false"
    )
  elif [[ "${LOGGER}" == "none" ]]; then
    LOGGER_ARGS=("trainer.logger=false")
  fi

  if [[ "${TRAINER_PROFILER:-}" == "advanced" ]]; then
    LOGGER_ARGS+=("+trainer.profiler=advanced")
  elif [[ "${TRAINER_PROFILER:-}" == "simple" ]]; then
    LOGGER_ARGS+=("+trainer.profiler=simple")
  elif [[ -n "${TRAINER_PROFILER:-}" && "${TRAINER_PROFILER:-}" != "none" ]]; then
    echo "Unknown TRAINER_PROFILER=${TRAINER_PROFILER}; expected advanced, simple, none, or unset." >&2
    exit 2
  fi
}

slurm_session_key() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "${SLURM_JOB_ID}"
  else
    return 1
  fi
}

pretrain_last_ckpt() {
  local key index_path run_dir
  if ! key="$(slurm_session_key)"; then
    echo "Cannot infer checkpoint: SLURM_JOB_ID is not set." >&2
    return 1
  fi
  index_path="${SPT_CACHE_DIR}/.slurm_index/${key}"
  if [[ ! -f "${index_path}" ]]; then
    echo "Missing SLURM run index: ${index_path}" >&2
    return 1
  fi
  run_dir="$(<"${index_path}")"
  echo "${run_dir}/checkpoints/last.ckpt"
}

manifest_value() {
  local key="$1"
  local manifest="$2"
  awk -F': ' -v key="${key}" '$1 == key {print $2; exit}' "${manifest}"
}

export_backbone_checkpoint() {
  local src="$1"
  local dst="$2"

  python - "${src}" "${dst}" <<'PY'
import os
import sys
import tempfile

import torch

src, dst = sys.argv[1], sys.argv[2]
checkpoint = torch.load(src, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict")
if state_dict is None:
    raise SystemExit(f"Checkpoint has no state_dict: {src}")

prefix = "model.backbone."
backbone = {
    key[len(prefix):]: value
    for key, value in state_dict.items()
    if key.startswith(prefix)
}
if not backbone:
    raise SystemExit(f"No {prefix!r} keys found in checkpoint: {src}")

os.makedirs(os.path.dirname(dst), exist_ok=True)
fd, tmp = tempfile.mkstemp(
    prefix=".backbone.", suffix=".pt.tmp", dir=os.path.dirname(dst)
)
os.close(fd)
try:
    torch.save(backbone, tmp)
    os.replace(tmp, dst)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)

print(f"Exported backbone checkpoint: {dst} ({len(backbone)} tensors)", file=sys.stderr)
PY
}

write_pretrain_manifest() {
  local ckpt="$1"
  local run_dir checkpoint_dir run_name
  run_dir="$(dirname "$(dirname "${ckpt}")")"
  checkpoint_dir="$(dirname "${ckpt}")"
  run_name="${RUN_NAME:-${RUN_GROUP}-seed${SEED}-pretrain}"

  cat > "${RUN_MANIFEST}" <<EOF
run_group: ${RUN_GROUP}
method: ${METHOD}
run_name: ${run_name}
seed: ${SEED}
output_dir: ${OUTPUT_DIR}
output_root: ${OUTPUT_ROOT}
spt_cache_dir: ${SPT_CACHE_DIR}
spt_run_dir: ${run_dir}
checkpoint_dir: ${checkpoint_dir}
last_ckpt: ${ckpt}
backbone_ckpt: ${BACKBONE_CKPT}
pretrain_config: ${CONFIG_ARCHIVE_DIR}/pretrain.yaml
probe_config: ${CONFIG_ARCHIVE_DIR}/linear_probe.yaml
pretrain_hparams: ${CONFIG_ARCHIVE_DIR}/pretrain_hparams.yaml
EOF
}

run_pretrain() {
  if [[ ! -f "${PRETRAIN_CONFIG}" ]]; then
    echo "Missing pretrain config: ${PRETRAIN_CONFIG}" >&2
    echo "ROOT_DIR=${ROOT_DIR}" >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    exit 1
  fi
  cp "${PRETRAIN_CONFIG}" "${CONFIG_ARCHIVE_DIR}/pretrain.yaml"
  build_logger_args pretrain
  spt run "${PRETRAIN_CONFIG}" "${METHOD_ARGS[@]}" "${LOGGER_ARGS[@]}"
  local ckpt
  ckpt="$(pretrain_last_ckpt)"
  if [[ -f "$(dirname "$(dirname "${ckpt}")")/hparams.yaml" ]]; then
    cp "$(dirname "$(dirname "${ckpt}")")/hparams.yaml" "${CONFIG_ARCHIVE_DIR}/pretrain_hparams.yaml"
  fi
  export_backbone_checkpoint "${ckpt}" "${BACKBONE_CKPT}"
  write_pretrain_manifest "${ckpt}"
  echo "Pretrain manifest: ${RUN_MANIFEST}"
  echo "Last checkpoint: ${ckpt}"
  echo "Backbone checkpoint: ${BACKBONE_CKPT}"
}

prepare_probe_backbone() {
  local source_ckpt manifest_backbone
  if [[ -n "${PROBE_TRAIN_SPLIT:-}" ]]; then
    export TRAIN_SPLIT="${PROBE_TRAIN_SPLIT}"
  fi
  if [[ -n "${PROBE_VAL_SPLIT:-}" ]]; then
    export VAL_SPLIT="${PROBE_VAL_SPLIT}"
  fi
  if [[ ! -f "${BACKBONE_CKPT}" ]]; then
    if [[ -f "${RUN_MANIFEST}" ]]; then
      manifest_backbone="$(manifest_value backbone_ckpt "${RUN_MANIFEST}")"
      if [[ -n "${manifest_backbone}" ]]; then
        export BACKBONE_CKPT="${manifest_backbone}"
      fi
    fi
  fi
  if [[ ! -f "${BACKBONE_CKPT}" ]]; then
    source_ckpt="${CKPT_PATH:-}"
    if [[ -z "${source_ckpt}" && -f "${RUN_MANIFEST}" ]]; then
      source_ckpt="$(manifest_value last_ckpt "${RUN_MANIFEST}")"
    fi
    if [[ -z "${source_ckpt}" ]]; then
      echo "Missing BACKBONE_CKPT for probe stage." >&2
      echo "Set BACKBONE_CKPT=/absolute/path/to/backbone.pt, or provide CKPT_PATH/RUN_MANIFEST so it can be exported." >&2
      exit 1
    fi
    if [[ "${source_ckpt}" != /* ]]; then
      echo "CKPT_PATH must be an absolute path; got: ${source_ckpt}" >&2
      exit 1
    fi
    if [[ ! -f "${source_ckpt}" ]]; then
      echo "Checkpoint does not exist: ${source_ckpt}" >&2
      exit 1
    fi
    export_backbone_checkpoint "${source_ckpt}" "${BACKBONE_CKPT}"
  fi
  if [[ "${BACKBONE_CKPT}" != /* ]]; then
    echo "BACKBONE_CKPT must be an absolute path; got: ${BACKBONE_CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${BACKBONE_CKPT}" ]]; then
    echo "Backbone checkpoint does not exist: ${BACKBONE_CKPT}" >&2
    exit 1
  fi
  if [[ -z "$(manifest_value backbone_ckpt "${RUN_MANIFEST}" 2>/dev/null || true)" && -f "${RUN_MANIFEST}" ]]; then
    printf 'backbone_ckpt: %s\n' "${BACKBONE_CKPT}" >> "${RUN_MANIFEST}"
  fi
}

run_probe() {
  prepare_probe_backbone
  if [[ ! -f "${PROBE_CONFIG}" ]]; then
    echo "Missing probe config: ${PROBE_CONFIG}" >&2
    echo "ROOT_DIR=${ROOT_DIR}" >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    exit 1
  fi
  cp "${PROBE_CONFIG}" "${CONFIG_ARCHIVE_DIR}/linear_probe.yaml"
  build_logger_args probe
  spt run "${PROBE_CONFIG}" "${LOGGER_ARGS[@]}"
}

run_dino_probe() {
  prepare_probe_backbone
  if [[ ! -f "${DINO_PROBE_CONFIG}" ]]; then
    echo "Missing DINO-style probe config: ${DINO_PROBE_CONFIG}" >&2
    echo "ROOT_DIR=${ROOT_DIR}" >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
    exit 1
  fi
  cp "${DINO_PROBE_CONFIG}" "${CONFIG_ARCHIVE_DIR}/linear_probe_dino_style.yaml"
  build_logger_args probe_dino
  spt run "${DINO_PROBE_CONFIG}" "${LOGGER_ARGS[@]}"
}

echo "Running ${METHOD} ${STAGE} on ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Pretrain config: ${PRETRAIN_CONFIG}"
echo "Probe config: ${PROBE_CONFIG}"
echo "DINO-style probe config: ${DINO_PROBE_CONFIG}"

case "${STAGE}" in
  pretrain)
    run_pretrain
    ;;
  probe|probe_all)
    run_probe
    run_dino_probe
    ;;
  probe_simple)
    run_probe
    ;;
  probe_dino)
    run_dino_probe
    ;;
  both|both_all)
    run_pretrain
    run_probe
    run_dino_probe
    ;;
  both_simple)
    run_pretrain
    run_probe
    ;;
  both_dino)
    run_pretrain
    run_dino_probe
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected pretrain, probe, probe_simple, probe_dino, probe_all, both, both_simple, both_dino, or both_all." >&2
    exit 2
    ;;
esac

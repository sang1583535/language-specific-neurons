#!/bin/bash

MODEL_TYPE_TO_USE="olmo2"

LANGUAGE_CODE_TO_IDENTIFY="en id km lo ms my ta th tl vi zh"

DATA_ACTIVATION_PATHS=(
    # /scratch/e1583535/language-specific-neurons/data/OLMo-2-1124-7B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-2B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-4B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-6B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-8B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-10B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-20B
    # /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-30B
    /scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-34B
)

DATA_LOG_PATHS=(
    # "OLMo-2-1124-7B"
    # "parallel-only-7B-34B/ckpt-2B"
    # "parallel-only-7B-34B/ckpt-4B"
    # "parallel-only-7B-34B/ckpt-6B"
    # "parallel-only-7B-34B/ckpt-8B"
    # "parallel-only-7B-34B/ckpt-10B"
    # "parallel-only-7B-34B/ckpt-20B"
    # "parallel-only-7B-34B/ckpt-30B"
    "parallel-only-7B-34B/ckpt-34B"
)

MASK_FILES_TO_USE=(
    # "/scratch/e1583535/language-specific-neurons/activation_mask/olmo2_7B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt2B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt4B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt6B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt8B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt10B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt10B_activation_mask"
    # "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt30B_activation_mask"
    "/scratch/e1583535/language-specific-neurons/activation_mask/parallel_only_7B_34B_ckpt34B_activation_mask"
)

for INDEX in "${!DATA_ACTIVATION_PATHS[@]}"; do
    DATA_ACTIVATION_PATH="${DATA_ACTIVATION_PATHS[$INDEX]}"
    DATA_LOG_PATH="${DATA_LOG_PATHS[$INDEX]}"
    MASK_FILE_TO_USE="${MASK_FILES_TO_USE[$INDEX]}"

    qsub -v LANGUAGE_CODE_TO_IDENTIFY="$LANGUAGE_CODE_TO_IDENTIFY",DATA_ACTIVATION_PATH="$DATA_ACTIVATION_PATH",DATA_LOG_PATH="$DATA_LOG_PATH",MODEL_TYPE_TO_USE="$MODEL_TYPE_TO_USE",MASK_FILE_TO_USE="$MASK_FILE_TO_USE" mass_identify.pbs
done
#!/bin/bash

LANGUAGES=(
  "en" 
  "id" 
  "km"
  "lo"
  "ms"
  "my"
  "ta"
  "th"
  "tl"
  "vi" 
  "zh"
)

DATA_PREFIX_PATH="/scratch/e1583535/language-specific-neurons/data/text_tokenized/c4-wiki-olmo2"

MODELS=(
  # "allenai/OLMo-2-1124-7B"

  "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-2B_limited-checkpoints/step3-unsharded-hf"
  "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-2B_limited-checkpoints/step6-unsharded-hf"
  "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-2B_limited-checkpoints/step9-unsharded-hf"
  "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-2B_limited-checkpoints/step12-unsharded-hf"

  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step477-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step954-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step1431-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step1908-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step2385-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step4770-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step7155-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step8290-unsharded-hf"

  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step477-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step954-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step1431-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step1908-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step2385-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step2862-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step3339-unsharded-hf"
  # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step3816-unsharded-hf"
  
)

OUTPUT_PREFIX_PATHS=(
  # "/scratch/e1583535/language-specific-neurons/data/OLMo-2-1124-7B"

  "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-12M"
  "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-25M"
  "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-37M"
  "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-50M"

  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-2B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-4B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-6B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-8B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-10B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-20B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-30B"
  # "/scratch/e1583535/language-specific-neurons/data/parallel-only-7B-34B/ckpt-34B"

  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-2B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-4B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-6B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-8B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-10B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-12B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-14B"
  # "/scratch/e1583535/language-specific-neurons/data/multilingual-7B-34B/ckpt-16B"
)

for i in "${!MODELS[@]}"; do
  MODEL=${MODELS[$i]}
  OUTPUT_PREFIX=${OUTPUT_PREFIX_PATHS[$i]}
  echo "Using model: $MODEL"
  echo "Output prefix: $OUTPUT_PREFIX"

  for LANG in "${LANGUAGES[@]}"; do
    echo "  Firing language: $LANG"

    qsub -v LANGUAGE=$LANG,DATA_PREFIX=$DATA_PREFIX_PATH,MODEL=$MODEL,OUTPUT_PREFIX=$OUTPUT_PREFIX mass_activation.pbs

    sleep 2

    echo "-- ++ -- ++ -- ++ -- ++ --"
  done
done

# for i in "${!LANGUAGES[@]}"; do
#   LANG=${LANGUAGES[$i]}
#   echo "Firing language: $LANG"

#   for j in "${!MODELS[@]}"; do
#     MODEL=${MODELS[$j]}
#     OUTPUT_PREFIX=${OUTPUT_PREFIX_PATHS[$j]}
#     echo "  Using model: $MODEL"
#     echo "  Output prefix: $OUTPUT_PREFIX"

#     qsub -v LANGUAGE=$LANG,DATA_PREFIX=$DATA_PREFIX_PATH,MODEL=$MODEL,OUTPUT_PREFIX=$OUTPUT_PREFIX mass_activation.pbs

#     echo "-- ++ -- ++ -- ++ -- ++ --"
#   done
# done
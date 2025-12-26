#!/bin/bash

LANGUAGES=(
    "en"
    "km"
    "lo"
    "my"
    "ms"
    "ta"
    "th"
    "tl"
)

DATA_FILES=(
    "20231101.en"
    "20231101.km"
    "20231101.lo"
    "20231101.my"
    "20231101.ms"
    "20231101.ta"
    "20231101.th"
    "20231101.tl"
)

TOKENIZER="allenai/OLMo-2-1124-7B"

MODEL_TYPE="olmo2"

OUTPUT_DIR="/scratch/e1583535/language-specific-neurons/data"

for i in "${!LANGUAGES[@]}"; do
    LANGUAGE=${LANGUAGES[$i]}
    DATA_FILE=${DATA_FILES[$i]}
    
    qsub -v LANGUAGE="$LANGUAGE",DATA_FILE="$DATA_FILE",TOKENIZER="$TOKENIZER",MODEL_TYPE="$MODEL_TYPE",OUTPUT_DIR="$OUTPUT_DIR" mass_build_wiki_id.pbs

    sleep 1
done
echo "Submitted all jobs."
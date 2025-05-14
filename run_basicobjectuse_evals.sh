#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "daphne.3object_use_colors"
    "daphne.3object_use_no_colors"
    "dd_object_use_easy2"
    "daphne.2object_use_colors_pretrained"
    "daveey.dist.2x4"
)


for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.sim \
        sim=object_use \
        run=object_use$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/object_use \


python3 -m tools.analyze +eval_db_uri=wandb://artifacts/object_use run=object_use_db ++analyze.output_path=s3://softmax-public/policydash/objectuse.html \

done

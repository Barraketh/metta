#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "daveey.dist.2x4"
    "navigation_training:v35"
    "b.daphne.navigation0"
    "daphne_navigation_train"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "b.daphne.navigation4"
    "gd2_sharing24_03"
    "gd2_sharing24_06"
    "gd2_sharing_48"
    "gd2_sharing_24"
    "gd2_sharing48_03"
    "gd2_sharing48_06"
    "MRM_test_mettabox"
    "georged_48_no_sharing"
    "georged_24_no_sharing"
    "dd_object_use_easy2"
    "daphne.3object_use_no_colors"
    "daphne.3object_use_colors"
    "daphne.2object_use_colors_pretrained"
 )

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/navigation_db

    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/multiagent_db

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db

    python3 -m tools.sim \
        sim=objectuse \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/objectuse_db
done

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/navigation_db run=navigation_db ++analyzer.output_path=s3://softmax-public/policydash/navigation.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/multiagent_db run=multiagent_db ++analyzer.output_path=s3://softmax-public/policydash/multiagent.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/memory_db run=memory_db ++analyzer.output_path=s3://softmax-public/policydash/memory.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/objectuse_db run=objectuse_db ++analyzer.output_path=s3://softmax-public/policydash/objectuse.html

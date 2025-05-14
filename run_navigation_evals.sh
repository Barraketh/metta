#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "daveey.dist.2x4"
    # "b.daphne.navigation0"
    # "b.daphne.navigation4"
    # "b.daphne.navigation1"
    # "b.daphne.navigation3"
    # "navigation_training"
    # "daphne_navigation_train"
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
        +eval_db_uri=wandb://artifacts/navigation_db_before_refactor \
        ++device=cpu \

done
#!/bin/bash

# Define the list of policy URIs

"""
MISSING

georged_sharing48_06 - check num_agents
georged_sharing24_06 - unexpected keyword teams in terain_from_numpy
georged_sharing_24_range_pretrained - unexpected keyword teams in terain_from_numpy
georged_sharing24_06_pretrained - unexpected keyword teams in terain_from_numpy
"""

POLICIES=(
    "daveey.dist.2x4"
    "gd2_sharing_48"
    "gd2_sharing48_06"
    "gd2_sharing48_03"
    "gd2_sharing24_03"
    "gd2_sharing_24"
    "gd2_sharing24_06"
    "b.daphne.navigation0"
    "b.daphne.navigation4"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "navigation_training"
    "daphne_navigation_train"
)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"

    DB=wandb://artifacts/object_use_db


    python3 -m tools.sim \
        sim=object_use \
        run=object_use_db$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=$DB \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/navigation_db_before_refactor run=navigation_db_before_refactor ++analyzer.output_path=s3://softmax-public/policydash/navigation_db_before_refactor.html

done

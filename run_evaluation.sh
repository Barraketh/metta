#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "breadcrumb_daphne_multiagent_training"
    "b.daphne.multiagent_training_pretrained"
    "b.daphne.multiagent_training2"
    "onlyred_daphne_multiagent_training"
    "b.daphne.multiagent_training_8agents_onlyred"
    "b.daphne.multiagent_training_4agents_onlyred"
    "b.daphne.multiagent_training_4agents"
    "4agents_daphne_multiagent_training"
    "8agents_daphne_multiagent_training"
    "b.daphne.multiagent_training"
    "b.daphne.multiagent_training_8agents"
    "b.daphne.objectuse_training_breadcrumb"
    "b.daphne.objectuse_training_breadcrumb_sm"
    "b.daphne.objectuse_training_breadcrumb_pretrained_sm"
    "b.daphne.objectuse_training_breadcrumb_pretrained"
    "b.daphne.multiagent_training_breadcrumb"
    "b.daphne.multiagent_training_breadcrumb_pretrained"
    "b.daphne.multiagent_training_breadcrumb_sm"
    "b.daphne.multiagent_training_breadcrumb_pretrained_sm"
    "b.daphne.32_multiagent_training"
    "b.daphne.32_multiagent_training_pretrained"
    "32agents_daphne_multiagent_training"
)
#!/bin/bash

    # "daphne_objectuse_allobjs_multienv:v94"
    # "b.daphne.object_use_multienv2:v65"
    # "training_regular_envset_nb:v76"
    # "daphne_objectuse_bigandsmall:v67"
    # "navigation_training:v35"
    # "training_regular_envset"
    # "training_prioritized_envset"
    # "b.daphne.navigation_prioritized_envset"
    # "b.daphne.navigation_regular_envset"
    # "b.daphne.objectuse_prioritized_envset"
    # "b.daphne.objectuse_regular_envset"
    # "b.daphne.prioritized_envset"
    # "b.daphne.uniform_envset_nb"
    # "b.daphne.regular_envset_nb"
    # "b.daphne.prioritized_envset_nb"
    # "b.daphne.regular_envset"
    # "training_regular_envset_nb"
    # "training_uniform_envset_nb"



for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    # python3 -m tools.sim \
    #     sim=navigation \
    #     run=navigation$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/navigation_db \
    #     # device=cpu \

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/memory_db \
        # device=cpu \

    # python3 -m tools.sim \
    #     sim=object_use \
    #     run=objectuse$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/objectuse_db \
    #     # device=cpu \

    # python3 -m tools.sim \
    #     sim=nav_sequence \
    #     run=nav_sequence$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/nav_sequence_db \
    #     # device=cpu \

    # python3 -m tools.sim \
    #     sim=nav_sequence \
    #     run=nav_sequence$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/nav_sequence_db \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db run=memory_db ++dashboard.output_path=s3://softmax-public/policydash/memory.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objectuse_db run=objectuse_db ++dashboard.output_path=s3://softmax-public/policydash/objectuse.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/nav_sequence_db run=nav_sequence_db ++dashboard.output_path=s3://softmax-public/policydash/nav_sequence.html

done

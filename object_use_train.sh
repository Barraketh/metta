# python -m devops.aws.batch.launch_task --cmd=train --run=georged_extended_sequence --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/extended_sequence --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=georged_extended_sequence_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/extended_sequence --skip-validation +trainer.initial_policy.uri=wandb://run/b.daphne.navigation1 \

python -m devops.aws.batch.launch_task --cmd=train --run=daphne.3object_use_colors --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=daphne.3object_use_no_colors --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/multienv_nc --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=daphne.2object_use_colors_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/multienv --skip-validation +trainer.initial_policy.uri=wandb://run/b.daphne.navigation1 \

# python -m devops.aws.batch.launch_task --cmd=train --run=daphne.2object_use_no_colors_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/multienv_nc --skip-validation +trainer.initial_policy.uri=wandb://run/b.daphne.navigation1 \

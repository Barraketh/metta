python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.navigation4 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/navigation/training/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.navigation5 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/navigation/training/multienv --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.navigation2 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/navigation/training/multienv --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.navigation3 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/navigation/training/multienv --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_extended_sequence --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_sharing03/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \


# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing24_03_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_sharing03/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing48_03_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_sharing03/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing24_06 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_sharing06/multienv --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing_24_range_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_range/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing24_06_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_sharing06/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing48_06_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_sharing06/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_24_nosharing_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/24/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation  \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_48_nosharing_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/48/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=gd_sharing_48_range_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_range/multienv +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_24_no_sharing --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/24/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing24_03 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_sharing03/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing_24 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/24_range/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_48_no_sharing --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/48/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing48_03 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_sharing03/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing48_06 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_sharing06/multienv --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=gd2_sharing_48 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/reward_sharing/48_range/multienv --skip-validation \


python -m devops.aws.batch.launch_task --cmd=train --run=daphne_navigation_train --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/navigation/training/multienv --skip-validation





# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_nc_pretrained2 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv --skip-validation  +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 \

#python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_c_pretrained2 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv_colors --skip-validation +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 \



# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_nc3 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_c3 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv_colors --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_mix3 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv_all --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.multiagent_mix_pretrained2 --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/multiagent/training/objects/multienv_all --skip-validation +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3 \

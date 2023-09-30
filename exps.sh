# example of training the model for CIFAR100 BIR, branch develop
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval=per_context

# example of training the model for MINI BIR, branch develop
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval=per_context --z-dim=300

# example of training the model for CIFAR100 BIR+SI, branch develop
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context

# example of training the model for MINI BIR+SI, branch develop
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300

# example of training the model for CIFAR100, our method, branch laten_match_dist_cycle
# python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=2 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --cycles=10 --model-type=resnet --model-type=resnet --eval_per_context

# example of training the model for MINI, our method, branch laten_match_dist_cycle
# python main.py --experiment=MINI --scenario=class --brain-inspired --seed=2 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --cycles=10 --model-type=resnet --z-dim=300 --model-type=resnet --eval_per_context

# example of training the model for CIFAR100, our method, branch laten_match_dist_cycle
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --cycles=10 --model-type=resnet --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --cycles=10 --model-type=resnet --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --cycles=5 --model-type=resnet --model-type=resnet --eval_per_context


# example of training the model for MINI, our method, branch laten_match_dist_cycle
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --cycles=10 --model-type=resnet --z-dim=300 --model-type=resnet --eval_per_context
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --cycles=10 --model-type=resnet --z-dim=300 --model-type=resnet --eval_per_context
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --cycles=5 --model-type=resnet --z-dim=300 --model-type=resnet --eval_per_context

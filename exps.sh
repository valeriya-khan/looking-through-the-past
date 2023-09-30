# example of training the model for CIFAR100 BIR, branch develop
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval=per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval=per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval=per_context

# example of training the model for MINI BIR, branch develop
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval=per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval=per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --brain-inspired --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval=per_context --z-dim=300

# example of training the model for CIFAR100 BIR+SI, branch develop
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context

# example of training the model for MINI BIR+SI, branch develop
python main.py --experiment=MINI --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --brain-inspired --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --reg-strength=100000000 --dg-prop=0.6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300

# example of running other methods from comparison table for CIFAR100, branch develop
python main.py --experiment=CIFAR50 --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context

python main.py --experiment=CIFAR50 --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context

python main.py --experiment=CIFAR50 --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context
python main.py --experiment=CIFAR50 --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context

# example of running other methods from comparison table for MINI, branch develop
python main.py --experiment=MINI --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=6 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300

python main.py --experiment=MINI --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300 
python main.py --experiment=MINI --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=11 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300

python main.py --experiment=MINI --scenario=class --si --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --ewc --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --lwf --seed=1 --pre-convE --freeze-convE --seed-to-ltag  --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
python main.py --experiment=MINI --scenario=class --seed=1 --pre-convE --freeze-convE --seed-to-ltag --convE-ltag=e50s --time --contexts=26 --active-classes=all-so-far --model-type=resnet --eval_per_context --z-dim=300
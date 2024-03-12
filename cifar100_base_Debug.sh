#!/bin/bash
python main.py -model icarl -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" --debug
python main.py -model lwf -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" --debug
python main.py -model finetune -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" --debug
echo "All commands have finished running"
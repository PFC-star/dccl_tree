#!/bin/bash


python main.py -model icarlwdua  -init 6 -incre 1 --dataset "cifar10" -p benchmark -d 0  1

python main.py -model lwfdua  -init 6 -incre 1  --dataset "cifar10" -p benchmark -d 0 1





wait      # 等待所有后台任务结束
echo "All commands have finished running."
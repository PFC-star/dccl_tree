#!/bin/bash

python main.py -model icarl  -init 6 -incre 1  -p benchmark -d 0  1  -dt True &
python main.py -model icarl  -init 6 -incre 1  -p benchmark -d 0  1   -dt True -so dcl &
python main.py -model icarlwdua  -init 6 -incre 1  -p benchmark -d 0 1    -dt True -so dcl   &






wait      # 等待所有后台任务结束
echo "All commands have finished running."
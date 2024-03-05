#!/bin/bash
python main.py -model icarl -init 6 -incre 1  -p benchmark -d 0 1 -dt True --debug &
python main.py -model icarl -init 6 -incre 1  -p benchmark -d 0 1 --debug &
python main.py -model icarl -init 6 -incre 1  -p benchmark -d 0 1 -dt True -so dcl --debug &

python main.py -model lwf -init 6 -incre 1  -p benchmark -d 0 1 -dt True --debug &
python main.py -model lwf -init 6 -incre 1  -p benchmark -d 0 1 --debug &
python main.py -model lwf -init 6 -incre 1  -p benchmark -d 0 1 -dt True -so dcl --debug &



wait      # 等待所有后台任务结束
echo "All commands have finished running."
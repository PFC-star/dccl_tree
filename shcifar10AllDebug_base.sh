#!/bin/bash
sh finetune_cifar10_debug.sh &
sh icarl_cifar10_debug.sh &
sh lwf_cifar10_debug.sh &

echo "All commands have finished running"
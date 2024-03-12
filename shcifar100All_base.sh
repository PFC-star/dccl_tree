#!/bin/bash
sh finetune_cifar100.sh &
sh icarl_cifar100.sh &
sh lwf_cifar100.sh &

echo "All commands have finished running"
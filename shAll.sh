#!/bin/bash
sh finetune_cifar10.sh &
sh icarl_cifar10.sh &
sh icarlwdua_cifar10.sh &
sh lwf_cifar10.sh &
sh lwfwdua_cifar10.sh &
sh joint_cifar10.sh &
echo "All commands have finished running."
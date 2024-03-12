#!/bin/bash

sh joint_cifar100.sh &
sh icarlwdua_cifar100.sh &
sh lwfwdua_cifar100.sh &
echo "All commands have finished running"
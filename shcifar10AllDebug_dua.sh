#!/bin/bash

sh joint_cifar10_debug.sh &
sh icarlwdua_cifar10_debug.sh &
sh lwfwdua_cifar10_debug.sh &
echo "All commands have finished running"
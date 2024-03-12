#!/bin/bash

sh joint_cifar10.sh &
sh icarlwdua_cifar10.sh &
sh lwfwdua_cifar10.sh &
echo "All commands have finished running"
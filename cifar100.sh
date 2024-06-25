 python main.py -model lwf -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" -tk 5  &
python main.py -model icarl -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" -tk 5  &

python main.py -model icarlwdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" -tk 5  &


python main.py -model finetune -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100"  -tk 5 &
python main.py -model joint -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100"   &
python main.py -model lwf -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100"   &


python main.py -model lwfwdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100" -tk 5   &
python main.py -model finetune -init 6 -incre 1  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar10"


python main.py -p benchmark -d 0 1   -model  icarl  --dataset imagenet200 -net resnet32 -d 0 1 -init 120 -incre 20 -so dccl -dt True -tk 1

-p benchmark -d 0 1  --debug -model  finetune  --dataset imagenet200 -net resnet32 -d 0 1 -init 120 -incre 20 -so dccl -dt True -tk 5
-p benchmark -d 0 1  --debug -model  finetune  --dataset cifar100 -net resnet32 -d 0 1 -init 60 -incre 10 -so dccl -dt True -tk 5
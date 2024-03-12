python main.py -model icarlwdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True --dataset "cifar100"

python main.py -model icarlwdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'resnet32' -dt True -so dcl --dataset "cifar100"


python main.py -model icarlwdua -init 60 -incre 10 -net 'resnet32'  --dataset "cifar100" -p benchmark -d 0 1
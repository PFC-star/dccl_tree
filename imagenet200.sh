python main.py -model icarl -init 120 -incre 20  -p benchmark -d 0 1 -net 'resnet32'  -so dccl -dt True --dataset "imagenet200" -tk 1   &



python main.py -model finetune -init 120 -incre 20  -p benchmark -d 0 1 -net 'resnet32'  -so dccl -dt True --dataset "imagenet200" -tk 1   &

python main.py -model icarlwdua -init 120 -incre 20  -p benchmark -d 0 1 -net 'resnet32'  -so dccl -dt True --dataset "imagenet200" -tk 1   &
 python main.py -model lwf -init 120 -incre 20  -p benchmark -d 0 1 -net 'resnet32'  -so dccl -dt True --dataset "imagenet200" -tk 1   &

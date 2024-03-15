 python main.py -model icarl -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &

python main.py -model icarlwdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &

python main.py -model finetune -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &
python main.py -model joint -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &
python main.py -model lwf -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &


python main.py -model lwfdua -init 60 -incre 10  -p benchmark -d 0 1 -net 'cosine_resnet34'   --dataset "domainNet" --debug &




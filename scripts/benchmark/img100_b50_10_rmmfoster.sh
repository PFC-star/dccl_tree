CUDA_VISIBLE_DEVICES=3 python main_rmm.py \
    -model rmm-foster \
    --dataset imagenet100 \
    -net resnet18 \
    -init 50 \
    -incre 10 \
    -p benchmark \
    -d 0\
    -m 0.3 0.3 0.3 0.4 0.4 0.4 \
    -c 0.0 0.0 0.1 0.1 0.1 0.0\
    --skip
import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    if param["debug"]==True:
        param['epochs']= 1
        param['init_epoch']= 1
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--memory_size','-ms',type=int, default=2000)
    parser.add_argument('--init_cls', '-init', type=int, default=6)
    parser.add_argument('--increment', '-incre', type=int, default=1)
    parser.add_argument('--model_name','-model', type=str, default=None, required=True)
    parser.add_argument('--convnet_type','-net', type=str, default='resnet32') # cosine_resnet32 resnet32
    parser.add_argument('--prefix','-p',type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--postfix', '-post', type=str, help='exp type', default='z')
    parser.add_argument('--device','-d', nargs='+', type=int, default=[0,1])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true",)
    parser.add_argument('--domainTrans','-dt',type=bool, default=False)
    parser.add_argument('--scenario', '-so',  type=str, default='dccl')
    parser.add_argument('--topk', '-tk', type=int, default=1)
    return parser


if __name__ == '__main__':
    main()

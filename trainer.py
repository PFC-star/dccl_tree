import copy
import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model
import pandas as pd

time_list = []
model_size=[]
topk = 'top2'
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args['time_str'] = time_str
    
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["convnet_type"],
        args["seed"],
        init_cls,
        args["increment"],
    )
    args['exp_name'] = exp_name

    if args['debug']:
        logfilename = "logs/debug/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )
    else:
        logfilename = "logs/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )

    args['logfilename'] = logfilename

    csv_name = "{}_{}_{}_B{}_Inc{}".format( 
        args["dataset"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"],
    )
    args['csv_name'] = csv_name
    os.makedirs(logfilename, exist_ok=True)

    log_path = os.path.join(args["logfilename"], "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Time Str >>> {args['time_str']}")
    # save config
    config_filepath = os.path.join(args["logfilename"], 'configs.json')
    with open(config_filepath, "w") as fd:
            json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)
    setattr(model.__class__, 'bn_stats', {})
    cnn_curve, nme_curve, no_nme = {"top1": [], topk: []}, {"top1": [], topk: []}, True

    cnn_acc_list =[]

    for task in range(data_manager.nb_tasks):
        start_time = time.time()
        logging.info(f"Start time:{start_time}")
        subResult = []
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        # 这里每次传的数据集都是一样的，可以修改为不一样的

        model.incremental_train(data_manager)
        model.save_bn_stats_in_model(model._network,task)
        model.save_bn_stats_to_file(model._network,args["dataset"],args['model_name'])
        if task == data_manager.nb_tasks-1:
            cnn_accy_dict, nme_accy_dict = model.eval_task(data_manager=data_manager,save_conf=True)
            # cnn_accy_dict = True if nme_accy_dict is None else False
            no_nme = True if nme_accy_dict is None else False
        else:
            cnn_accy_dict, nme_accy_dict = model.eval_task(data_manager,save_conf=False)
        cnn_acc_list.append(cnn_accy_dict)
        print("task {}:".format(task))
        print(cnn_accy_dict)
        model.after_task(data_manager)
        cnn_accy = cnn_accy_dict['dataset ID {}:'.format(task)]
        nme_accy = nme_accy_dict['dataset ID {}:'.format(task)]
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[topk].append(cnn_accy[topk])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve[topk].append(nme_accy[topk])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve[topk]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve[topk]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[topk].append(cnn_accy[topk])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve[topk]))
        end_time = time.time()
        logging.info(f"End Time:{end_time}")
        cost_time = end_time - start_time
        print("save model {}".format(task))
        model_size.append(save_model(args, model))
        time_list.append(cost_time)
        if args["model_name"] =="joint":
            break
    # save_time(args, cost_time)
    # save_results(args, cnn_curve, nme_curve, no_nme)

    # if args["debug"]==False:
    # save_all_results(args=args, cnn_curve=cnn_curve, nme_curve=nme_curve, no_nme=no_nme,cost_time=cost_time,cnn_acc=cnn_accy)
    save_allll_results(args=args,cnn_acc_list=cnn_acc_list,cost_time=cost_time,cnn_curve=cnn_curve, nme_curve=nme_curve, no_nme=no_nme)
    if args['model_name'] not in ["podnet", "coil"]:
        save_fc(args, model)
    else:
        model_size.append(save_model(args, model))

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")

def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve[topk]
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve[topk]
    
    #-------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]},")
            f.write(f"{sum(cnn_top1)/len(cnn_top1)} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    #-------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")


    #-------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")       

        #-------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")


def save_all_results(args, cnn_curve, nme_curve,cost_time ,cnn_acc,no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve[topk]
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve[topk]

    # -------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]},")
            f.write(f"{sum(cnn_top1) / len(cnn_top1)},")
            for key, value in args.items():
                f.write(f"{value}, ")
            f.write(f"{cost_time}, ")
            f.write(f"{args['domainTrans']} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    # -------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")

    # -------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")

                # -------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")


def save_all_results_excel(args, cnn_curve, nme_curve, cost_time, cnn_acc, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve[topk]
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve[topk]

    # -------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]},")
            f.write(f"{sum(cnn_top1) / len(cnn_top1)},")
            for key, value in args.items():
                f.write(f"{value}, ")
            f.write(f"{cost_time}, ")
            f.write(f"{args['domainTrans']} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")


def save_allll_results(args,cnn_acc_list,cost_time,cnn_curve, nme_curve, no_nme):
    # 解析cnn_acc_list,以最终的格式来排列
    data = []
    for taskDict in cnn_acc_list:
        task_result = []
        # cnn_acc_dict 为一个task对应的数据
        for datasetID, datasetResult in taskDict.items():
            #  v 为dataset ID 0对应的字典
            acc_list = []
            for acc in datasetResult['grouped'].items():
                k, v = acc
                if k == 'new' or k == 'old':
                    continue

                acc_list.append(v)

            print(len(acc_list))
            for i in range(20):
                if (len(acc_list) <= 10):
                    acc_list.append(None)
                else:
                    break
            task_result.extend(acc_list)


        data.append(task_result)
    total_acc = []
    total_forget = []
    for i,model_acc in enumerate(data):
        temp_acc_lst = []
        #  计算平均准确率 i=0  对应  0-5   i=1 对应 1-6  以此类推
        for j in range(i+1):

            temp_acc_lst.append(model_acc[j*11])

        total_acc.append(np.average(temp_acc_lst))
    for acc in total_acc:
        total_forget.append(max(total_acc)-acc)
    # 调用插入数组函数
    for i,model_acc in enumerate(data):
        model_acc.insert(0, total_acc[i])
        model_acc.insert(0,total_forget[i])
        model_acc.insert(2, model_size[i])
        model_acc.insert(3, time_list[i])

    argsKeyList = []
    argsValueList = []
    for key, value in args.items():
         argsKeyList.extend(["{}".format(key)])
         argsValueList.extend(["{}".format(value)])

    data.append(argsKeyList)
    data.append(argsValueList)
    df = pd.DataFrame(data)
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1",f"{args['dataset']}","debug")
    os.makedirs(_log_dir, exist_ok=True)
    if args['domainTrans']:
        sheet_name = args['model_name']+" "+args['convnet_type']+" " + 'dccl' + "debug"
        if args['scenario']=='dcl':
            sheet_name = args['model_name'] +" "+args['convnet_type']+" " +'dcl' + "debug"
    else:
        sheet_name = args['model_name'] +" "+args['convnet_type']+" " +  'ccl' + "debug"
    _log_path = os.path.join(_log_dir, f"{sheet_name}.xlsx")
    writer = pd.ExcelWriter(_log_path, engine='xlsxwriter')

    df.to_excel(writer, index=False, sheet_name=sheet_name)
    writer.close()
    print("sheet_name", sheet_name)




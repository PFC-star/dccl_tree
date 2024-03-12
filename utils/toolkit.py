import os
import numpy as np
import torch
import  json

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10,cur_task=0):
    if increment==10:
        dataset_name='CIFAR100'
    if increment==1:
        dataset_name='CIFAR10'

    assert len(y_pred) == len(y_true), "Data length error."
    dclflag = False
    if dataset_name=='CIFAR100':
        if np.max(y_true)== 59 and cur_task!=0:
            dclflag=True
            taskID = np.max(y_true) + 1 - 60
    if dataset_name == 'CIFAR10':
        if np.max(y_true)== 5 and cur_task!=0:
            dclflag=True
            taskID = np.max(y_true) + 1 - 6
    all_acc = {}
    # Grouped accuracy

    all_acc_temp={}
    for class_id in range(0, np.max(y_true)+1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc_temp[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    tempTatalacc=[]
    if dclflag:
        cur_task = 0
    # print("cur_task: ",  cur_task)

    if dataset_name == 'CIFAR10':

        if np.max(y_true)+1 == 10 and np.min(y_true)==0:
            for class_id in range(0, 10, increment):
                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
                )
                # print(label)
                tempTatalacc.append(all_acc_temp[label])
            all_acc["total"] = np.average(tempTatalacc)
        else:
            for class_id in range(cur_task, cur_task+6, increment):
                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
                )
                # print(label)
                tempTatalacc.append(all_acc_temp[label])
            all_acc["total"] = np.average(tempTatalacc)
    if dataset_name == 'CIFAR100':
        if np.max(y_true) + 1 == 100 and np.min(y_true) == 0:
            for class_id in range(0, 100, increment):
                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
                )
                # print(label)
                tempTatalacc.append(all_acc_temp[label])
            all_acc["total"] = np.average(tempTatalacc)
        else:
            for class_id in range(cur_task*10, cur_task*10+60, increment):
                label = "{}-{}".format(
                    str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
                )
                # print(label)
                tempTatalacc.append(all_acc_temp[label])
            all_acc["total"] = np.average(tempTatalacc)

    for class_id in range(0, np.max(y_true)+1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    print("Saving last model ")
    # _path = os.path.join(args['logfilename'], "model_params_best.pt")

    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1", f"{args['dataset']}",f"{args['postfix']}")
    os.makedirs(_log_dir, exist_ok=True)
    if args['domainTrans']:
        sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'dccl' + 'last '  + ".pt"
        if args['scenario'] == 'dcl':
            sheet_name =  args['model_name'] + " " + args['convnet_type'] + " " + 'dcl'+ 'last '  + ".pt"
    else:
        sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'ccl' + 'last '  + ".pt"
    model_path = os.path.join(_log_dir, sheet_name)
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight.state_dict(), model_path)
    param_size = 0
    buffer_size = 0
    for param in model._network.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model._network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb
def save_model_ing(args, model,task):
    #used in PODNet
    print("Saving best model {}..... ".format(task+1))
    # _path = os.path.join(args['logfilename'], "model_params_best.pt")

    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1", f"{args['dataset']}",f"{args['postfix']}")
    os.makedirs(_log_dir, exist_ok=True)
    if args['domainTrans']:
        sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'dccl' + 'best ' +str(task )+ ".pt"
        if args['scenario'] == 'dcl':
            sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'dcl' + 'best '+str(task )+ ".pt"
    else:
        sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'ccl' + 'best '+str(task )+ ".pt"
    model_path = os.path.join(_log_dir, sheet_name)

    if len(args['device']) > 1:
        weight = model
    else:
        weight = model.cpu()
    torch.save(weight.state_dict(), model_path)
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb

def loadBestModel(args, task):

    if task ==-1 or task == 0 :
        if args['dataset'] == "cifar10":
            model_path = os.path.join("logs/benchmark/cifar10/finetune/0308-13-10-39-411_cifar10_resnet32_2024_B6_Inc1",
                                 "model_params.pt")
        if args['dataset'] == "cifar100":
            model_path = os.path.join("logs/benchmark/cifar100/finetune/0309-18-46-53-848_cifar100_resnet32_2024_B60_Inc10",
                                 "model_params.pt")
        return model_path
    # _path = os.path.join(args['logfilename'], "model_params_best.pt")

    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1", f"{args['dataset']}",f"{args['postfix']}")
    # os.makedirs(_log_dir, exist_ok=True)
    if args['domainTrans']:
        sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'dccl' + 'best ' +str(task )+ ".pt"
        if args['scenario'] == 'dcl':
            sheet_name = args['model_name'] + " " + args['convnet_type'] + " " + 'dcl' + 'best ' +str(task )+ ".pt"
    else:
        sheet_name =  args['model_name'] + " " + args['convnet_type'] + " " + 'ccl' + 'best ' +str(task )+ ".pt"
    model_path = os.path.join(_log_dir, sheet_name )
    return model_path

import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os

import pandas as pd

from os.path import exists, join, realpath, split
import os
class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        # if args['topk']:
        #     self.topk = args['topk']
        # else:
        self.topk=2
        self.increment=args['increment']
        self.domainTrans=args['domainTrans']
        if self.domainTrans:
            self.domain = [
                            'None',
                            'RandomHorizontalFlip',
                            'ColorJitter',
                            'RandomRotation',
                            'RandomAffine' ,
                          ]
            # self.domain = [
            #
            #     'RandomHorizontalFlip',
            #     'ColorJitter',
            #     'RandomRotation',
            #     'RandomGrayscale',
            #     'RandomVerticalFlip',
            # ]
        else:
            self.domain=['None','None','None','None','None','None']
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, test_acc):
        assert self.args['model_name'] == 'finetune'
        checkpoint_name = f"checkpoints/finetune_{self.args['csv_name']}"
        _checkpoint_cpu = copy.deepcopy(self._network)
        if isinstance(_checkpoint_cpu, nn.DataParallel):
            _checkpoint_cpu = _checkpoint_cpu.module
        _checkpoint_cpu.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "convnet": _checkpoint_cpu.convnet.state_dict(),
            "fc":_checkpoint_cpu.fc.state_dict(),
            "test_acc": test_acc
        }
        torch.save(save_dict, "{}_{}.pkl".format(checkpoint_name, self._cur_task))
    
    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true,cur_task):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes,increment=self.increment,cur_task=cur_task)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret
    def loadBNall(self, net, task,cur_task):
        """
            Saves the running estimates of all batch norm layers for a given
            task, in the net.bn_stats attribute.
        """
        bn_stats = torch.load('checkpoints/cifar10/derwdua/BN_stats.pt')
        state_dict = net.state_dict()
        updateBnStats = {}


        # modify bn_stats with convnetsName such as convents.0 or convents.1
        # 'convnets.0.bn_1.running_var'
        for i in range(cur_task + 1):
            for name in bn_stats[task]:
                convnetsName = 'convnets.{}'.format( i)
                updateBnStats[convnetsName  + name] = bn_stats[task][name]
        net.load_state_dict(updateBnStats, strict=False)

    def loadBN(self, net,task, cur_task):
        """
            Saves the running estimates of all batch norm layers for a given
            task, in the net.bn_stats attribute.
        """
        path_name ='checkpoints/'+ self.args['dataset']+'/'+ self.args['model_name']+'/'+'BN_stats.pt'
        bn_stats = torch.load(path_name)
        state_dict = net.state_dict()
        updateBnStats = {}

        # modify bn_stats with convnetsName such as convents.0 or convents.1
        # 'convnets.0.bn_1.running_var'
        for i in range(task + 1):
            # convnetsName = 'convnets.{}'.format(i)
            convnetsName='convnets.'
            for name in bn_stats[cur_task]:
                # updateBnStats[convnetsName+name[10:]+'.running_mean'] = bn_stats[cur_task][name]['running_mean']
                # updateBnStats[convnetsName+name[10:] + '.running_var'] = bn_stats[cur_task][name]['running_var']
                updateBnStats[  name +'.running_mean'] = bn_stats[cur_task][name]['running_mean']
                updateBnStats[  name  + '.running_var'] = bn_stats[cur_task][name]['running_var']

        net.load_state_dict(updateBnStats, strict=False)

    def save_bn_stats_in_model_all(self, net, task):
        """
            Saves the running estimates of all batch norm layers for a given
            task, in the net.bn_stats attribute.
        """
        state_dict = net.state_dict()
        net.bn_stats[task] = {}
        convnetsName = 'convnets.{}'.format(task)

        for name in state_dict:

            if name.startswith(convnetsName) and ('bn' and  "running_mean") in name or ('bn' and  "running_var") in name:
                net.bn_stats[task][name[10:]] =  state_dict[name].detach().clone()

        print()
        # for layer_name, m in net.named_modules():
        #     print("------layer_name------\n", layer_name)
        #     print("------m--------\n", m)
        #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        #         net.bn_stats[task] = {
        #             'running_mean': state_dict[layer_name + '.running_mean'].detach().clone(),
        #             'running_var': state_dict[layer_name + '.running_var'].detach().clone()
        #         }

    def save_bn_stats_in_model(self,net, task):
        """
            Saves the running estimates of all batch norm layers for a given
            task, in the net.bn_stats attribute.
        """
        state_dict = net.state_dict()
        net.bn_stats[task] = {}
        for layer_name, m in net.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                net.bn_stats[task][layer_name] = {
                    'running_mean': state_dict[layer_name + '.running_mean'].detach().clone(),
                    'running_var': state_dict[layer_name + '.running_var'].detach().clone()
                }
        print()
    def save_bn_stats_to_file(self, net, dataset_str=None, model_str=None, file_name=None):
        """
            Saves net.bn_stats content to a file.
        """
        # ckpt_folder = 'checkpoints/' + dataset_str + '/' + model_str + '/'
        ckpt_folder = join('checkpoints', dataset_str, model_str)
        os.makedirs(ckpt_folder, exist_ok=True)
        if not file_name:
            file_name = 'BN_stats.pt'
        torch.save(net.bn_stats, join(ckpt_folder, file_name))

    def eval_task(self,data_manager,save_conf=False):
        cnn_accy_dict={}
        nme_accy_dict={}
        for cur_task in range(self._cur_task+1):
            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes), source="test",
                mode="test",
                domain_type=self.domain[cur_task],
                domainTrans=self.domainTrans
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers']
            )



            if self.args['model_name'] == "derwwdua" or self.args['model_name'] == "icarlwwdua" or self.args['model_name'] == "lwfwwdua":
                # 要将所有的分支的域换成对应的BN域
                # self.loadBN(self._network,self._cur_task,cur_task)
                print("----------------DDDDDDUUUUUUUUAAAAAA---------------------------")
                mom_pre = 0.1

                train_dataset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    mode="train",
                    # appendent=self._get_memory(),
                    domainTrans=self.domainTrans,
                    domain_type=self.domain[cur_task]
                )
                self.train_loader = DataLoader(
                    train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers']
                )

                self._network.eval()
                for i, (_, inputs, targets) in enumerate(self.train_loader):
                    if i> 3:
                        break
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    mom_new = (mom_pre * 0.94)
                    for m in self._network.modules():
                        # print(m)
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.train()
                            m.momentum = mom_new + 0.005
                    mom_pre = mom_new

                    _ = self._network(inputs)


            # 'convnets.0.stage_1.0.bn_a.running_mean'
            # self._network.state_dict()['convnets.0.stage_1.0.bn_a.running_mean']
            y_pred, y_true = self._eval_cnn(self.test_loader)

            cnn_accy = self._evaluate(y_pred, y_true,cur_task)

            if hasattr(self, "_class_means"):
                y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                nme_accy = self._evaluate(y_pred, y_true,cur_task)
            else:
                nme_accy = None

            if save_conf:
                _pred = y_pred.T[0]
                _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
                _target_path = os.path.join(self.args['logfilename'], "target.npy")
                np.save(_pred_path, _pred)
                np.save(_target_path, y_true)

                _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
                os.makedirs(_save_dir, exist_ok=True)
                _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
                with open(_save_path, "a+") as f:
                    f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")
            cnn_accy_dict.setdefault('dataset ID {}:'.format(cur_task), cnn_accy)
            nme_accy_dict.setdefault('dataset ID {}:'.format(cur_task), nme_accy)
        return cnn_accy_dict, nme_accy_dict

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            # print("------targets---------")
            # print(targets)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            # print("------outputs---------")
            # print(outputs.shape )
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            # print("------predicts---------")
            # print(predicts.shape)
            # print(torch.topk(
            #     outputs, k=self.topk, dim=1, largest=True, sorted=True
            # ))
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt),
                domain_type=self.domain[self._cur_task],
                domainTrans=self.domainTrans

            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                domain_type=self.domain[self._cur_task],
                domainTrans=self.domainTrans
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) +self.args['EPSILON'] )).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                
                if len(vectors) == 0:
                    break
            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            # exemplar_targets = np.full(m, class_idx)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),

            domain_type = self.domain[self._cur_task],
            domainTrans = self.domainTrans
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets),
                domain_type=self.domain[self._cur_task],
                domainTrans=self.domainTrans
            )
            class_loader = DataLoader(
                class_dset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                domain_type=self.domain[self._cur_task],
                domainTrans=self.domainTrans
            )
            class_loader = DataLoader(
                class_dset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
                domain_type=self.domain[self._cur_task],
                domainTrans=self.domainTrans
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + +self.args['EPSILON'])).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
    def compute_task_acc(self,data_manager,total_acc_max,task):
        cnn_acc_list_temp = []
        args  = self.args
        cnn_accy_dict_temp, nme_accy_dict_temp = self.eval_task(data_manager=data_manager, save_conf=True)
        cnn_acc_list_temp.append(cnn_accy_dict_temp)
            # 解析cnn_acc_list,以最终的格式来排列
        data = []
        if args['dataset'] != 'domainNet':
            for taskDict in cnn_acc_list_temp:
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
            for i, model_acc in enumerate(data):
                temp_acc_lst = []
                #  计算平均准确率 i=0  对应  0-5   i=1 对应 1-6  以此类推
                for j in range(i + 1):
                    temp_acc_lst.append(model_acc[j * 11])

                total_acc.append(np.average(temp_acc_lst))
        else:
            for taskDict in cnn_acc_list_temp:
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
                        if (len(acc_list) <= 13):
                            acc_list.append(None)
                        else:
                            break
                    task_result.extend(acc_list)

                data.append(task_result)
            total_acc = []
            total_forget = []
            for i, model_acc in enumerate(data):
                temp_acc_lst = []
                #  计算平均准确率 i=0  对应  0-5   i=1 对应 1-6  以此类推
                for j in range(i + 1):
                    temp_acc_lst.append(model_acc[j * 14])

                total_acc.append(np.average(temp_acc_lst))
        for acc in total_acc:
            total_forget.append(max(total_acc) - acc)
        # 调用插入数组函数
        if total_acc_max <  total_acc[-1]:
            for i, model_acc in enumerate(data):
                model_acc.insert(0, total_acc[i])
                print("best_acc : ".format(i), total_acc[i])
                model_acc.insert(0, total_forget[i])

            argsKeyList = []
            argsValueList = []
            for key, value in self.args.items():
                argsKeyList.extend(["{}".format(key)])
                argsValueList.extend(["{}".format(value)])

            data.append(argsKeyList)
            data.append(argsValueList)
            df = pd.DataFrame(data)
            _log_dir = os.path.join("./results/", f"{self.args['prefix']}", "cnn_top1", f"{self.args['dataset']}", f"{self.args['postfix']}")
            os.makedirs(_log_dir, exist_ok=True)
            if self.args['domainTrans']:
                sheet_name = self.args['model_name'] + " " + self.args['convnet_type'][:5] + " " + 'dccl' + 'b'+str(task )
                if self.args['scenario'] == 'dcl':
                    sheet_name = self.args['model_name'] + " " + self.args['convnet_type'][:5] + " " + 'dcl'+ 'b'+str(task )
            else:
                sheet_name = self.args['model_name'] + " " + self.args['convnet_type'][:5] + " " + 'ccl'+ 'b'+str(task )
            _log_path = os.path.join(_log_dir, f"{sheet_name}.xlsx")
            writer = pd.ExcelWriter(_log_path, engine='xlsxwriter')

            df.to_excel(writer, index=False, sheet_name=sheet_name)
            writer.close()
            print("sheet_name", sheet_name)
        return total_acc[-1]
    def compute_task_acc_joint(self,test_loader,total_acc_max,task):
        cnn_acc_list_temp = []
        cnn_accy_dict_temp, nme_accy_dict_temp = self.eval_task_joint(test_loader, save_conf=True)
        cnn_acc_list_temp.append(cnn_accy_dict_temp)
            # 解析cnn_acc_list,以最终的格式来排列
        data = []
        for taskDict in cnn_acc_list_temp:
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
        for i, model_acc in enumerate(data):
            temp_acc_lst = []
            #  计算平均准确率 i=0  对应  0-5   i=1 对应 1-6  以此类推
            for j in range(task+1):
                temp_acc_lst.append(model_acc[j * 11])

            total_acc.append(np.average(temp_acc_lst))
        for acc in total_acc:
            total_forget.append(max(total_acc) - acc)
        # 调用插入数组函数
        if total_acc_max <  total_acc[-1]:
            for i, model_acc in enumerate(data):
                model_acc.insert(0, total_acc[i])
                print("best_acc : ".format(i), total_acc[i])
                model_acc.insert(0, total_forget[i])

            argsKeyList = []
            argsValueList = []
            for key, value in self.args.items():
                argsKeyList.extend(["{}".format(key)])
                argsValueList.extend(["{}".format(value)])

            data.append(argsKeyList)
            data.append(argsValueList)
            df = pd.DataFrame(data)
            # _log_dir = os.path.join("./results/", f"{self.args['prefix']}", "cnn_top1", f"{self.args['dataset']}", f"{self.args['postfix']}")
            # os.makedirs(_log_dir, exist_ok=True)
            # if self.args['domainTrans']:
            #     sheet_name = self.args['model_name'] + " " + self.args['convnet_type'] + " " + 'dccl' + 'best '+str(task )
            #     if self.args['scenario'] == 'dcl':
            #         sheet_name = self.args['model_name'] + " " + self.args['convnet_type'] + " " + 'dcl'+ 'best '+str(task )
            # else:
            #     sheet_name = self.args['model_name'] + " " + self.args['convnet_type'] + " " + 'ccl'+ 'best '+str(task )
            # _log_path = os.path.join(_log_dir, f"{sheet_name}.xlsx")
            # writer = pd.ExcelWriter(_log_path, engine='xlsxwriter')
            #
            # df.to_excel(writer, index=False, sheet_name=sheet_name)
            # writer.close()
            # print("sheet_name", sheet_name)
        return total_acc[-1]

    def eval_task_joint(self, test_loader, save_conf=False):
        cnn_accy_dict = {}
        nme_accy_dict = {}
        for cur_task in range(self._cur_task + 1):

            self.test_loader  = test_loader

            if self.args['model_name'] == "derwwdua" or self.args['model_name'] == "icarlwwdua" or self.args[
                'model_name'] == "lwfwwdua":
                # 要将所有的分支的域换成对应的BN域
                # self.loadBN(self._network,self._cur_task,cur_task)
                print("----------------DDDDDDUUUUUUUUAAAAAA---------------------------")
                mom_pre = 0.1

                train_dataset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    mode="train",
                    # appendent=self._get_memory(),
                    domainTrans=self.domainTrans,
                    domain_type=self.domain[cur_task]
                )
                self.train_loader = DataLoader(
                    train_dataset, batch_size=self.args['batch_size'], shuffle=True,
                    num_workers=self.args['num_workers']
                )

                self._network.eval()
                for i, (_, inputs, targets) in enumerate(self.train_loader):
                    if i > 3:
                        break
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    mom_new = (mom_pre * 0.94)
                    for m in self._network.modules():
                        # print(m)
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.train()
                            m.momentum = mom_new + 0.005
                    mom_pre = mom_new

                    _ = self._network(inputs)

            # 'convnets.0.stage_1.0.bn_a.running_mean'
            # self._network.state_dict()['convnets.0.stage_1.0.bn_a.running_mean']
            y_pred, y_true = self._eval_cnn(self.test_loader)

            cnn_accy = self._evaluate(y_pred, y_true, cur_task)

            if hasattr(self, "_class_means"):
                y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                nme_accy = self._evaluate(y_pred, y_true, cur_task)
            else:
                nme_accy = None

            if save_conf:
                _pred = y_pred.T[0]
                _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
                _target_path = os.path.join(self.args['logfilename'], "target.npy")
                np.save(_pred_path, _pred)
                np.save(_target_path, y_true)

                _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
                os.makedirs(_save_dir, exist_ok=True)
                _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
                with open(_save_path, "a+") as f:
                    f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")
            cnn_accy_dict.setdefault('dataset ID {}:'.format(cur_task), cnn_accy)
            nme_accy_dict.setdefault('dataset ID {}:'.format(cur_task), nme_accy)
        return cnn_accy_dict, nme_accy_dict
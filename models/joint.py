import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torch.utils.data import ConcatDataset


import os
import copy
from utils.toolkit import save_model_ing,loadBestModel
class Joint(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args["convnet_type"], False)
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones = args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.total_acc_max = -1
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']

    def after_task(self,data_manager,task):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args['scenario'] == 'dcl':
            if self.args['dataset'] == 'cifar10':
                self._total_classes = 6
                self._known_classes = 0
            if self.args['dataset'] == 'cifar100':
                self._total_classes = 60
                self._known_classes = 0
            if self.args['dataset'] == 'domainNet':
                self._total_classes = 60
                self._known_classes = 0
        else:
            if self.args['dataset'] == 'cifar10':
                self._total_classes = 10
                self._known_classes = 0

            if self.args['dataset'] == 'cifar100':
                self._total_classes = 100
                self._known_classes = 0
            if self.args['dataset'] == 'domainNet':
                self._total_classes = 110
                self._known_classes = 0
         
        self.total_acc_max = -1
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        logging.info(
            "domain:{} ".format(self.domain[self._cur_task])
        )
        train_dataset_lst = globals()
        self.train_loader_lst = globals()
        test_dataset_lst = globals()
        self.test_loader_lst = globals()

        if self.args['dataset'] == 'domainNet':
            for i in range(6):
                print(i)
                data_manager._setup_data('domainNet', False, 2024, i)

                test_dataset_lst['test_dataset_{}'.format(i)] = data_manager.get_dataset(
                    np.arange(0,  self._total_classes), source="test", mode="test",
                    domainTrans=self.args['domainTrans'],
                    domain_type=self.domain[i],
                )
                train_dataset_lst['train_dataset_{}'.format(i)] = data_manager.get_dataset(
                    np.arange(0,  self._total_classes), source="train", mode="train",
                    domainTrans=self.args['domainTrans'],
                    domain_type=self.domain[i],
                )
            concat_test_set = ConcatDataset(
            [test_dataset_0, test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5])
            concat_train_set = ConcatDataset(
                [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4,train_dataset_5])
        else:
            for i in range(5):
                if self.domainTrans:
                    domain_=i
                else:
                    domain_=0
                train_dataset_lst['train_dataset_{}'.format(i)] = data_manager.get_dataset(
                # 能够看到所有的数据，然后重新按照所有数据学习一遍，以前的相当于预训练了
                np.arange(0, self._total_classes),
                source="train",
                mode="train",
                domainTrans=self.domainTrans,
                domain_type=self.domain[domain_],)



                test_dataset_lst['test_dataset_{}'.format(i)] = data_manager.get_dataset(
                    np.arange(0, self._total_classes), source="test", mode="test",
                    domainTrans=self.domainTrans,
                    domain_type=self.domain[domain_],
                )

            concat_train_set = ConcatDataset([train_dataset_0, train_dataset_1, train_dataset_2,train_dataset_3,train_dataset_4])
            concat_test_set = ConcatDataset( [test_dataset_0, test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4])
        self._contact_train_loader = DataLoader(
            concat_train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )

        self._contact_test_loader = DataLoader(
            concat_test_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 交替各个数据集 D0，D1 D2 D3 D4 D5，那如何续上呢？

        self._train(self._contact_train_loader,self._contact_test_loader,data_manager)



        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader,data_manager):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        print("\ndomain_type:",self.domain[self._cur_task])
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay
            )
            if self.args['skip'] :
                if len(self._multiple_gpus) > 1:
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._init_train(train_loader, test_loader, optimizer, data_manager=data_manager,scheduler=None)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.lrate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
            )
            self._init_train(train_loader, test_loader, optimizer,  data_manager=data_manager,scheduler=None)

    def _init_train(self, train_loader, test_loader, optimizer,  data_manager,scheduler=None):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)

            total_acc = self.compute_task_acc_joint(test_loader, self.total_acc_max,0)
            if total_acc >= self.total_acc_max:
                self.best_model = copy.deepcopy(self._network)
                save_model_ing(args=self.args, model=self.best_model.module,task = 1)

                self.total_acc_max = total_acc

                print("total_acc_max:", self.total_acc_max)
                print("total_acc:", total_acc)
                print("test_acc:", test_acc)


            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['epochs'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print("test_acc:", test_acc)

            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer,  scheduler=None):

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            
            total_acc = self.compute_task_acc(data_manager,self.total_acc_max)
            if total_acc >= self.total_acc_max:
                self.best_model = copy.deepcopy(self._network)
                save_model_ing(args=self.args, model=self.best_model)

                self.total_acc_max =  total_acc
                print("total_acc_max:", self.total_acc_max)
                print("total_acc:", total_acc)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['epochs'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print("test_acc:", test_acc)

            prog_bar.set_description(info)
        logging.info(info)



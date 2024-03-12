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

import os
import copy
from utils.toolkit import save_model_ing,loadBestModel





# -p benchmark -d 0 1  --debug -model  finetune -dt True --dataset domainNet -net resnet34_imagenet -d 0 1 -init 200 -incre 25




class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args["convnet_type"], False)
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones = args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']

        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']


    def after_task(self,data_manager,task):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args['scenario'] == 'dcl':
            if self.args['dataset']== 'cifar10':
                self._total_classes = 6
                self._known_classes = 0
            if self.args['dataset']== 'cifar100':
                self._total_classes = 60
                self._known_classes = 0
            if self.args['dataset']== 'domainNet':
                self._total_classes = 200
                self._known_classes = 0
        else:
            if self._cur_task != 0:
                if self.args['dataset'] == 'cifar10':
                    self._known_classes = self._known_classes - 5
                if self.args['dataset'] == 'cifar100':
                    self._known_classes = self._known_classes - 50
                if self.args['dataset'] == 'domainNet':
                    self._known_classes = self._known_classes - 175



        if  self._cur_task==0:
            pass
        else:
             # 加载上一个任务最优的模型
            print("-----Load Model  {}------".format(self._cur_task))
            model_path = loadBestModel(self.args, self._cur_task - 1)
            self._network.load_state_dict(torch.load(model_path) )

        self.total_acc_max = -1
        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        logging.info(
            "domain:{} ".format(self.domain[self._cur_task])
        )
        if self.args['dataset'] == 'domainNet':
            data_manager._setup_data('domainNet', False, 2024,self._cur_task)



        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            domainTrans=self.domainTrans,
            domain_type=self.domain[self._cur_task],
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
            domainTrans=self.domainTrans,
            domain_type=self.domain[self._cur_task],
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers']
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers']
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader,data_manager=data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader,data_manager):
        self._network.to(self._device)
        print("\ndomain_type:",self.domain[self._cur_task])
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args['init_lr'],
                weight_decay=self.args['init_weight_decay'],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['init_milestones'], gamma=self.args['init_lr_decay']
                )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=self.init_epoch,
            # ) #check
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
                self._init_train(train_loader, test_loader, optimizer,data_manager=data_manager, scheduler=scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args['lrate'],
                momentum=0.9,
                weight_decay=self.args['weight_decay'],
            ) # 1e-5
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     optimizer=optimizer, milestones=self.args['milestones'], gamma=self.args['lrate_decay']
            # )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=self.args['init_epoch'],
            # )  # check
            self._update_representation(train_loader, test_loader, optimizer,data_manager, scheduler=None)

    def _init_train(self, train_loader, test_loader, optimizer,data_manager, scheduler):
        prog_bar = tqdm(range(self.args['init_epoch']))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)




            test_acc = self._compute_accuracy(self._network, test_loader)
            # total_acc  = self.compute_task_acc(data_manager)
            # if total_acc >= self.total_acc_max:
            #     self.best_model = copy.deepcopy(self._network)
            #     save_model_ing(args=self.args, model=self.best_model)
            #
            # self.total_acc_max = np.max(total_acc,self.total_acc_max)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['init_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print("test_acc:",test_acc)
            # self.last_model = copy.deepcopy(self._network)
            # save_model_ing(args=self.args, model=self.last_model)
            # print("total_acc_max:", self.total_acc_max)
            # print("total_acc:", total_acc)
            prog_bar.set_description(info)

        logging.info(info)

        # # save checkpoint for fair comparison & save runing time
        # test_acc = self._compute_accuracy(self._network, test_loader)
        # self.save_checkpoint(test_acc)
        # logging.info("Save checkpoint successfully!")
    def _init_train_1(self, train_loader, test_loader, optimizer,data_manager, scheduler=None):
        # _path = os.path.join("model_params_finetune_100.pt")
        if self.args['dataset'] == "cifar10":
            _path = os.path.join("logs/benchmark/cifar10/finetune/0308-13-10-39-411_cifar10_resnet32_2024_B6_Inc1",
                                 "model_params.pt")
        if self.args['dataset'] == "cifar100":
            # _path = os.path.join("logs/benchmark/cifar100/finetune/0309-18-46-53-848_cifar100_resnet32_2024_B60_Inc10",
            #                      "model_params.pt")
            _path = os.path.join("results/benchmark/cnn_top1/cifar100/last80",
                                      "cifar100_50.pt")
        self._network.module.load_state_dict(torch.load(_path) )
        print("-----Load Model  {}------".format( self._cur_task))
    def _update_representation(self, train_loader, test_loader, optimizer, data_manager,scheduler=None ):
     


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
                    logits[:, self._known_classes :], fake_targets
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
            # 保存每个任务的最佳模型

            total_acc = self.compute_task_acc(data_manager,self.total_acc_max,task = self._cur_task)
            if total_acc >= self.total_acc_max:
                self.best_model = copy.deepcopy(self._network)
                save_model_ing(args=self.args, model=self.best_model.module,task = self._cur_task)

                self.total_acc_max =  total_acc
                print("task:{} total_acc_max:".format(self._cur_task), self.total_acc_max)
                print("task:{} total_acc:".format(self._cur_task), total_acc)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['init_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            print("test_acc:", test_acc)

            prog_bar.set_description(info)

        logging.info(info)


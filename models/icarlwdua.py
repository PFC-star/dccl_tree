import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import  os



class iCaRLWDUA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args["convnet_type"], False)
        self.args = args
    def after_task(self,data_manager):

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args['scenario'] == 'dcl':
            self._total_classes = 6
            self._known_classes = 0
        else:
            if self._cur_task != 0:
                self._known_classes = self._known_classes - 5
        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        logging.info(
            "domain:{} ".format(self.domain[self._cur_task])
        )
        # 这里正式建立数据集
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),  # 这个地方会加入重放数据集
            domainTrans=self.domainTrans,
            domain_type=self.domain[self._cur_task],
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers']
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
            domainTrans=self.domainTrans,
            domain_type=self.domain[self._cur_task],
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers']
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        print("\ndomain_type:", self.domain[self._cur_task])
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
            #     T_max=self.args['init_epoch'],
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
                self._init_train(train_loader, test_loader, optimizer, scheduler=None)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args['lrate'],
                momentum=0.9,
                weight_decay=self.args['weight_decay'],
            )# 1e-5
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     optimizer=optimizer, milestones=self.args['milestones'], gamma=self.args['lrate_decay']
            # )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args['init_epoch'],
            ) #check
            self._update_representation(train_loader, test_loader, optimizer, scheduler=None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler=None):
        prog_bar = tqdm(range(self.args['init_epoch']))
        _path = os.path.join("model_params_finetune_100.pt")
        # torch.load(_path, self._network.state_dict())
        self._network.module.load_state_dict(torch.load(_path))
        # for _, epoch in enumerate(prog_bar):
        #     self._network.train()
        #     losses = 0.0
        #     correct, total = 0, 0
        #     for i, (_, inputs, targets) in enumerate(train_loader):
        #         inputs, targets = inputs.to(self._device), targets.to(self._device)
        #         logits = self._network(inputs)["logits"]
        #
        #         loss = F.cross_entropy(logits, targets)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         losses += loss.item()
        #
        #         _, preds = torch.max(logits, dim=1)
        #         correct += preds.eq(targets.expand_as(preds)).cpu().sum()
        #         total += len(targets)
        #
        #     # scheduler.step()
        #     train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        #
        #     if epoch % 5 == 0:
        #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
        #             self._cur_task,
        #             epoch + 1,
        #             self.args['init_epoch'],
        #             losses / len(train_loader),
        #             train_acc,
        #
        #         )
        #     else:
        #         test_acc = self._compute_accuracy(self._network, test_loader)
        #         info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
        #             self._cur_task,
        #             epoch + 1,
        #             self.args['init_epoch'],
        #             losses / len(train_loader),
        #             train_acc,
        #             test_acc,
        #         )
        #     prog_bar.set_description(info)
        #
        # logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler=None):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            # add
            mom_pre = 0.1
            # add
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # # add
                # mom_new = (mom_pre * 0.94)
                # for m in self._network.modules():
                #     # print(m)
                #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                #         m.train()
                #         m.momentum = mom_new + 0.005
                # mom_pre = mom_new
                # # add
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)


                if self.args['scenario'] == 'dcl':
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes + 6],
                        self._old_network(inputs)["logits"],
                        self.args["T"],
                    )
                else:
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes + 5],
                        self._old_network(inputs)["logits"],
                        self.args["T"],
                    )

                loss =  loss_kd + loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
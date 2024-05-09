import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from pytorch_loss import FocalLossV3

from models.discriminator import FCDiscriminator
from .Net import Net
from .losses import cross_entropy2d
from .utils import get_scheduler


class Model():
    def __init__(self, stage, iters=90000, logger=None, lr=0.0001, resume_path=None, class_num=6, classifier='fcn',
                 bn='bn'):
        self.nets_DP = []
        self.nets = []
        self.logger = logger
        self.default_gpu = 0
        self.class_num = class_num
        self.resume_path = resume_path
        self.best_iou = -100
        self.stage = stage
        self.iters = iters
        self.lr = lr
        self.adv_source_label = 0
        self.adv_target_label = 1
        self.adv_weight = 1
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        if self.stage == 'warm_up':
            self.BaseNet = Net(classifier, backbone='resnet50', num_classes=self.class_num, freeze_bn=False,
                               restore_from=self.resume_path,
                               norm_layer=BatchNorm)
        else:
            self.BaseNet = Net(classifier, backbone='resnet50', num_classes=self.class_num, freeze_bn=False,
                               restore_from=self.resume_path,
                               norm_layer=BatchNorm, bn_clr=True)
        self.nets.extend([self.BaseNet])

        self.optimizers = []
        self.schedulers = []
        optimizer_cls = torch.optim.SGD
        optimizer_params = {'lr': self.lr, 'weight_decay': 2e-4, 'momentum': 0.9}

        self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, train_iters=self.iters)
        self.schedulers.extend([self.BaseSchedule])

        self.bceloss = torch.nn.MSELoss()
        self.focal_loss = FocalLossV3(alpha=0.25, gamma=2, reduction='mean')
        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)

        if self.stage == 'warm_up':
            self.net_D = FCDiscriminator(inplanes=self.class_num)
            self.net_D_DP = self.init_device(self.net_D, gpu_id=self.default_gpu, whether_DP=True)
            self.nets.extend([self.net_D])
            self.nets_DP.append(self.net_D_DP)

            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
            self.optimizers.extend([self.optimizer_D])
            self.DSchedule = get_scheduler(self.optimizer_D, train_iters=self.iters)
            self.schedulers.extend([self.DSchedule])

        if self.stage == 'stage2':
            self.teacher = Net(classifier, backbone='resnet50', num_classes=self.class_num, freeze_bn=False,
                               restore_from=self.resume_path,
                               norm_layer=BatchNorm)
            self.teacher.eval()
            self.teacher_DP = self.init_device(self.teacher, gpu_id=self.default_gpu, whether_DP=True)

        self.bceloss = torch.nn.MSELoss()

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets_DP.append(self.BaseNet_DP)

    def oracle(self, x, label):
        target_output = self.BaseNet_DP(x)
        loss_oracle = cross_entropy2d(input=target_output['out'], target=label, reduction='mean')
        loss_oracle.backward()
        self.BaseOpti.step()
        return loss_oracle.item()

    def step_warm_up(self, source_x, source_label, target_x, target_label):
        for param in self.net_D.parameters():
            param.requires_grad = False  # 优化分割网络
        self.BaseOpti.zero_grad()

        source_output = self.BaseNet_DP(source_x)

        loss_source = cross_entropy2d(input=source_output['out'], target=source_label, reduction='mean')

        target_output = self.BaseNet_DP(target_x)
        # loss_target = cross_entropy2d(input=target_output['out'], target=target_label, reduction='mean')
        target_D_out = self.net_D_DP(F.softmax(target_output['out'], dim=1))
        loss_adv = self.bceloss(target_D_out,
                                torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_source_label).to(
                                    target_D_out.device)) * self.adv_weight
        loss_G = loss_adv + loss_source\
                 # + loss_target
        loss_G.backward()
        self.BaseOpti.step()

        for param in self.net_D.parameters():
            param.requires_grad = True  # 优化判别器
        self.optimizer_D.zero_grad()
        source_D_out = self.net_D_DP(F.softmax(source_output['out'].detach(), dim=1))
        target_D_out = self.net_D_DP(F.softmax(target_output['out'].detach(), dim=1))
        loss_D = self.bceloss(source_D_out, torch.FloatTensor(source_D_out.data.size()).fill_(self.adv_source_label).to(
            source_D_out.device)) + \
                 self.bceloss(target_D_out, torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_target_label).to(
                     target_D_out.device))
        loss_D.backward()
        self.optimizer_D.step()

        return loss_source.item(), loss_adv.item(), loss_D.item()\
            # , loss_target.item()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            # net = DataParallelWithCallback(net, device_ids=[0])
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net

    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net is None:
            for net in self.nets:
                net.eval()
            if logger is not None:
                logger.info("Successfully set the model eval mode")
        else:
            net.eval()
            if logger is not None:
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net is None:
            for net in self.nets:
                net.train()
        else:
            net.train()
        return

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

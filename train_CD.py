import argparse
import os
import random
import time
# import nni
from argparse import ArgumentParser

import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn import SyncBatchNorm
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from Dataloader.dataset import *
from models.CD_Net import CD_Net
from models.utils import get_logger, get_scheduler
from utils.metrics import runningScore, averageMeter, RunningMetrics_CD
from Dataloader.CD_dataset import LEVID_CDset, SECONDset, WHU_CDset, SYSU_CDset, CDD_set
from models.utils import distribute
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import tqdm
# from nni.utils import merge_parameter
# from pynvml import *

from models.utils import EarlyStopping


# CUDA_VISIBLE_DEVICES= python -m torch.distributed.launch --nproc_per_node=4 train_CD.py
# ssh -L 12580:127.0.0.1:12580 liuyikun@211.87.232.115
def find_gpu():
    nvmlInit()
    mem = []
    nvidia_count = nvmlDeviceGetCount()
    for i in range(nvidia_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        mem.append(memory_info.free)

    index = np.where(np.array(mem) > 25000000000)[0]
    gpu_index = index[-1]
    return str(gpu_index)


def validation(model, logger, test_loader, running_metrics_val, iters, writer):
    iters = iters
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for idx, val_data, in enumerate(tqdm.tqdm(test_loader)):
            image_A = val_data['A']
            image_B = val_data['B']
            label = val_data['L']
            if distribution:
                image_A = image_A.cuda(args.local_rank, non_blocking=True)
                image_B = image_B.cuda(args.local_rank, non_blocking=True)
                label = label.cuda(args.local_rank, non_blocking=True)
            else:
                image_A = image_A.cuda()
                image_B = image_B.cuda()
                label = label.cuda()

            out_A, out_B, result = model.basenet(image_A, image_B)
            out = result.max(1)[1].detach().cpu().numpy()
            lbl = label.data.detach().cpu().numpy()
            running_metrics_val.update(lbl, out)

        score = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)
            logger.info('{}: {}'.format(k, v))

        running_metrics_val.reset()
        pred = vutils.make_grid(result, normalize=True, scale_each=True)
        label = vutils.make_grid(label, normalize=True, scale_each=True)
        writer.add_image('pred', pred, epoch)
        writer.add_image('lbl', label, epoch)
        writer.add_scalars('metrics', {'Kappa': score["Kappa"],
                                       'IoU': score['IoU'],
                                       'F1': score['F1'],
                                       'OA': score['OA'],
                                       'recall': score['recall'],
                                       'precision': score['precision']}, epoch)
        torch.cuda.empty_cache()
        if score['F1'] >= model.best:
            model.best = score['F1']
            state = {}
            new_state = {
                "model_state": model.state_dict(),
            }
            state[model.__class__.__name__] = new_state
            state['iter'] = iters + 1
            state['best'] = model.best
            save_path = os.path.join(logdir,
                                     "best_model_{}_{}.pkl".format(classifier, Note))
            torch.save(state, save_path)
    return score['F1']


Note = 'l2'
params = {'alpha': 1, 'beta': 1, 'theta': 1, 'dataset': 'LEVIR-CD', 'channel': 256, 'out_channel': 64, 'dilation': [True, True, True], 'MFDS': True, 'CFI': True, 'CMR': True, 'adv': True, 'cos': True}
# optimized_params = nni.get_next_parameter()
# params.update(optimized_params)
backbone = 'resnet50'
classifier = 'fcn'
# dataset = 'Potsdam'
dataset = params['dataset']
# dataset = 'SYSU-CD'
# dataset = 'WHU-CD'
# dataset = 'CDD'
# dataset = 'Building'
# dataset = 'SECOND'
inplanes = 3
num_class = 2
size = '256x256'
print_interval = 100
seed = 1337
batch_size = 8
start_epoch = 0
epochs = 200
lr = 0.01  # best lr {Potsdam: 0.01(CE loss)}, {Vaihingen: 0.01(CE loss), 0.05(Dual Focal loss)}
resume_path = None
# distribution = True
distribution = False
gpu_id = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
#
local_rank = None
if distribution:
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    print(torch.cuda.device_count())  # 打印gpu数量
    torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
    print('world_size', torch.distributed.get_world_size())  # 打印当前进程数
    local_rank = args.local_rank
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

logdir = 'logs/{}/{}'.format(dataset, Note)
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = get_logger(logdir, Note)
if distribution == False or dist.get_rank() == 0:
    writer = SummaryWriter(logdir)

logger.info(Note)
print(Note)
logger.info(params)
print(params)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print_info = "abstract:\n\t classifier: {}\n\t dataset: {}\n\t num_class: {}\n\t size: {}\n\t batch_size: {}\n\t lr: {}\n\t gpu_id: {}".format(
    classifier, dataset, num_class, size, batch_size, lr, gpu_id)
print(print_info)
logger.info(print_info)
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("----Loading Datasets----")
train_sampler = None
val_sampler = None
test_sampler = None
if dataset == 'LEVIR-CD':
    train_set = LEVID_CDset(mode='train')
    print("get {} images from LEVIR-CD train set".format(len(train_set)))
    test_set = LEVID_CDset(mode='test')
    print("get {} images from LEVIR-CD test set".format(len(test_set)))

elif dataset == 'WHU-CD':
    train_set = WHU_CDset(mode='train')
    print("get {} images from WHU_CD train set".format(len(train_set)))
    test_set = WHU_CDset(mode='test')
    print("get {} images from WHU_CD test set".format(len(test_set)))

elif dataset == 'CDD':
    train_set = CDD_set(mode='train')
    print("get {} images from CDD train set".format(len(train_set)))
    test_set = CDD_set(mode='test')
    print("get {} images from CDD test set".format(len(test_set)))

elif dataset == 'SYSU-CD':
    train_set = SYSU_CDset(mode='train')
    print("get {} images from SYSU-CD train set".format(len(train_set)))
    test_set = SYSU_CDset(mode='test')
    print("get {} images from SYSU-CD test set".format(len(test_set)))
if distribution:
    train_sampler = DistributedSampler(train_set)
    test_sampler = DistributedSampler(test_set)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=False, sampler=train_sampler)
test_loader = DataLoader(test_set, batch_size=batch_size * 6, num_workers=2, shuffle=False, sampler=test_sampler)
train_iters = len(train_loader) * epochs

# model initialization
print("----Model Initialization----")

model = CD_Net(restore_from=resume_path,
               lr=lr, distribution=distribution, local_rank=local_rank, iters=train_iters, params=params)

running_metrics_val = RunningMetrics_CD(num_classes=num_class)
time_meter = averageMeter()
print("Start Training!")
model.iter = 0
early_stopping = EarlyStopping(patience=60)
for epoch in range(start_epoch, epochs):
    for idx, data in enumerate(train_loader):
        image_A = data['A']
        # path_A = data['paths_A']
        image_B = data['B']
        lbl = data['L'].long()
        if distribution:
            image_A = image_A.cuda(args.local_rank, non_blocking=True)
            image_B = image_B.cuda(args.local_rank, non_blocking=True)
            lbl = lbl.cuda(args.local_rank, non_blocking=True)
        else:
            image_A = image_A.cuda()
            image_B = image_B.cuda()
            lbl = lbl.cuda()

        model.iter += 1
        i = model.iter
        model.train()
        start_ts = time.time()

        loss_dic = model.forward(image_A, image_B, lbl)

        # if distribution == False or dist.get_rank() == 0:
        #     writer.add_scalar('train_loss', loss_C + loss_cos + loss_adv + loss_D, model.iter)
        time_meter.update(time.time() - start_ts)
        res_time = (train_iters - i) * time_meter.avg
        m, s = divmod(res_time, 60)
        h, m = divmod(m, 60)
        # print log
        if (i + 1) % print_interval == 0 and (distribution == False or dist.get_rank() == 0):
            fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss:{}  Time/Image: {:.4f}  ETA: {:d}h: {:d}m: {:d}s"
            print_str = fmt_str.format(epoch + 1, epochs, i + 1, train_iters, loss_dic,
                                       time_meter.avg / batch_size, int(h), int(m), int(s))
            print(print_str)
            logger.info(print_str)
            time_meter.reset()
        model.scheduler_step()
    # evaluation
    if distribution == False or dist.get_rank() == 0:
        score = validation(model, logger, test_loader, running_metrics_val, iters=model.iter, writer=writer)
        logger.info('Best F1 until now is {}'.format(model.best))
        print("Best F1 until now is {}.".format(model.best))
    # early_stopping(score)
    torch.cuda.empty_cache()
    # nni.report_intermediate_result(score)
    if early_stopping.early_stop:
        print("Early stopping")
        break
# nni.report_final_result(model.best)

print(Note)
print(print_info)

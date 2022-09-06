# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: train.py
@time: 2021/11/25 10:11
"""
import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.hrt import HighResolutionTransformer
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
from tools.pytorch_utils import Save_Handle, AverageMeter
import tools.log_utils as log_utils

import torch.backends.cudnn as cudnn
from options import opt
import yaml

save_list = Save_Handle(max_num=1)

path = "./config/hrt_base.yaml"
config = yaml.load(open(path, 'r'), yaml.SafeLoader)['MODEL']['HRT']
# set the device for training

model_name = opt.model_name
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
edge_root = opt.edge_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

time_str = datetime.strftime(datetime.now(),"%m%d-%H%M%S")
logger = log_utils.get_logger(save_path + 'train-{}-{:s}.log'.format(model_name, time_str))
log_utils.print_config(vars(opt), logger)
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info('using gpu 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logger.info('using gpu 1')
cudnn.benchmark = True

model = HighResolutionTransformer(config, 1000)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

num_parms = 0
for p in model.parameters():
    num_parms += p.numel()
logger.info("Total Parameters (For Reference): {}".format(num_parms))

start_epoch = 0
if (opt.hr_load is not None):
    model.init_weights(opt.hr_load, opt.cnn_load)
    logger.info('loading pretrained model from ' + opt.hr_load)
    logger.info('loading pretrained model from ' + opt.cnn_load)
elif opt.resume:
    logger.info('loading pretrained model from last stop ' + opt.resume)
    suf = opt.resume.rsplit('.', 1)[-1]
    if suf == "tar":
        checkpoint = torch.load(opt.resume, torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info('model load successfully, optimizer load successfully!')

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

# load data
logger.info('load data...')
train_loader = get_loader(image_root, gt_root,depth_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root,test_depth_root, opt.trainsize)
total_step = len(train_loader)

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depth,edge) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.cuda()
            s = model(images,depth)

            loss = structure_loss(s, gts)
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:

                logger.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||Mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logger.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + '{}_epoch_{}.pth'.format(model_name,epoch))
        temp_save_path = save_path + "{}_ckpt.tar".format(epoch)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, temp_save_path)
        save_list.append(temp_save_path)

    except KeyboardInterrupt:
        logger.info('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + '{}_epoch_{}.pth'.format(model_name,epoch + 1))
        logger.info('save checkpoints successfully!')
        raise

def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res = model(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        logger.info('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + '{}_epoch_best.pth'.format(model_name))
        logger.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    logger.info("Start train...")
    for epoch in range(start_epoch, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)

import logging
import time

import numpy as np
import torch

from lib.utils.img_utils import trans_coords_from_patch_to_org_3d
from lib.core.integral_loss import get_result_func, heatmap_to_joints
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


def train_integral(config, train_loader, model, criterion, heatmap_criterion, bone_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    joint_losses = AverageMeter()  # 核心关节损失
    heatmap_losses = AverageMeter()  # 热图损失
    bone_losses = AverageMeter()  # 骨长损失

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_data, batch_label, batch_label_weight, meta = data

        optimizer.zero_grad()

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_label_weight = batch_label_weight.cuda()

        batch_size = batch_data.size(0)
        # compute output
        preds = model(batch_data)

        loss = criterion(preds, batch_label, batch_label_weight)
        # 1. 确定真实热图：你的`batch_label`就是真实热图（标签），直接使用（无需额外提取）
        target_heatmap = batch_label  # 你的真实热图就是batch_label，已移到GPU，直接用
        # 2. 计算热图损失（用你传入的heatmap_criterion，模型输出是preds）
        heatmap_loss = heatmap_criterion(preds, target_heatmap)
        pred_joints = heatmap_to_joints(config, preds)  # 预测关节 (B, 17×3)
        gt_joints = heatmap_to_joints(config, batch_label)  # 真实关节 (B, 17×3)
        # 2. 计算骨长损失（传入关节坐标+可见性权重）
        bone_loss = bone_criterion(pred_joints, gt_joints, batch_label_weight)
        # 3. 加权叠加总损失（核心损失loss占0.85，热图损失占0.15，保持主导）
        total_loss = 0.7 * loss + 0.2 * heatmap_loss + 0.1*bone_loss
        del batch_data, batch_label, batch_label_weight, preds

        # compute gradient and do update step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # record loss
        losses.update(total_loss.item(), batch_size)
        joint_losses.update(loss.item(), batch_size)
        heatmap_losses.update(heatmap_loss.item(), batch_size)
        bone_losses.update(bone_loss.item(), batch_size)
        del loss, heatmap_loss, bone_loss, total_loss, pred_joints, gt_joints
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'JointLoss {joint.val:.5f} ({joint.avg:.5f})\t' \
                  'HeatmapLoss {heat.val:.5f} ({heat.avg:.5f})\t' \
                  'BoneLoss {bone.val:.5f} ({bone.avg:.5f})\t' \
                  'TotalLoss {total.val:.5f} ({total.avg:.5f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=batch_size / batch_time.val,
                data_time=data_time,
                joint=joint_losses, heat=heatmap_losses, bone=bone_losses, total=losses)
            logger.info(msg)


def validate_integral(val_loader, model):
    print("Validation stage")
    result_func = get_result_func()

    # switch to evaluate mode
    model.eval()

    preds_in_patch_with_score = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            batch_data, batch_label, batch_label_weight, meta = data

            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

            # compute output
            preds = model(batch_data)
            del batch_data, batch_label, batch_label_weight


            preds_in_patch_with_score.append(result_func(256, 256, preds))
            del preds

        _p = np.asarray(preds_in_patch_with_score)

        # Dirty solution for partial batches
        if len(_p.shape) < 2:
            tp = np.zeros(((_p.shape[0] - 1) * _p[0].shape[0] + _p[-1].shape[0], _p[0].shape[1], _p[0].shape[2]))

            start = 0
            end = _p[0].shape[0]

            for t in _p:
                tp[start:end] = t
                start = end
                end += t.shape[0]

            _p = tp
        else:
            _p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))

        preds_in_patch_with_score = _p[0: len(val_loader.dataset)]

        return preds_in_patch_with_score


def eval_integral(epoch, preds_in_patch_with_score, val_loader, final_output_path, debug=False):
    print("Evaluation stage")
    # From patch to original image coordinate system
    imdb_list = val_loader.dataset.db
    imdb = val_loader.dataset

    preds_in_img_with_score = []

    for n_sample in range(len(val_loader.dataset)):
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], 256, 256,
                                              2000, 2000))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    name_value, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path, debug=debug)
    for name, value in name_value:
        logger.info('Epoch[%d] Validation-%s %f', epoch, name, value)

    return perf

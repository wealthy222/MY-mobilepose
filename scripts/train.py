import argparse
import os
import pprint
import shutil
import _init_paths
import math  # 余弦退火需要计算三角函数，必须导入

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.function import train_integral
from lib.core.function import validate_integral, eval_integral
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger

from lib.utils.cameras import Camera
import lib.core.integral_loss as loss
import lib.dataset as dataset
import lib.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int,
                        default=8)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    best_perf = 0.0

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose3d_resnet.get_pose_net(config, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)

    shutil.copy2(
        args.cfg,
        final_output_dir
    )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    loss_fn = eval('loss.'+config.LOSS.FN)
    criterion = loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM).cuda()
    heatmap_criterion = loss.HeatmapCrossEntropyLoss(num_joints=17).cuda()
    bone_criterion = loss.BoneLengthRegularizationLoss(num_joints=17).cuda()

    # define training, validation and evaluation routines
    train = train_integral
    validate = validate_integral
    evaluate = eval_integral

    optimizer = get_optimizer(config, model)

    # 新代码：余弦退火LR（适配140轮长训练）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.TRAIN.END_EPOCH,  # 总轮数=train.yaml里的END_EPOCH=140
        eta_min=1e-6,  # 最小学习率，避免衰减到0
        last_epoch=-1  # 从第0轮开始衰减
    )

    # Resume from a trained model
    # 修复核心：判断条件改成“只有RESUME非空时，才加载”
    if config.MODEL.RESUME != '' and os.path.isfile(config.MODEL.RESUME):
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
    else:
        # 首次训练，RESUME为空，跳过加载，打印提示
        logger.info('=> no resume model found, start training from scratch')

    # Choose the dataset, either Human3.6M or mpii
    ds = eval('dataset.'+config.DATASET.DATASET)

    # Data loading code
    train_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TRAIN_SET,
        is_train=True
    )
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):

        # train for one epoch
        train(config, train_loader, model, criterion, heatmap_criterion, bone_criterion, optimizer, epoch)

        # evaluate on validation set
        preds_in_patch_with_score = validate(valid_loader, model)
        acc = evaluate(epoch, preds_in_patch_with_score, valid_loader, final_output_dir, debug=config.DEBUG.DEBUG)

        perf_indicator = 500. - acc if config.DATASET.DATASET == 'h36m' or 'mpii_3dhp' or 'jta' else acc

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        lr_scheduler.step()

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

if __name__ == '__main__':
    main()

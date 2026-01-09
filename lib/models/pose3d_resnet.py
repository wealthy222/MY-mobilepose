import os
import logging

import torch
import torch.nn as nn
from .mobilenetv3 import MobileNetV3_Large

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class PoseMobileNetV3(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PoseMobileNetV3, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.volume = cfg.MODEL.VOLUME

        # 核心替换：MobileNetV3-Large Backbone
        self.backbone = MobileNetV3_Large(
            num_joints=cfg.MODEL.NUM_JOINTS,
            heatmap_size=cfg.MODEL.HEATMAP_SIZE if hasattr(cfg.MODEL, 'HEATMAP_SIZE') else (64, 64)
        )

        # 以下复制自PoseResNet的deconv和final层（完全不变）
        self.inplanes = cfg.MODEL.NUM_JOINTS * 4

        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        self.subpixel_upsample = nn.Sequential(
            # 第一层：卷积扩展通道→BN→激活→PixelShuffle（放大2倍）
            nn.Conv2d(
                in_channels=self.inplanes,  # 现在是deconv最后一层通道（256）
                out_channels=self.inplanes * 4,  # 256×4=1024
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.inplanes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),  # 尺寸×2，通道÷4（1024→256）

            # 第二层：再次卷积扩展→BN→激活→PixelShuffle（再放大2倍）
            nn.Conv2d(
                in_channels=self.inplanes,  # 256
                out_channels=self.inplanes * 4,  # 1024
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.inplanes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),  # 尺寸再次×2，总计×4
        )
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS * cfg.MODEL.DEPTH_RES if self.volume else cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        if not self.volume:
            self.avgpool = nn.AvgPool2d(kernel_size=int(cfg.MODEL.IMAGE_SIZE[0] / 2 ** 5), stride=1)
            self.depth_fc = nn.Linear(2048, cfg.MODEL.NUM_JOINTS * cfg.MODEL.DEPTH_RES)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # MobileNetV3前向传播（替换原ResNet的conv1-layer4）
        B = x.size(0)
        x = self.backbone(x)

        # 以下复制自PoseResNet的forward后半段（完全不变）
        if self.volume:
            x = self.deconv_layers(x)
            x = self.subpixel_upsample(x)  # 亚像素放大4倍（64×64→256×256）
            x = self.final_layer(x)
            return x
        else:
            y = x

            x = self.deconv_layers(x)
            x = self.subpixel_upsample(x)  # 亚像素放大4倍
            x = self.final_layer(x)

            y = self.avgpool(y)
            y = y.view(y.size(0), -1)
            y = self.depth_fc(y)

            return x, y

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            if 'mpii' in pretrained:
                logger.info('=> loading pretrained MPII model {}'.format(pretrained))
                self.load_pretrained_pose_model(pretrained)
            elif 'coco' in pretrained:
                logger.info('=> loading pretrained COCO model {}'.format(pretrained))
                self.load_pretrained_pose_model(pretrained)
            elif 'imagenet' in pretrained:
                pretrained_state_dict = torch.load(pretrained)
                logger.info('=> loading pretrained imagenet model {}'.format(pretrained))
                self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

    def load_pretrained_pose_model(self, pretrained):
        def removekey(d, key):
            r = dict(d)
            del r[key]
            return r

        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()

        if len(pretrained_dict.keys()) == len([x for x in list(pretrained_dict.keys()) if 'module' in x]):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove 'module'
                new_state_dict[name] = v
            pretrained_dict = new_state_dict

        for k in pretrained_dict.keys():
            if k in model_dict.keys():
                if model_dict[k].shape != pretrained_dict[k].shape:
                    logger.info('WARNING! There is a mismatch in => %s (%s, %s)' % (k, model_dict[k].size(),
                                                                                    pretrained_dict[k].size()))
                    pretrained_dict = removekey(pretrained_dict, k)
            else:
                logger.info('%s not in model_dict' % k)

        self.load_state_dict(pretrained_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    # 只保留MobileNetV3逻辑，彻底删除ResNet分支
    model = PoseMobileNetV3(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model

import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math

def weighted_mse_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_smooth_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    diff = input - target
    abs = torch.abs(diff)
    out = torch.where(abs<1., 0.5*diff**2, abs-0.5)

    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    # 替换旧版broadcast用法，直接在目标设备创建张量
    accu_x = accu_x * torch.arange(float(x_dim), device=accu_x.device)
    accu_y = accu_y * torch.arange(float(y_dim), device=accu_y.device)
    accu_z = accu_z * torch.arange(float(z_dim), device=accu_z.device)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z

def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    else:
        assert 0, 'Not Implemented!'
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class L2JointLocationLoss(nn.Module):
    def __init__(self, num_joints,size_average=True, reduce=True, norm=False):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        print(num_joints)

        pred_jts = softmax_integral_tensor(preds, self.num_joints, self.output_3d, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class L1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class SmoothL1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(SmoothL1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_smooth_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

def get_loss_func(config):
    if config.loss_type == 'L1':
        return L1JointLocationLoss(config.output_3d)
    elif config.loss_type == 'L2':
        return L2JointLocationLoss(config.output_3d)
    else:
        assert 0, 'Error. Unknown heatmap type {}'.format(config.heatmap_type)

def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis

def reverse_joint_location_label(patch_width, patch_height, joints):
    joints = joints.reshape((joints.shape[0] // 3, 3))

    joints[:, 0] = (joints[:, 0] + 0.5) * patch_width
    joints[:, 1] = (joints[:, 1] + 0.5) * patch_height
    joints[:, 2] = joints[:, 2] * patch_width
    return joints

def get_joint_location_result(patch_width, patch_height, preds):
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]

    num_joints = 17  # 固定17个关节
    hm_depth = preds.shape[1] // num_joints

    pred_jts = softmax_integral_tensor(preds, num_joints, True, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

def get_label_func():
    return generate_joint_location_label

def get_result_func():
    return get_joint_location_result

def merge_flip_func(a, b, flip_pair):
    # NOTE: flip test of integral is implemented in net_modules.py
    return a

def get_merge_func(loss_config):
    return merge_flip_func

# 新增：交叉熵热图损失类（适配你的MobileNetV3和H36M）
class HeatmapCrossEntropyLoss(nn.Module):
    def __init__(self, num_joints=17, use_soft=True, eps=1e-8):
        super().__init__()
        self.num_joints = num_joints
        self.use_soft = use_soft  # 和你现有softmax逻辑对齐
        self.eps = eps  # 避免log(0)报错

    def forward(self, pred_heatmap, target_heatmap):
        # 1. 可选：softmax（和你现有softmax_integral_tensor逻辑一致）
        if self.use_soft:
            batch_size = pred_heatmap.shape[0]
            pred_flat = pred_heatmap.reshape(batch_size, -1)
            pred_flat = F.softmax(pred_flat, dim=1)
            pred_heatmap = pred_flat.reshape(pred_heatmap.shape)

        # 2. 计算二元交叉熵损失（直接调用PyTorch API，简单高效）
        return F.binary_cross_entropy(
            input=pred_heatmap + self.eps,
            target=target_heatmap + self.eps,
            reduction='mean'
        )


# ===== 纯新增：骨长比例正则化损失（适配17关节H36M，保障生理合理性）=====
class BoneLengthRegularizationLoss(nn.Module):
    def __init__(self, num_joints=17, eps=1e-8):
        super().__init__()
        self.num_joints = num_joints
        self.eps = eps

        # 【关键】H36M 17关节标准骨骼连接对（父关节→子关节，符合人体生理结构）
        # 索引对应H36M关节顺序：0-骨盆、1-右髋、2-右膝、3-右踝、4-左髋、5-左膝、6-左踝、7-躯干、8-胸部、9-颈部、10-头部、11-右肩、12-右肘、13-右腕、14-左肩、15-左肘、16-左腕
        self.bone_pairs = [
            (0, 1), (1, 2), (2, 3),  # 右侧下肢
            (0, 4), (4, 5), (5, 6),  # 左侧下肢
            (0, 7), (7, 8), (8, 9), (9, 10),  # 躯干+头部
            (8, 11), (11, 12), (12, 13),  # 右侧上肢
            (8, 14), (14, 15), (15, 16)  # 左侧上肢
        ]

    def _compute_bone_length(self, joints_3d):
        """
        计算骨骼长度（输入：关节3D坐标，形状(B, num_joints×3) 或 (B, num_joints, 3)）
        输出：骨骼长度，形状(B, num_bones)
        """
        # 适配你的关节坐标维度（从热图积分输出的是(B, num_joints×3)，转换为(B, num_joints, 3)）
        if len(joints_3d.shape) == 2:
            joints_3d = joints_3d.reshape(-1, self.num_joints, 3)

        bone_lengths = []
        for (parent_idx, child_idx) in self.bone_pairs:
            # 提取父关节和子关节的3D坐标
            parent_joint = joints_3d[:, parent_idx, :]  # (B, 3)
            child_joint = joints_3d[:, child_idx, :]  # (B, 3)

            # 计算欧氏距离（骨骼长度），避免除零
            bone_length = torch.norm(child_joint - parent_joint, dim=1, keepdim=True) + self.eps  # (B, 1)
            bone_lengths.append(bone_length)

        # 拼接所有骨骼长度，形状(B, num_bones)
        return torch.cat(bone_lengths, dim=1)

    def forward(self, pred_joints, gt_joints, joint_vis=None):
        """
        Args:
            pred_joints: 预测3D关节坐标 (B, num_joints×3) 或 (B, num_joints, 3)
            gt_joints: 真实3D关节坐标 (B, num_joints×3) 或 (B, num_joints, 3)
            joint_vis: 关节可见性权重 (B, num_joints×3)（可选，过滤遮挡关节）
        Returns:
            骨长正则化损失（L1损失，和现有关节损失风格一致）
        """
        # 1. 计算预测骨骼长度和真实骨骼长度
        pred_bone_lengths = self._compute_bone_length(pred_joints)
        gt_bone_lengths = self._compute_bone_length(gt_joints)

        # 2. 可选：根据关节可见性过滤遮挡骨骼（如果joint_vis不为空）
        if joint_vis is not None:
            # 转换关节可见性为骨骼可见性（父+子关节都可见，骨骼才有效）
            joint_vis = joint_vis.reshape(-1, self.num_joints, 3)[:, :, 0]  # (B, num_joints)
            bone_vis = []
            for (parent_idx, child_idx) in self.bone_pairs:
                bone_vis.append(joint_vis[:, parent_idx] * joint_vis[:, child_idx])  # 父+子都可见为1
            bone_vis = torch.stack(bone_vis, dim=1)  # (B, num_bones)
            # 加权过滤遮挡骨骼
            pred_bone_lengths = pred_bone_lengths * bone_vis
            gt_bone_lengths = gt_bone_lengths * bone_vis

        # 3. 计算骨长L1损失（稳定，和现有L1/SmoothL1关节损失对齐）
        bone_loss = torch.mean(torch.abs(pred_bone_lengths - gt_bone_lengths))

        return bone_loss


# ===== 新增：辅助函数（从热图转换为关节坐标，适配你的模型输出）=====
def heatmap_to_joints(config, heatmap):
    """
    从模型输出的热图，转换为3D关节坐标（复用你的softmax_integral_tensor逻辑）
    """
    num_joints = config.MODEL.NUM_JOINTS
    hm_width = heatmap.shape[-1]
    hm_height = heatmap.shape[-2]
    hm_depth = heatmap.shape[1] // num_joints

    pred_joints = softmax_integral_tensor(
        heatmap, num_joints, output_3d=True,
        hm_width=hm_width, hm_height=hm_height, hm_depth=hm_depth
    )
    return pred_joints
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


def cross_entropy2d(input, target, weight=None, size_average=True, softmax_used=False, reduction='mean',
                    cls_num_list=None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')

    # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target = target.view(-1)
    if softmax_used:
        loss = F.nll_loss(
            input, target, weight=weight, size_average=size_average, ignore_index=255
        )
    else:
        loss = F.cross_entropy(
            input, target, weight=weight, ignore_index=255, reduction=reduction
        )
    return loss


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label == 1).float()) + 0.0001
        neg_num = torch.sum((label == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1 - label) / 2 * mask *
                           torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
                           ) / neg_num
        loss = loss_1 + loss_2
        return loss


def hybrid_loss(predictions, target, weight=None):
    """Calculating the loss"""
    if weight is None:
        weight = [0.2, 0.2, 0.2, 0.2, 0.2]
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i, prediction in enumerate(predictions):
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i] * (bce + dice)  # - ssimloss

    return loss


def contrastive_loss(a, b, label):
    cos_sim = nn.CosineSimilarity()
    mask_0 = torch.where(label == 0, 1, 0)
    # 由于裁剪的问题，可能导致一张图里发生变化的像素为0，所以需要加一个很小的数
    loss = (cos_sim(a * mask_0, b * mask_0).sum() + 0.001) / (cos_sim(a * label, b * label).sum() + 0.001)
    return loss

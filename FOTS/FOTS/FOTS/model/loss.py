import torch
import torch.nn as nn
from torch.nn import CTCLoss
import einops

from .ghm import GHMC


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map

class DetectionLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(DetectionLoss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo), torch.sum(pred_score + pred_geo)

        # classify_loss = self.ghmc(einops.rearrange(pred_score, 'b c h w -> (b h w) c'),
        #                           einops.rearrange(gt_score, 'b c h w -> (b h w) c'),
        #                           einops.rearrange(ignored_map, 'b c h w -> (b h w) c'))

        #print("I am inside the detection loss")

        classify_loss, iou = self.get_dice_loss(
            gt_score, pred_score * (1 - ignored_map.byte())
        )
        iou_loss_map, angle_loss_map = self.get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        print(
            "classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}".format(
                classify_loss, angle_loss, iou_loss
            )
        )
        return geo_loss, classify_loss , iou

    def get_dice_loss(self, gt_score, pred_score):
        inter = torch.sum(gt_score * pred_score)
        union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
        return 1.0 - (2 * inter / union) , inter/union

    def get_geo_loss(self, gt_geo, pred_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
        return iou_loss_map, angle_loss_map


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss(zero_infinity=True) # pred, pred_len, labels, labels_len

    def forward(self, *input):
        gt, pred = input[0], input[1]
        loss = self.ctc_loss( torch.log_softmax(pred[0], dim=-1), gt[0].cpu(), pred[1].int(), gt[1].cpu())
        if torch.isnan(loss):
            raise RuntimeError()
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        self.detectionLoss = DetectionLoss()
        self.recogitionLoss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                y_true_recog, y_pred_recog,
                training_mask):

        if self.mode == 'recognition':
            recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
            reg_loss = torch.tensor([0.], device=recognition_loss.device)
            cls_loss = torch.tensor([0.], device=recognition_loss.device)
        elif self.mode == 'detection':
            reg_loss, cls_loss, iou = self.detectionLoss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask)
            recognition_loss = torch.tensor([0.], device=reg_loss.device)
        elif self.mode == 'united':
            reg_loss, cls_loss, iou = self.detectionLoss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask)
            if y_true_recog:
                recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
                if recognition_loss <0 :
                    import ipdb; ipdb.set_trace()

        #recognition_loss = recognition_loss.to(detection_loss.device)
        return dict(reg_loss=reg_loss, cls_loss=cls_loss, recog_loss=recognition_loss)
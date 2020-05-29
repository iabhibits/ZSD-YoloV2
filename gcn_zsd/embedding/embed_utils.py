"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items

def compute_index(score_thresh):

    # for i in range(score_thresh.shape[1]):
    #     for j in range(score_thresh.shape[2]):
    #         if score_thresh[0][i][j] == True:
    #             return i,j
    index = []
    for i in range(score_thresh.shape[1]):
        for j in range(score_thresh.shape[2]):
            if score_thresh[0][i][j] == True:
                index.append([i,j,i*196+j])
    
    return index

def get_class_name(index,cls_max,cls_max_idx):
    cls_max_flat = cls_max.view(-1)
    cls_max_idx_flat = cls_max_idx.view(-1)
    max_i = -1
    max_j = -1
    max_sc = -1
    for ind in index:
        if cls_max_flat[ind[2]] > max_sc:
            max_sc = cls_max_flat[ind[2]]
            max_i = ind[0]
            max_j = ind[1]
    cls_idx = cls_max_idx_flat[max_i*196+max_j]
    return max_i,max_j,cls_idx


def gcn_post_processing(logits, image_size, gt_classes, anchors, conf_threshold, nms_threshold):
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    class_output, bb_output = logits
    # if isinstance(logits, Variable):
    #     logits = logits.data

    if isinstance(class_output, Variable):
        class_output = class_output.data
    if isinstance(bb_output, Variable):
        bb_output = bb_output.data

    # if logits.dim() == 3:
    #     logits.unsqueeze_(0)

    if class_output.dim() == 3:
        class_output.unsqueeze_(0)
    if bb_output.dim() == 3 :
        bb_output.unsqueeze_(0)

    batch = class_output.size(0)
    h = class_output.size(2)
    w = class_output.size(3)
    class_output_emb = class_output

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    #logits = logits.view(batch, num_anchors, -1, h * w)
    class_output = class_output.view(batch, num_anchors, -1, h* w)
    bb_output = bb_output.view(batch,num_anchors,-1,h*w)
    bb_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)
    bb_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)
    bb_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)
    bb_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)
    bb_output[:, :, 4, :].sigmoid_()

    with torch.no_grad():
        cls_scores = torch.nn.functional.softmax(class_output[:, :, 0:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    cls_max.mul_(bb_output[:, :, 4, :])

    score_thresh = cls_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    # Made changes to get classifier weights
    index = compute_index(score_thresh)
    row,col,cls_idx = get_class_name(index,cls_max,cls_max_idx)
    if col < 14:
        x_1 = 0
        x_2 = col
    else:
        x_1 = col//14
        x_2 = col - (col//14)*14

    emb = class_output_emb.squeeze()
    emb = emb.transpose(0,2)
    embed = emb[x_1][x_2]
    # embed = torch.transpose(embed, 0, 1)
    # embed = embed[col]

    output = embed, cls_idx
    return output

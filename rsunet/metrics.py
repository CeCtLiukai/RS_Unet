import torch

def mean_iou(
    pred,
    target,
    num_classes,
    ignore_index = 255      
):
    #排除无效像素, 生成bool矩阵
    valid  = (target!=ignore_index)

    #利用bool掩码生成有效矩阵
    target = target[valid]
    pred = pred[valid]

    #ious保存结果
    ious = []
    for class_idx in range(num_classes):
        #统计对应的类别的bool矩阵
        #不要直接在用targetcls，否则就是一直拿bool张量做比较
        target_cls = (target == class_idx)
        pred_cls = (pred == class_idx)

        intersection = (target_cls & pred_cls).sum().float()
        union = (target_cls | pred_cls).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(ious).mean()

def update_confusion_matrix(
        confmat,
        pred,
        target,
        num_classes,
        ignore_index=255
):
    #拉平方便后续构建索引
    pred = pred.flatten().long().cpu()
    target = target.flatten().long().cpu()

    #生层无效索引mask
    mask = (target != ignore_index)

    #将mask应用于pred和target
    pred = pred[mask]
    target = target[mask]

    #再次校验
    valid_range = (
        (target>=0)
        &(target<num_classes)
        &(pred>=0)
        &(pred<num_classes)
    )

    pred = pred[valid_range]
    target = target[valid_range]

    #构建(target,pred)唯一索引索引
    index = target * num_classes + pred
    '利用target * num_class将其内部原本连续的标签分开'
    '分开多大，应该至少分开num_classes,这样加上偏移后才不会重叠'

    #统计的是一个batch的confmat
    hist = torch.bincount(
        index,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).to(confmat.device)

    confmat += hist
    return confmat


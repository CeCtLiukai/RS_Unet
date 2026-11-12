import torch


def seg_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index=255):
    """
    :param logits:模型输出得分 (N,C,H,W)
    :param target:标签 (N,H,W)
    :param ignore_index:未标注/背景类映射下标 255
    :return:
    """
    # '统一对对其像素数'
    n, c, h, w = logits.shape
    logits = logits.permute(0, 2, 3, 1)
    logits = logits.reshape(-1, c)

    target = target.reshape(-1).long()

    # '找到需要舍去的元素'
    '因为有些target_val被映射到ignore'
    valid = target != ignore_index

    logits = logits[valid]
    target = target[valid]

    if target.numel() == 0:
        return logits.sum() * 0.0

    # '做损失'
    log_prob = torch.log_softmax(logits, dim=1)

    selected_log_probs = log_prob[
        torch.arange(
            target.numel(),
            device=target.device
        ),
        target
    ]

    loss = -selected_log_probs.mean()
    return loss
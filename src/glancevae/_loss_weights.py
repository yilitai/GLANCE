import copy


def get_scheduled_weight(
    epoch: int,
    max_weight: float,
    start_epoch: int = 0,
    warmup_epochs: int = 0,
    min_weight: float = 0.0,
    end_epoch: int | None = None,
) -> float:
    """
    更通用的损失权重调度函数：
    - start_epoch 之前：权重为 0
    - [start_epoch, start_epoch + warmup_epochs]：线性爬坡
    - 达到 max_weight 后保持常数
    - 若设置 end_epoch，则在 end_epoch 之后权重重新变为 0

    Parameters
    ----------
    epoch
        当前 epoch，从 0 开始。
    max_weight
        该损失项的目标最大权重。
    start_epoch
        该损失项开始启用的 epoch。
    warmup_epochs
        线性 warmup 的 epoch 数。若 <= 0，则到 start_epoch 直接开启到 max_weight。
    min_weight
        warmup 起始权重。
    end_epoch
        若不为 None，则在 epoch > end_epoch 后关闭该损失项。

    Returns
    -------
    float
        当前 epoch 下的权重。
    """
    if end_epoch is not None and epoch > end_epoch:
        return 0.0

    if epoch < start_epoch:
        return 0.0

    if warmup_epochs <= 0:
        return max_weight

    effective_epoch = epoch - start_epoch
    t = min(max(effective_epoch / warmup_epochs, 0.0), 1.0)
    return min_weight + (max_weight - min_weight) * t


def merge_schedules(default_dict, input_dict):
    final_dict = copy.deepcopy(default_dict)

    if input_dict is None:
        return final_dict

    for k, v in input_dict.items():
        if k in final_dict and isinstance(v, dict) and isinstance(final_dict[k], dict):
            # Deep merge: only override the keys the user explicitly provides.
            # Keys present in the default but absent from the user dict are kept.
            final_dict[k].update(v)
        else:
            final_dict[k] = v

    return final_dict

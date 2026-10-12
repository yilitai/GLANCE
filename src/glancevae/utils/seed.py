import random
import os
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    设置全局随机种子以保证实验的可复现性。

    涵盖: Python random, NumPy, PyTorch (CPU & GPU), CUDA 卷积算法确定性, 以及 Hash 种子。

    Args:
        seed (int): 随机种子数值.
        deterministic (bool): 是否强制使用确定性卷积算法 (cuDNN).
                              设置为 True 可以保证复现，但可能会降低训练速度。
    """
    # 1. Python 标准库
    random.seed(seed)

    # 2. 操作系统环境 (影响 Hash 行为，如 set, dict 的迭代顺序)
    # 注意：这通常只对新启动的子进程有效，要在当前进程完全生效需要在脚本运行前设置
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy (Pandas 依赖这个)
    np.random.seed(seed)

    # 4. PyTorch
    torch.manual_seed(seed)  # CPU 种子
    torch.cuda.manual_seed(seed)  # 当前 GPU 种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 种子 (如果是多卡训练)

    # 5. cuDNN 确定性设置 (关键)
    if deterministic:
        # 强制使用确定性算法 (可能会变慢，但能保证结果一致)
        torch.backends.cudnn.deterministic = True
        # 禁止 cudnn 寻找最优算法 (因为最优算法可能是不确定的)
        torch.backends.cudnn.benchmark = False
    else:
        # 允许 cudnn 寻找最快的算法 (通常用于生产环境或不追求严格复现的场景)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Global seed set to {seed} (deterministic={deterministic})")


def check_random_state(random_state):
    """
    sklearn-style random state validation.

    Parameters
    ----------
    random_state : None, int, or np.random.Generator
        None  -> fully random (new Generator each call)
        int   -> seed a new Generator with this integer
        Generator -> use as-is

    Returns
    -------
    np.random.Generator
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise ValueError(
        f"random_state must be None, an int, or a np.random.Generator, "
        f"got {type(random_state)}"
    )

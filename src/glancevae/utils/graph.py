import numpy as np
import torch

from typing import Union, Tuple
from scipy.spatial import cKDTree


def run_tree_query(
    coords_np: np.ndarray,
    k: int,
    thread_limits: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:

    # 1. 构建 KDTree
    tree = cKDTree(coords_np)

    # 2. 查询
    distances, indices = (None, None)
    if thread_limits <= 0:
        distances, indices = tree.query(
            coords_np,
            k=k + 1,
            workers=-1,
        )

    else:
        distances, indices = tree.query(
            coords_np,
            k=k + 1,
            workers=thread_limits,
        )

    # 4. 返回结果
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    return neighbor_distances, neighbor_indices


def graph_indexing(
    coords: Union[np.ndarray, torch.Tensor],
    k: int,
    thread_limits: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    # 1. 统一转换为 Numpy Array (CPU)
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    _, neighbor_indices = run_tree_query(
        coords_np=coords_np,
        k=k,
        thread_limits=thread_limits,
    )

    return neighbor_indices


def batched_graph_indexing(
    coords: Union[np.ndarray, torch.Tensor],
    batch_index: Union[np.ndarray, torch.Tensor],
    k: int,
    thread_limits: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在每个 batch 内部分别做 kNN（cKDTree），返回 numpy 格式结果
    顺序与原 coords 保持一致，indices 为全局 index
    """

    # -------- 1. 转 numpy --------
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    if isinstance(batch_index, torch.Tensor):
        batch_np = batch_index.detach().cpu().numpy()
    else:
        batch_np = batch_index

    N = coords_np.shape[0]

    # -------- 2. 预分配 --------
    neighbor_indices = np.empty((N, k), dtype=np.int64)

    # -------- 3. 按 batch 处理 --------
    for b in np.unique(batch_np):
        mask = batch_np == b
        idx = np.where(mask)[0]  # 全局 index
        coords_sub = coords_np[mask]

        _, ind_sub = run_tree_query(
            coords_sub,
            k=k,
            thread_limits=thread_limits,
        )

        # 局部 → 全局 index
        neighbor_indices[idx] = idx[ind_sub]

    return neighbor_indices


def batched_graph_indexing_with_distance(
    coords: Union[np.ndarray, torch.Tensor],
    batch_index: Union[np.ndarray, torch.Tensor],
    k: int,
    thread_limits: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在每个 batch 内部分别做 kNN（cKDTree），返回 numpy 格式结果
    顺序与原 coords 保持一致，indices 为全局 index
    """

    # -------- 1. 转 numpy --------
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    if isinstance(batch_index, torch.Tensor):
        batch_np = batch_index.detach().cpu().numpy()
    else:
        batch_np = batch_index

    N = coords_np.shape[0]

    # -------- 2. 预分配 --------
    neighbor_indices = np.empty((N, k), dtype=np.int64)
    neighbor_distances = np.empty((N, k), dtype=np.float32)

    # -------- 3. 按 batch 处理 --------
    for b in np.unique(batch_np):
        mask = batch_np == b
        idx = np.where(mask)[0]  # 全局 index
        coords_sub = coords_np[mask]

        dist_sub, ind_sub = run_tree_query(
            coords_sub,
            k=k,
            thread_limits=thread_limits,
        )

        # 局部 → 全局 index
        neighbor_indices[idx] = idx[ind_sub]
        neighbor_distances[idx] = dist_sub

    return neighbor_distances, neighbor_indices

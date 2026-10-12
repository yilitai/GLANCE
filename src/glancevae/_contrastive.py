import torch
import torch.nn.functional as F

import numpy as np

from typing import Union, Tuple
from scipy.spatial import cKDTree

from glancevae.utils.graph import run_tree_query
from glancevae.utils.seed import check_random_state


def same_batch_contrastive_loss(
    z,
    pos_edges,
    neg_edges,
    margin: float = 10.0,
):
    """
    Same-batch contrastive loss with adaptive margin.

    The effective margin is computed as:
        effective_margin = margin * median(pos_dist)

    This makes the margin scale-invariant: it adapts to the current
    latent space scale rather than relying on an absolute distance threshold.

    The effective margin is additionally capped at the 90th percentile of the
    current negative pair distances to prevent gradient explosion in early
    training when the latent space is not yet converged.

    Parameters
    ----------
    z : torch.Tensor
        shape = (N, D)
    pos_edges : torch.LongTensor
        shape = (E_pos, 2), each row is (i, j)
    neg_edges : torch.LongTensor
        shape = (E_neg, 2), each row is (i, k)
    margin : float
        Multiplier on the median positive pair distance.
        Negative pairs must be at least (margin * median_pos_dist) apart.
        Default 10.0.

    Returns
    -------
    loss : torch.Tensor
        scalar
    """
    loss = 0.0
    n_terms = 0
    pos_dist = None

    if pos_edges is not None and pos_edges.numel() > 0:
        zi = z[pos_edges[:, 0]]
        zj = z[pos_edges[:, 1]]
        pos_dist_sq = ((zi - zj) ** 2).sum(dim=1)
        pos_dist = torch.sqrt(pos_dist_sq + 1e-8)
        # Clamp pos_loss to prevent runaway gradients in early training
        # when the latent space is not yet converged and distances are large.
        pos_loss = pos_dist_sq.mean().clamp(max=100.0)
        loss = loss + pos_loss
        n_terms += 1

    if neg_edges is not None and neg_edges.numel() > 0:
        zi = z[neg_edges[:, 0]]
        zk = z[neg_edges[:, 1]]
        neg_dist = torch.sqrt(((zi - zk) ** 2).sum(dim=1) + 1e-8)

        # Adaptive margin: scale relative to current positive distance distribution.
        # detach to avoid backprop through the threshold itself.
        if pos_dist is not None:
            effective_margin = margin * pos_dist.detach().median()
        else:
            effective_margin = margin * neg_dist.detach().median()

        # Cap at 90th percentile of neg_dist to prevent gradient explosion
        # in early training.
        margin_cap = torch.quantile(neg_dist.detach(), 0.9)
        effective_margin = effective_margin.clamp(max=margin_cap)

        # Clamp neg_loss for the same reason as pos_loss above.
        neg_loss = F.relu(effective_margin - neg_dist).pow(2).mean().clamp(max=100.0)
        loss = loss + neg_loss
        n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    return loss / n_terms


def build_same_batch_edges(
    x,
    batch_index,
    k_pos=10,
    n_neg=20,
    far_quantile=0.7,
    random_state=None,
    thread_limits: int = -1,
):
    """
    Build same-batch positive (kNN) and negative (far pairs) edges.
    Uses cKDTree for fast neighbor search and far-point threshold estimation.

    Parameters
    ----------
    x : np.ndarray
        shape = (N, D)
    batch_index : np.ndarray
        shape = (N,)
    k_pos : int
        Number of positive neighbors per sample.
    n_neg : int
        Number of negative samples per sample.
    far_quantile : float
        Negatives are sampled from points farther than this quantile within batch.
    random_state : None, int, or np.random.Generator
        Controls randomness of negative sampling.
        None (default) -> fully random each call.
        int -> reproducible seed.
        np.random.Generator -> use directly.
    thread_limits : int
        Number of threads for cKDTree queries. -1 = use all cores.
    """
    rng = check_random_state(random_state)
    workers = -1 if thread_limits <= 0 else thread_limits

    pos_edges = []
    neg_edges = []

    unique_batches = np.unique(batch_index)

    for b in unique_batches:
        idx = np.where(batch_index == b)[0]
        xb = x[idx]
        n = len(idx)

        if n <= k_pos + 1:
            continue

        # ------------------------------------------------------------------ #
        # Positive edges: same-batch kNN via cKDTree
        # ------------------------------------------------------------------ #
        actl_k_pos = min(k_pos, n - 1)
        _, ind_pos = run_tree_query(xb, k=actl_k_pos, thread_limits=thread_limits)

        # Vectorised: build all pos edges at once
        local_anchors = np.repeat(np.arange(n), actl_k_pos)  # (n * k,)
        local_neighbors = ind_pos.ravel()  # (n * k,)
        global_anchors = idx[local_anchors]
        global_neighbors = idx[local_neighbors]
        batch_pos = np.stack([global_anchors, global_neighbors], axis=1)
        pos_edges.append(batch_pos)

        # ------------------------------------------------------------------ #
        # Negative edges: query far candidates via cKDTree, then subsample
        # randomly. Query enough far neighbours to give the RNG real choice,
        # then pick n_neg per anchor via vectorised shuffle.
        # ------------------------------------------------------------------ #
        tree = cKDTree(xb)

        far_k = max(1, int(np.ceil(far_quantile * n)))
        far_k = min(far_k, n - 1)

        # Query all far candidates (beyond the near neighbourhood) as the
        # sampling pool. Using all far points rather than a fixed-size subset
        # preserves the original sampling distribution and avoids bias toward
        # a narrow distance band.
        n_neg_pool = n - far_k - 1
        if n_neg_pool > 0:
            total_k = min(far_k + n_neg_pool + 1, n)
            _, ind_far = tree.query(xb, k=total_k, workers=workers)

            # ind_far[:, 0] is self, [:, 1:far_k+1] are near neighbours,
            # [:, far_k+1:] are the far candidate pool  (n, n_neg_pool)
            pool = ind_far[:, far_k + 1 :]  # (n, n_neg_pool)
            pool_size = pool.shape[1]
            actl_n_neg = min(n_neg, pool_size)

            # Sample n_neg candidates per anchor without replacement.
            # rng.choice per row preserves the original sampling distribution.
            sampled_cols = np.stack(
                [
                    rng.choice(pool_size, size=actl_n_neg, replace=False)
                    for _ in range(n)
                ]
            )  # (n, actl_n_neg)
            neg_local = pool[np.arange(n)[:, None], sampled_cols]  # (n, actl_n_neg)

            # Vectorised edge construction
            local_anchors_neg = np.repeat(np.arange(n), actl_n_neg)
            local_negs_flat = neg_local.ravel()
            batch_neg = np.stack([idx[local_anchors_neg], idx[local_negs_flat]], axis=1)
            neg_edges.append(batch_neg)

    if len(pos_edges) == 0:
        pos_edges_t = torch.empty((0, 2), dtype=torch.long)
    else:
        pos_edges_t = torch.from_numpy(np.concatenate(pos_edges, axis=0)).long()

    if len(neg_edges) == 0:
        neg_edges_t = torch.empty((0, 2), dtype=torch.long)
    else:
        neg_edges_t = torch.from_numpy(np.concatenate(neg_edges, axis=0)).long()

    return pos_edges_t, neg_edges_t


def geom_contrastive_loss(
    z,
    pos_edges,
    neg_edges,
):
    """
    Geometric refinement contrastive loss for the ``geom_reg`` path.

    Positive (attractive) edges use squared Euclidean distance, identical
    to ``same_batch_contrastive_loss``.

    Negative (repulsive) edges use log-repulsive loss ``-log(d + eps)``
    instead of a margin hinge. This produces a non-zero gradient at every
    distance so isolated / rare clusters are continuously pushed away from
    large clusters even when already far apart.

    ``margin`` is intentionally absent: the log form needs no threshold.

    Parameters
    ----------
    z : torch.Tensor, shape (N, D)
    pos_edges : torch.LongTensor, shape (E_pos, 2)
    neg_edges : torch.LongTensor, shape (E_neg, 2)

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    loss = 0.0
    n_terms = 0

    if pos_edges is not None and pos_edges.numel() > 0:
        zi = z[pos_edges[:, 0]]
        zj = z[pos_edges[:, 1]]
        pos_dist_sq = ((zi - zj) ** 2).sum(dim=1)
        pos_loss = pos_dist_sq.mean().clamp(max=100.0)
        loss = loss + pos_loss
        n_terms += 1

    if neg_edges is not None and neg_edges.numel() > 0:
        zi = z[neg_edges[:, 0]]
        zk = z[neg_edges[:, 1]]
        neg_dist = torch.sqrt(((zi - zk) ** 2).sum(dim=1) + 1e-8)
        neg_loss = -torch.log(neg_dist + 1e-8).mean()
        loss = loss + neg_loss
        n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    return loss / n_terms


def build_cross_batch_geom_edges(
    coords: Union[np.ndarray, torch.Tensor],
    k_pos: int = 10,
    n_neg: int = 20,
    far_quantile: float = 0.7,
    mutual: bool = True,
    random_state=None,
    thread_limits: int = -1,
    verbose: bool = False,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Build cross-batch attractive (K-RNN) and repulsive (far-tail) edges
    on a latent embedding for the ``geom_reg`` path.

    Positive edges apply k-reciprocal filtering (``mutual=True``) to
    protect rare / isolated clusters: large-cluster cells do not
    reciprocate edges toward rare cells, so those edges are dropped.

    Negative edges are sampled globally from the far tail and used with
    ``geom_contrastive_loss`` (log-repulsive), which has a non-zero
    gradient at every distance.

    Parameters
    ----------
    coords : np.ndarray or torch.Tensor, shape (N, D)
    k_pos : int
    n_neg : int
    far_quantile : float
    mutual : bool, default True
    random_state : None, int, or np.random.Generator
    thread_limits : int
    verbose : bool

    Returns
    -------
    pos_edges : torch.LongTensor (E_pos, 2)  -- CPU
    neg_edges : torch.LongTensor (E_neg, 2)  -- CPU
    """

    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords, dtype=np.float32)

    rng = check_random_state(random_state)
    workers = -1 if thread_limits <= 0 else thread_limits
    n = coords_np.shape[0]
    actl_k_pos = min(k_pos, n - 1)

    if n <= actl_k_pos + 1:
        return (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
        )

    # ---- Positive edges ----
    _, ind_pos = run_tree_query(coords_np, k=actl_k_pos, thread_limits=thread_limits)
    if mutual:
        adj = np.zeros((n, n), dtype=bool)
        adj[np.repeat(np.arange(n), actl_k_pos), ind_pos.ravel()] = True
        adj_recip = adj & adj.T
        src, dst = np.nonzero(adj_recip)
        pos_edges_np = np.stack([src, dst], axis=1).astype(np.int64)
    else:
        anchors = np.repeat(np.arange(n), actl_k_pos)
        pos_edges_np = np.stack([anchors, ind_pos.ravel()], axis=1)

    # ---- Negative edges (far-tail global sample) ----
    tree = cKDTree(coords_np)
    far_k = min(max(1, int(np.ceil(far_quantile * n))), n - 1)
    n_neg_pool = n - far_k - 1

    if n_neg_pool > 0:
        total_k = min(far_k + n_neg_pool + 1, n)
        _, ind_far = tree.query(coords_np, k=total_k, workers=workers)
        pool = ind_far[:, far_k + 1 :]
        pool_size = pool.shape[1]
        actl_n_neg = min(n_neg, pool_size)
        sampled_cols = np.stack(
            [rng.choice(pool_size, size=actl_n_neg, replace=False) for _ in range(n)]
        )
        neg_local = pool[np.arange(n)[:, None], sampled_cols]
        anchors_neg = np.repeat(np.arange(n), actl_n_neg)
        neg_edges_np = np.stack([anchors_neg, neg_local.ravel()], axis=1)
    else:
        neg_edges_np = np.empty((0, 2), dtype=np.int64)

    pos_edges_t = torch.from_numpy(pos_edges_np).long()
    neg_edges_t = torch.from_numpy(neg_edges_np).long()

    if verbose:
        n_plain = n * actl_k_pos
        tag = (
            f"mutual, {pos_edges_t.shape[0]}/{n_plain} ({pos_edges_t.shape[0]/max(n_plain,1):.1%})"
            if mutual
            else "plain"
        )
        print(
            f"[build_cross_batch_geom_edges] pos={pos_edges_t.shape[0]} ({tag}), neg={neg_edges_t.shape[0]}"
        )

    return pos_edges_t, neg_edges_t

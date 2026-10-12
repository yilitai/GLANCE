import torch
import numpy as np

from scipy.special import stdtr  # Student-T CDF, scipy always has this
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def studentt_cdf(x, loc, scale, df):
    """
    Compute the CDF of a Student-T distribution at x.

    Uses scipy.special.stdtr which is the standard Student-T CDF
    F(t; df) where t = (x - loc) / scale.

    scipy.special.stdtr is available in all scipy versions and is
    numerically stable across the full range of df values.

    Parameters
    ----------
    x     : Tensor, any shape
    loc   : Tensor, broadcastable to x
    scale : Tensor, broadcastable to x
    df    : Tensor, broadcastable to x

    Returns
    -------
    cdf : Tensor, same shape as inputs after broadcasting
    """
    # Standardise: t = (x - loc) / scale
    t = (x - loc) / scale

    # Convert to numpy for scipy, compute, convert back
    device = x.device
    dtype = x.dtype

    t_np = t.detach().cpu().numpy().astype(np.float64)
    df_np = df.detach().cpu().numpy().astype(np.float64)

    cdf_np = stdtr(df_np, t_np)  # shape same as t_np

    return torch.as_tensor(cdf_np, dtype=dtype, device=device)


def find_bimodal_tau_single(values, n_points=500):
    values = values[~np.isnan(values)]
    if len(values) < 20:
        return float(np.median(values))

    kde = gaussian_kde(values, bw_method="scott")
    x = np.linspace(values.min(), values.max(), n_points)
    density = kde(x)

    peaks, _ = find_peaks(density, height=density.max() * 0.1, distance=n_points // 10)

    if len(peaks) >= 2:
        top2 = peaks[np.argsort(density[peaks])[-2:]]
        top2 = np.sort(top2)
        valley_region = density[top2[0] : top2[1]]
        valley_idx = top2[0] + np.argmin(valley_region)
        return float(x[valley_idx])
    else:
        return float(np.median(values))


def find_bimodal_tau(agg_loc_matrix, n_points=500, n_jobs=-1):
    """
    Parameters
    ----------
    agg_loc_matrix : ndarray, shape (N, G)
        agg_loc from get_distribution_params, cells × proteins.
    n_points : int
        KDE resolution. 500 is sufficient, 1000 is slower with little gain.
    n_jobs : int
        Number of parallel jobs. -1 = use all cores.

    Returns
    -------
    tau : ndarray, shape (G,)
    """
    G = agg_loc_matrix.shape[1]

    tau = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(find_bimodal_tau_single)(agg_loc_matrix[:, g], n_points)
        for g in range(G)
    )

    return np.array(tau)

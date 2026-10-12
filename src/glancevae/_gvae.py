import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.distributions import Normal, StudentT, kl_divergence

from glancevae._base import GaussianEncoder, StudentTDecoder
from glancevae._loss_weights import get_scheduled_weight, merge_schedules
from glancevae._settings import settings
from glancevae._contrastive import (
    same_batch_contrastive_loss,
    geom_contrastive_loss,
    build_cross_batch_geom_edges,
)

logger = logging.getLogger("glancevae")

DEFAULT_LOSS_SCHEDULE = {
    "recon_loss": {
        "max_weight": 1.0,
        "start_epoch": 0,
        "warmup_epochs": 0,
        "min_weight": 0.0,
        "end_epoch": None,
    },
    "kl_z": {
        "max_weight": 2.0,
        "start_epoch": 0,
        "warmup_epochs": 1000,
        "min_weight": 0.0,
        "end_epoch": None,
    },
    "manifold_ctr": {
        "max_weight": 1.5,
        "start_epoch": 100,
        "warmup_epochs": 300,
        "min_weight": 0.0,
        "end_epoch": None,
    },
    "geom_reg": {
        "max_weight": 0.02,
        "start_epoch": 2000,  # prepare_geom_refinement 里的 start_epoch 会覆盖这个
        "warmup_epochs": 300,
        "min_weight": 0.0,
        "end_epoch": None,
    },
}


class GaussianVAE(nn.Module):
    def __init__(
        self,
        n_in,
        n_latent=128,
        n_batch_classes=1,
        n_batch_embed=10,
        ae_n_hidden=128,
        ae_n_layers=1,
        noise_std: float = 0.0,
    ):
        """
        Parameters
        ----------
        noise_std : float
            Standard deviation of Gaussian noise added to the input
            expression during training (denoising VAE).  The encoder
            receives the corrupted input; the decoder is always trained
            to reconstruct the original clean expression.  Set to 0.0
            to disable (original behaviour).  Values in [0.1, 0.3] work
            well for log-normalised proteomics data.
        """
        super().__init__()

        self.noise_std = noise_std

        self.batch_embedding = None
        n_cov = 0

        if n_batch_classes > 1:
            self.batch_embedding = nn.Embedding(n_batch_classes, n_batch_embed)
            n_cov = n_batch_embed

        self.encoder = GaussianEncoder(
            n_in=n_in,
            n_out=n_latent,
            n_hidden=ae_n_hidden,
            n_layers=ae_n_layers,
            n_cov=n_cov,
        )

        self.decoder = StudentTDecoder(
            n_in=n_latent,
            n_out=n_in,
            n_hidden=ae_n_hidden,
            n_layers=ae_n_layers,
            n_cov=n_cov,
        )

    def forward(self, features, batch_index=None):
        cov = None
        if self.batch_embedding is not None and batch_index is not None:
            if batch_index.dim() == 2:
                batch_index = batch_index.squeeze(-1)
            cov = self.batch_embedding(batch_index)

        # Corrupt input to encoder during training only.
        # The decoder always reconstructs the original clean expression.
        if self.training and self.noise_std > 0.0:
            features_enc = features + torch.randn_like(features) * self.noise_std
        else:
            features_enc = features

        q_m, q_v, z = self.encoder(features_enc, cov=cov)
        p_loc, p_scale, p_df = self.decoder(z, cov=cov)

        px = StudentT(df=p_df, loc=p_loc, scale=p_scale)
        qz = Normal(q_m, q_v.sqrt())

        return {
            "q_m": q_m,
            "q_v": q_v,
            "px": px,
            "qz": qz,
            "z": z,
            "features": features,  # clean target for loss
        }

    def decode(self, z, batch_index=None):
        cov = None
        if self.batch_embedding is not None and batch_index is not None:
            if batch_index.dim() == 2:
                batch_index = batch_index.squeeze(-1)
            cov = self.batch_embedding(batch_index)

        p_loc, p_scale, p_df = self.decoder(z, cov=cov)
        return p_loc, p_scale, p_df

    def loss(
        self,
        features,
        outputs,
        pos_edges=None,
        neg_edges=None,
        ctr_margin: float = 10.0,
        geom_pos_edges=None,
        geom_neg_edges=None,
    ):
        # Reconstruct clean expression regardless of whether input was noised
        recon_loss = -outputs["px"].log_prob(features).sum(dim=-1)

        pz = Normal(
            torch.zeros_like(outputs["q_m"]),
            torch.ones_like(outputs["q_v"]),
        )
        kl_z = kl_divergence(outputs["qz"], pz).sum(dim=-1)

        loss_dict = {
            "recon_loss": recon_loss.mean(),
            "kl_z": kl_z.mean(),
        }

        if pos_edges is not None or neg_edges is not None:
            loss_dict["manifold_ctr"] = same_batch_contrastive_loss(
                z=outputs["z"],
                pos_edges=pos_edges,
                neg_edges=neg_edges,
                margin=ctr_margin,
            )

        if geom_pos_edges is not None or geom_neg_edges is not None:
            loss_dict["geom_reg"] = geom_contrastive_loss(
                z=outputs["q_m"],
                pos_edges=geom_pos_edges,
                neg_edges=geom_neg_edges,
            )

        return loss_dict


def train_gvae(
    model,
    features,
    batch_index=None,
    pos_edges=None,
    neg_edges=None,
    ctr_margin: float = 10.0,
    max_ctr_edges: int = 20000,
    geom_cfg=None,
    geom_default_schedule=None,
    num_epochs=2500,
    loss_schedule=None,
    lr=0.001,
    log_details=False,
    log_step=10,
    tqdm_desc="Training GLANCE",
    tqdm_ncols=50,
):
    """
    Parameters
    ----------
    loss_schedule : dict or None
        Example::

            {
                "recon_loss":      {"max_weight": 1.0},
                "kl_z":            {"max_weight": 2.0, "warmup_epochs": 1000},
                "manifold_ctr":    {"max_weight": 1.0, "start_epoch": 100,
                                    "warmup_epochs": 400},
            }

    ctr_margin : float
        Adaptive margin multiplier for within-batch contrastive loss.
        Default 10.0.
    max_ctr_edges : int
        Maximum contrastive edges sampled per epoch.  -1 = use all.
    geom_cfg : dict or None
        When not None, enables in-loop geometric refinement (``geom_reg``)
        using K-RNN attractive + log-repulsive negative edges on ``q_m``.
        Built by ``ProGA.prepare_geom_refinement``::

            {
                "k_pos", "n_neg", "far_quantile", "mutual",
                "rebuild_every", "max_edges", "thread_limits", "random_state"
            }

    geom_default_schedule : dict or None
        Injected by ``prepare_geom_refinement``; merged between
        hard-coded defaults and user ``loss_schedule``.
    num_epochs : int, default 2500
    """
    # Start from the canonical defaults; pop loss items that are not active
    # in this run so they never appear in actl_schedule or log output.
    default_schedule = copy.deepcopy(DEFAULT_LOSS_SCHEDULE)

    has_within_ctr = (pos_edges is not None and pos_edges.numel() > 0) or (
        neg_edges is not None and neg_edges.numel() > 0
    )
    if not has_within_ctr:
        default_schedule.pop("manifold_ctr", None)

    has_geom = geom_cfg is not None
    if not has_geom:
        default_schedule.pop("geom_reg", None)

    # Three-level merge: hard-coded -> geom_default_schedule -> user.
    actl_schedule = merge_schedules(default_schedule, geom_default_schedule)
    actl_schedule = merge_schedules(actl_schedule, loss_schedule)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_iter = (
        range(num_epochs)
        if log_details
        else tqdm(
            range(num_epochs),
            desc=f"[GLANCE] {tqdm_desc}",
            ncols=tqdm_ncols,
            disable=settings.tqdm_disabled,
        )
    )

    def _sample_edges(edges, max_edges):
        if edges is None or edges.numel() == 0 or max_edges < 0:
            return edges
        n = edges.shape[0]
        if n <= max_edges:
            return edges
        return edges[torch.randperm(n, device=edges.device)[:max_edges]]

    if log_details:
        logger.info("[GLANCE] active loss schedule: %s", actl_schedule)

    # Geometric edges built lazily at geom_start_epoch.
    geom_pos_edges = None
    geom_neg_edges = None
    device = features.device

    if has_geom:
        geom_start = int(actl_schedule.get("geom_reg", {}).get("start_epoch", 2000))

    global_step = 0
    for epoch in epoch_iter:
        model.train()

        # ---- Build / rebuild geometric edges ----
        if has_geom:
            rebuild_every = geom_cfg.get("rebuild_every", None)
            do_initial = epoch == geom_start
            do_periodic = (
                rebuild_every is not None
                and epoch > geom_start
                and (epoch - geom_start) % int(rebuild_every) == 0
            )
            if do_initial or do_periodic:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    q_m = model(features=features, batch_index=batch_index)["q_m"]
                if was_training:
                    model.train()

                gp, gn = build_cross_batch_geom_edges(
                    coords=q_m,
                    k_pos=geom_cfg["k_pos"],
                    n_neg=geom_cfg["n_neg"],
                    far_quantile=geom_cfg["far_quantile"],
                    mutual=geom_cfg.get("mutual", True),
                    random_state=geom_cfg.get("random_state", None),
                    thread_limits=geom_cfg["thread_limits"],
                    verbose=False,
                )
                geom_pos_edges = gp.to(device)
                geom_neg_edges = gn.to(device)

        optimizer.zero_grad()

        outputs = model(features=features, batch_index=batch_index)

        # Sample within-batch edges (original logic unchanged)
        epoch_pos = None
        epoch_neg = None
        if has_within_ctr:
            epoch_pos = _sample_edges(pos_edges, max_ctr_edges)
            epoch_neg = _sample_edges(neg_edges, max_ctr_edges)

        # Sample geometric edges
        epoch_geom_pos = None
        epoch_geom_neg = None
        if has_geom and geom_pos_edges is not None:
            max_g = geom_cfg.get("max_edges", 50000)
            epoch_geom_pos = _sample_edges(geom_pos_edges, max_g)
            epoch_geom_neg = _sample_edges(geom_neg_edges, max_g)

        loss_dict = model.loss(
            features=features,
            outputs=outputs,
            pos_edges=epoch_pos,
            neg_edges=epoch_neg,
            ctr_margin=ctr_margin,
            geom_pos_edges=epoch_geom_pos,
            geom_neg_edges=epoch_geom_neg,
        )

        current_weights = {
            name: get_scheduled_weight(
                epoch=epoch,
                max_weight=actl_schedule.get(name, {}).get("max_weight", 1.0),
                start_epoch=actl_schedule.get(name, {}).get("start_epoch", 0),
                warmup_epochs=actl_schedule.get(name, {}).get("warmup_epochs", 0),
                min_weight=actl_schedule.get(name, {}).get("min_weight", 0.0),
                end_epoch=actl_schedule.get(name, {}).get("end_epoch", None),
            )
            for name in loss_dict
        }

        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            total_loss = total_loss + current_weights[loss_name] * loss_value

        total_loss.backward()

        if has_within_ctr or has_geom:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        global_step += 1

        if log_details and epoch % log_step == 0:
            percent = 100 * (epoch + 1) / num_epochs
            loss_lines = "\n".join(
                f"  {name}: {loss_dict[name].item():.4f} "
                f"(w={current_weights.get(name, 0.0):.4f})"
                for name in loss_dict
            )
            logger.info(
                "[GLANCE] %s | %6.2f%% | Epoch %04d\n" "  Total: %.4f\n%s\n%s",
                tqdm_desc,
                percent,
                epoch + 1,
                total_loss.item(),
                loss_lines,
                "=" * 52,
            )

    if has_geom and geom_pos_edges is not None:
        logger.info(
            "[GLANCE] Training complete | "
            "Geometric edges: pos_edges=%d, neg_edges=%d",
            geom_pos_edges.shape[0],
            geom_neg_edges.shape[0],
        )
        if batch_index is not None:
            bi = batch_index
            if bi.dim() == 2:
                bi = bi.squeeze(-1)
            same = (
                (bi[geom_pos_edges[:, 0]] == bi[geom_pos_edges[:, 1]])
                .float()
                .mean()
                .item()
            )
            n_b = int(bi.max().item()) + 1
            logger.info(
                "[GLANCE] Same-batch pos-edge fraction: %.3f "
                "(random baseline: %.3f)",
                same,
                1 / max(n_b, 1),
            )
    else:
        logger.info("[GLANCE] Training complete.")

    return model

import gc
import os
import copy
import logging

import torch
import numpy as np
import warnings

from typing import Dict, Union
from scipy.linalg import orthogonal_procrustes

from glancevae._gvae import GaussianVAE, train_gvae, DEFAULT_LOSS_SCHEDULE
from glancevae.utils.label_encoder import BioLabelEncoder
from glancevae.utils.distribution import studentt_cdf, find_bimodal_tau
from glancevae._contrastive import build_same_batch_edges

logger = logging.getLogger("glancevae")


class GLANCE(object):
    def __init__(
        self,
        adata,
        label_col=None,
        key_added="GLANCE",
        n_latent=64,
        n_batch_embed=10,
        ae_n_hidden=None,
        ae_n_layers=2,
        lr=0.001,
        ensemble_size=3,
        noise_std: float = 0.3,
        random_state: Union[None, int, np.random.Generator] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.adata = adata

        self.original_features = torch.as_tensor(
            adata.X,
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------------
        # has_batch is decided exactly once, here, and is the single
        # source of truth for every downstream branch in this class.
        # No other method should re-derive "do we have batches?" by
        # checking self.batch_encoder is None directly -- they should
        # check self.has_batch instead. This keeps the no-batch /
        # batch-aware logic from drifting apart across methods.
        # ------------------------------------------------------------
        self.has_batch = label_col is not None

        if self.has_batch:
            self.batch_encoder = BioLabelEncoder(
                label_col=label_col,
                mode="torch",
                device=device,
            )
            self.batch_index = self.batch_encoder.fit_transform(self.adata)
            self.n_batch_classes = self.batch_encoder.n_classes_
        else:
            self.batch_encoder = None
            self.batch_index = None
            self.n_batch_classes = 1
            n_batch_embed = 0

        self.key_added = key_added
        self.n_in = self.original_features.shape[1]
        self.n_latent = n_latent
        self.n_batch_embed = n_batch_embed
        self.ae_n_layers = ae_n_layers
        self.ae_n_hidden = (
            ae_n_hidden if ae_n_hidden is not None else n_latent + n_batch_embed
        )
        self.lr = lr
        self.ensemble_size = ensemble_size
        self.noise_std = noise_std
        self.random_state = random_state
        self.device = device

        self.model = GaussianVAE(
            n_in=self.n_in,
            n_latent=self.n_latent,
            n_batch_classes=self.n_batch_classes,
            n_batch_embed=self.n_batch_embed,
            ae_n_hidden=self.ae_n_hidden,
            ae_n_layers=self.ae_n_layers,
            noise_std=self.noise_std,
        ).to(device)

        self.ensemble_state_dicts = []
        self.ensemble_outputs = []
        self.trained = False
        self.rotation_matrices = []

        self.pos_edges = None
        self.neg_edges = None

        self._geom_cfg = None
        self._geom_default_schedule = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_model_parameters(self):
        def weight_reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.model.apply(weight_reset)
        if hasattr(self.model, "prior_log_scale"):
            torch.nn.init.constant_(self.model.prior_log_scale, 0.541)

    def _get_numpy_arrays(self):
        x_np = (
            self.original_features.detach().cpu().numpy()
            if isinstance(self.original_features, torch.Tensor)
            else np.asarray(self.original_features)
        )

        if not self.has_batch:
            return x_np, None

        batch_index_np = (
            self.batch_index.detach().cpu().numpy().squeeze()
            if isinstance(self.batch_index, torch.Tensor)
            else np.asarray(self.batch_index).squeeze()
        )
        return x_np, batch_index_np

    def _resolve_decode_condition(self, batch_index, ref_batch):
        """
        Single point of truth for turning (batch_index, ref_batch) user
        arguments into the batch_index tensor actually passed to
        ``self.model.decode``.

        Centralising this here means ``get_distribution_params`` (and any
        future method that needs to decode under a condition) does not
        re-implement the has_batch / ref_batch / batch_index trichotomy
        itself -- it just calls this and gets back either ``None``
        (no-batch model) or a ``(N,)`` LongTensor.

        Rules
        -----
        - No-batch model (``self.has_batch is False``):
            ``batch_index`` and ``ref_batch`` must both be left as ``None``.
            Passing either raises, rather than being silently ignored --
            a silently-ignored ref_batch is a much worse failure mode than
            an explicit error, since the user would otherwise believe their
            argument had an effect.
        - Batch-aware model:
            exactly one of ``batch_index`` / ``ref_batch`` must be given.
            ``batch_index`` takes priority if both are somehow provided.
        """
        if not self.has_batch:
            if batch_index is not None or ref_batch is not None:
                raise RuntimeError(
                    "This GLANCE instance was constructed without a batch "
                    "column (label_col=None), so there is no batch condition "
                    "to decode under. ref_batch/batch_index are not applicable "
                    "and must be left as None."
                )
            return None

        if batch_index is not None:
            return batch_index.to(self.device)

        if ref_batch is not None:
            n_cells = self.original_features.shape[0]
            return torch.full(
                (n_cells,),
                int(ref_batch),
                dtype=torch.long,
                device=self.device,
            )

        raise RuntimeError(
            "A decode condition must be specified. "
            "Provide either ref_batch (int) or batch_index (Tensor of shape (N,)). "
            "To select a principled reference batch automatically, "
            "call select_reference_batch() first."
        )

    # ------------------------------------------------------------------
    # Graph preparation
    # ------------------------------------------------------------------

    def prepare_local_manifold(
        self,
        k_pos: int = 10,
        n_neg: int = 20,
        far_quantile: float = 0.7,
        random_state: Union[None, int, np.random.Generator] = None,
        thread_limits: int = -1,
    ):
        """
        Pre-compute within-batch attractive and repulsive edges on the
        input feature space for the ``manifold_ctr`` regulariser.

        Parameters
        ----------
        k_pos : int
            Number of within-batch positive neighbours per cell.
        n_neg : int
            Number of within-batch negative samples per cell.
        far_quantile : float
            Negatives are sampled from points farther than this quantile
            within each batch.
        random_state : None, int, or np.random.Generator
            Controls randomness of negative sampling. If ``None``,
            falls back to the ``random_state`` set at model construction.
            If that is also ``None``, sampling is fully random.
        thread_limits : int
            Number of threads for cKDTree queries. -1 = use all cores.
        """
        rs = random_state if random_state is not None else self.random_state
        x_np, batch_index_np = self._get_numpy_arrays()

        # No-batch mode: treat all cells as one batch, reducing to a global
        # manifold constraint. This is the one place a synthetic all-zero
        # batch vector is still needed, because build_same_batch_edges
        # itself is batch-shaped; it is intentionally local to this method
        # rather than baked into _get_numpy_arrays, so that other callers
        # of _get_numpy_arrays still get a clean None for "no batch".
        if batch_index_np is None:
            batch_index_np = np.zeros(x_np.shape[0], dtype=np.int64)

        pos_edges, neg_edges = build_same_batch_edges(
            x=x_np,
            batch_index=batch_index_np,
            k_pos=k_pos,
            n_neg=n_neg,
            far_quantile=far_quantile,
            random_state=rs,
            thread_limits=thread_limits,
        )
        self.pos_edges = pos_edges.to(self.device)
        self.neg_edges = neg_edges.to(self.device)

        logger.info(
            "[GLANCE] Local manifold graph built | "
            "cells=%d, batches=%d, k_pos=%d, n_neg=%d, far_quantile=%.2f | "
            "pos_edges=%d, neg_edges=%d",
            x_np.shape[0],
            len(np.unique(batch_index_np)),
            k_pos,
            n_neg,
            far_quantile,
            self.pos_edges.shape[0],
            self.neg_edges.shape[0],
        )

    # ------------------------------------------------------------------
    # In-loop geometric refinement
    # ------------------------------------------------------------------

    def prepare_geom_refinement(
        self,
        start_epoch: int = 2000,
        k_pos: int = 10,
        n_neg: int = 20,
        far_quantile: float = 0.95,
        mutual: bool = True,
        rebuild_every: Union[int, None] = None,
        max_edges: int = 50000,
        thread_limits: int = -1,
        random_state: Union[None, int, np.random.Generator] = None,
    ):
        """
        Enable in-loop geometric refinement (``geom_reg``) starting at
        ``start_epoch``.

        Uses K-RNN positive edges and log-repulsive negative edges on
        ``q_m``. The log-repulsive form ``-log(d+eps)`` has a non-zero
        gradient at every distance, continuously pushing isolated clusters
        away from large ones. K-RNN filtering protects rare clusters from
        one-sided attractive edges toward large clusters.

        Not calling this method leaves training as a plain cVAE
        (plus ``manifold_ctr`` if ``prepare_local_manifold`` was called).

        Parameters
        ----------
        start_epoch : int
            Default 2000. Use with ``num_epochs=2500`` so cVAE converges
            fully before geometric refinement begins.
        far_quantile : float
            Default 0.95. Negatives drawn only from the extreme far tail,
            reducing the chance of large-cluster internal pairs being
            sampled as negatives and pulled apart.
        random_state : None, int, or np.random.Generator
            Controls the negative-sampling RNG. If ``None``, falls back
            to the ``random_state`` set at model construction. If that
            is also ``None``, sampling is fully random.
        k_pos, n_neg, mutual, rebuild_every, max_edges, thread_limits
            See ``build_cross_batch_geom_edges``.
        """
        rs = random_state if random_state is not None else self.random_state

        self._geom_cfg = {
            "k_pos": int(k_pos),
            "n_neg": int(n_neg),
            "far_quantile": float(far_quantile),
            "mutual": bool(mutual),
            "rebuild_every": int(rebuild_every) if rebuild_every is not None else None,
            "max_edges": int(max_edges),
            "thread_limits": int(thread_limits),
            "random_state": rs,
        }
        self._geom_default_schedule = {
            "geom_reg": {
                **copy.deepcopy(DEFAULT_LOSS_SCHEDULE["geom_reg"]),
                "start_epoch": int(start_epoch),
            }
        }

        logger.info(
            "[GLANCE] Geometric refinement registered | "
            "start_epoch=%d, k_pos=%d, n_neg=%d, "
            "far_quantile=%.2f, mutual=%s, rebuild_every=%s",
            start_epoch,
            k_pos,
            n_neg,
            far_quantile,
            mutual,
            rebuild_every,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        num_epochs=2500,
        loss_schedule: Union[Dict, None] = None,
        ctr_margin: float = 10.0,
        max_ctr_edges: int = 20000,
        log_details: bool = False,
        log_step: int = 10,
        tqdm_ncols: int = 70,
    ):
        self.ensemble_state_dicts = []
        self.ensemble_outputs = []
        self.rotation_matrices = []

        for i in range(self.ensemble_size):
            tqdm_desc = (
                f"Ensemble: {i + 1}/{self.ensemble_size}"
                if self.ensemble_size > 1
                else "Training VAE"
            )
            self._reset_model_parameters()

            train_gvae(
                model=self.model,
                features=self.original_features,
                batch_index=self.batch_index,
                pos_edges=self.pos_edges,
                neg_edges=self.neg_edges,
                ctr_margin=ctr_margin,
                max_ctr_edges=max_ctr_edges,
                geom_cfg=self._geom_cfg,
                geom_default_schedule=self._geom_default_schedule,
                num_epochs=num_epochs,
                lr=self.lr,
                loss_schedule=loss_schedule,
                log_details=log_details,
                log_step=log_step,
                tqdm_desc=tqdm_desc,
                tqdm_ncols=tqdm_ncols,
            )

            self.ensemble_state_dicts.append(
                {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            )

        self.trained = True

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, key_added=None, return_output=False):
        if not self.trained or not self.ensemble_state_dicts:
            raise RuntimeError("Model not trained yet")

        self.ensemble_outputs = []
        raw_latents = []
        self.model.eval()

        for state_dict in self.ensemble_state_dicts:
            self.model.load_state_dict(state_dict)
            with torch.no_grad():
                outputs = self.model(
                    features=self.original_features,
                    batch_index=self.batch_index,
                )
            self.ensemble_outputs.append(
                {
                    "q_m": outputs["q_m"].detach().cpu(),
                    "z": outputs["z"].detach().cpu(),
                }
            )
            raw_latents.append(outputs["q_m"].detach().cpu().numpy())

        if len(raw_latents) == 1:
            final_z = raw_latents[0]
            self.rotation_matrices = [np.eye(self.n_latent)]
        else:
            ref_z = raw_latents[0]
            aligned = [ref_z]
            self.rotation_matrices = [np.eye(self.n_latent)]
            for z_i in raw_latents[1:]:
                R, _ = orthogonal_procrustes(z_i, ref_z)
                self.rotation_matrices.append(R)
                aligned.append(np.dot(z_i, R))
            final_z = np.mean(aligned, axis=0)

        if return_output:
            return final_z
        target_key = key_added if key_added is not None else self.key_added
        self.adata.obsm[target_key] = final_z

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        num_epochs=2500,
        loss_schedule: Union[Dict, None] = None,
        ctr_margin: float = 10.0,
        max_ctr_edges: int = 20000,
        key_added=None,
        return_output=False,
        log_details: bool = False,
        log_step: int = 10,
        tqdm_ncols: int = 70,
    ):
        self.fit(
            num_epochs=num_epochs,
            loss_schedule=loss_schedule,
            ctr_margin=ctr_margin,
            max_ctr_edges=max_ctr_edges,
            log_details=log_details,
            log_step=log_step,
            tqdm_ncols=tqdm_ncols,
        )
        return self.transform(key_added=key_added, return_output=return_output)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        if not self.trained:
            warnings.warn(
                "GLANCE has not been trained yet. Saving empty state",
                category=UserWarning,
                stacklevel=2,
            )

        torch.save(
            {
                "ensemble_state_dicts": self.ensemble_state_dicts,
                "trained": self.trained,
                "ensemble_size": self.ensemble_size,
                "has_batch": self.has_batch,
                "hyper_parameters": {
                    "lr": self.lr,
                    "n_in": self.n_in,
                    "n_latent": self.n_latent,
                    "n_batch_classes": self.n_batch_classes,
                    "n_batch_embed": self.n_batch_embed,
                    "ae_n_hidden": self.ae_n_hidden,
                    "ae_n_layers": self.ae_n_layers,
                    "noise_std": self.noise_std,
                    "random_state": self.random_state,
                },
            },
            path,
        )
        logger.info("[GLANCE] Model saved to: %s", path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")

        if hasattr(self, "model"):
            self.model.cpu()
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for key, val in checkpoint["hyper_parameters"].items():
            setattr(self, key, val)

        # has_batch may be absent in checkpoints saved before this field
        # was introduced. Default to False rather than True: a missing
        # field most often means "this attribute didn't exist yet", and
        # defaulting to no-batch fails closed -- at worst you lose access
        # to ref_batch/select_reference_batch and have to retrain, rather
        # than silently requiring a ref_batch argument that doesn't apply.
        self.has_batch = checkpoint.get("has_batch", False)

        self.model = GaussianVAE(
            n_in=self.n_in,
            n_latent=self.n_latent,
            n_batch_classes=self.n_batch_classes,
            n_batch_embed=self.n_batch_embed,
            ae_n_hidden=self.ae_n_hidden,
            ae_n_layers=self.ae_n_layers,
            noise_std=getattr(self, "noise_std", 0.0),
        ).to(self.device)

        self.ensemble_state_dicts = checkpoint["ensemble_state_dicts"]
        self.trained = checkpoint["trained"]
        self.ensemble_size = checkpoint.get(
            "ensemble_size", len(self.ensemble_state_dicts)
        )
        self.ensemble_outputs = []
        self.rotation_matrices = []
        self.pos_edges = None
        self.neg_edges = None
        self._geom_cfg = None
        self._geom_default_schedule = None

        # random_state may be absent in checkpoints saved before this
        # field was introduced; fall back to None in that case.
        if not hasattr(self, "random_state"):
            self.random_state = None

        logger.info(
            "[GLANCE] Model loaded | "
            "n_latent=%d, n_layers=%d, noise_std=%.2f, "
            "has_batch=%s, random_state=%s",
            self.n_latent,
            self.ae_n_layers,
            self.noise_std,
            self.has_batch,
            self.random_state,
        )

    # ------------------------------------------------------------------
    # distribution parameters / abundance scores
    # ------------------------------------------------------------------

    def get_distribution_params(
        self,
        batch_index=None,
        ref_batch=None,
        use_mean=True,
        return_numpy=True,
    ):
        """
        Decode each ensemble model under a specified batch condition and
        aggregate the K Student-T distributions into a mixture, returning
        its sufficient statistics.

        In no-batch mode (model constructed with ``label_col=None``),
        there is exactly one decode condition -- the model itself --
        and ``batch_index``/``ref_batch`` must both be left as ``None``.
        See ``select_reference_batch`` for the batch-aware case.

        Parameters
        ----------
        batch_index : torch.Tensor of shape (N,) or None
            Per-cell batch index (integer encoded). Takes priority over
            ref_batch. Must be None in no-batch mode.
        ref_batch : int or None
            If set, all cells are decoded under this single batch condition.
            Ignored when batch_index is provided. Must be None in no-batch
            mode.
        use_mean : bool
            True  = use posterior mean q_m as the latent (deterministic,
                    recommended).
            False = use sampled z.
        return_numpy : bool
            True  = return numpy arrays.
            False = return torch.Tensors.

        Returns
        -------
        agg_loc : ndarray or Tensor, shape (N, G)
            Mixture mean (arithmetic mean of loc across K models).
        agg_var : ndarray or Tensor, shape (N, G)
            Mixture variance via the law of total variance:
                E[Var] + Var[E]
            Within-model variance = df / (df - 2) * scale^2  (df clamped > 2).
        per_run : list of dict
            Raw distribution parameters for each model run.
            Each dict contains:
                "loc"   : Tensor (N, G)
                "scale" : Tensor (N, G)
                "df"    : Tensor (N, G)  -- already softplus-transformed (from base)

        See Also
        --------
        select_reference_batch : choose a principled ref_batch from the data.
        compute_abundance_scores : compute per-cell abundance scores directly.
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        if not hasattr(self, "ensemble_outputs") or len(self.ensemble_outputs) == 0:
            raise RuntimeError("ensemble_outputs is empty. Run transform() first.")
        if len(self.ensemble_outputs) != len(self.ensemble_state_dicts):
            raise RuntimeError(
                f"ensemble_outputs length ({len(self.ensemble_outputs)}) "
                f"!= ensemble_state_dicts length ({len(self.ensemble_state_dicts)})."
            )

        # Single point of truth for the no-batch / ref_batch / batch_index
        # trichotomy -- see _resolve_decode_condition's docstring for why
        # this is centralised rather than re-checked here.
        batch_index_decode = self._resolve_decode_condition(batch_index, ref_batch)

        # Decode each ensemble model independently
        self.model.eval()
        per_run = []

        for i, state_dict in enumerate(self.ensemble_state_dicts):
            self.model.load_state_dict(state_dict)

            latent = self.ensemble_outputs[i]["q_m" if use_mean else "z"]
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent).float().to(self.device)
            elif isinstance(latent, torch.Tensor):
                latent = latent.to(self.device)
            else:
                raise TypeError(
                    f"Unsupported latent type in ensemble_outputs[{i}]: {type(latent)}"
                )

            with torch.no_grad():
                # GaussianVAE.decode() must return (loc, scale, df).
                # batch_index_decode is None in no-batch mode, which is the
                # same convention GaussianVAE.decode/forward already use
                # for "no batch_embedding" -- so no further branching is
                # needed here.
                loc, scale, df = self.model.decode(
                    latent, batch_index=batch_index_decode
                )

            # df from StudentTDecoder.forward() is softplus-transformed, shape (G,).
            # Broadcast to (N, G) for downstream matrix operations.
            if df.dim() == 1:
                df = df.unsqueeze(0).expand(loc.shape[0], -1).contiguous()

            per_run.append(
                {
                    "loc": loc.cpu(),
                    "scale": scale.cpu(),
                    "df": df.cpu(),
                }
            )

        # Aggregate into a mixture distribution
        locs = torch.stack([r["loc"] for r in per_run], dim=0)  # (K, N, G)
        scales = torch.stack([r["scale"] for r in per_run], dim=0)  # (K, N, G)
        dfs = torch.stack([r["df"] for r in per_run], dim=0)  # (K, N, G)

        agg_loc = locs.mean(dim=0)  # (N, G)

        # Law of total variance: E[Var] + Var[E]
        # Student-T variance = df / (df - 2) * scale^2, requires df > 2
        df_safe = dfs.clamp(min=2.01)
        within_var = (df_safe / (df_safe - 2.0)) * scales.pow(2)  # (K, N, G)
        between_var = (locs - agg_loc.unsqueeze(0)).pow(2)  # (K, N, G)
        agg_var = within_var.mean(dim=0) + between_var.mean(dim=0)  # (N, G)

        if return_numpy:
            return agg_loc.numpy(), agg_var.numpy(), per_run
        return agg_loc, agg_var, per_run

    def select_reference_batch(
        self,
        cell_type_col=None,
    ):
        """
        Select the reference batch based on data coverage in adata.obs.

        Only meaningful in batch-aware mode. In no-batch mode there is a
        single decode condition by construction, so this raises rather
        than returning a vacuous answer.

        The batch column is taken from self.batch_encoder.label_col (set at
        model construction).  Selection priority:

        1. Cell type diversity (number of unique cell types) -- primary criterion.
            A batch that covers more cell types provides a more representative
            reference condition; the decoder has seen those cell types under
            that condition during training.
        2. Cell count -- tiebreaker when two batches have equal diversity.

        This criterion operates entirely on adata.obs and does not touch the
        model, avoiding any circularity.

        Parameters
        ----------
        cell_type_col : str or None
            Column in adata.obs containing cell type labels.
            If None, selection falls back to cell count only.

        Returns
        -------
        best_batch_idx : int
            Integer-encoded batch index of the selected reference batch.
            Pass directly to get_distribution_params() or
            compute_abundance_scores() as ref_batch.
        summary : dict  {int -> {"label": str, "n_cells": int, "n_cell_types": int}}
            Coverage statistics for every batch, for user inspection.

        Examples
        --------
        >>> best, summary = glance.select_reference_batch(cell_type_col="cell_type")
        >>> print(summary)
        >>> scores, tau, _ = glance.compute_abundance_scores(ref_batch=best)

        Notes
        -----
        When prior experimental knowledge is available (e.g. one preparation
        method is known to yield higher protein detection rates), passing
        ref_batch directly to compute_abundance_scores() is preferred over
        automatic selection.
        """
        if not self.has_batch:
            raise RuntimeError(
                "This GLANCE instance was constructed without a batch column "
                "(label_col=None), so there is no reference batch to select. "
                "Call get_distribution_params()/compute_abundance_scores() "
                "directly with no ref_batch/batch_index argument instead."
            )

        batch_col = self.batch_encoder.label_col
        if batch_col not in self.adata.obs.columns:
            raise RuntimeError(
                f"Batch column '{batch_col}' not found in adata.obs. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )
        if cell_type_col is not None and cell_type_col not in self.adata.obs.columns:
            raise RuntimeError(
                f"Cell type column '{cell_type_col}' not found in adata.obs. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        obs = self.adata.obs
        batch_col_values = obs[batch_col]

        # Build summary over unique batch labels
        summary = {}
        for label in batch_col_values.unique():
            mask = batch_col_values == label
            n_cells = int(mask.sum())
            n_types = (
                int(obs.loc[mask, cell_type_col].nunique())
                if cell_type_col is not None
                else 0
            )
            # Retrieve the integer index from the fitted encoder's label2id mapping
            b_idx = int(self.batch_encoder.label2id[label])
            summary[b_idx] = {
                "label": str(label),
                "n_cells": n_cells,
                "n_cell_types": n_types,
            }

        # Select: cell type diversity first, cell count as tiebreaker
        best_batch_idx = max(
            summary,
            key=lambda b: (summary[b]["n_cell_types"], summary[b]["n_cells"]),
        )

        logger.info(
            "[GLANCE] select_reference_batch | cell_type_col=%s | summary: %s | "
            "selected: %d (%s)",
            cell_type_col,
            {
                b: f"label={v['label']}, n_cells={v['n_cells']}, "
                f"n_cell_types={v['n_cell_types']}"
                for b, v in summary.items()
            },
            best_batch_idx,
            summary[best_batch_idx]["label"],
        )

        return best_batch_idx, summary

    def compute_abundance_scores(
        self,
        batch_index=None,
        ref_batch=None,
        use_mean=True,
        tau_method="median",  # "median" | "bimodal" | "custom"
        store_key=None,
        return_numpy=True,
        custom_tau=None,
    ):
        """
        Compute a per-cell per-protein abundance score defined as:

            P(x > tau) = 1 - CDF_{StudentT}(tau; loc, scale, df)

        Scores are computed separately for each ensemble model and averaged,
        approximating the CDF of the mixture distribution.

        When averaged over a cell population, the abundance score equals the
        percent positivity of that population for each protein.

        In batch-aware mode, a decode condition must be specified via
        ref_batch or batch_index (see select_reference_batch for an
        automatic, principled choice). In no-batch mode, both must be
        left as None -- see get_distribution_params.

        Parameters
        ----------
        batch_index : torch.Tensor of shape (N,) or None
            Per-cell batch index. See get_distribution_params.
        ref_batch : int or None
            Reference batch condition. See get_distribution_params.
        use_mean : bool
            True = use posterior mean q_m as the latent (recommended).
        tau_method : str
            Strategy for determining the abundance threshold per protein.
            - "median"  : column-wise median of agg_loc (default, robust,
                        recommended for most use cases).
            - "bimodal" : valley between the two highest-density peaks of the
                        agg_loc distribution. More interpretable for proteins
                        with clear on/off expression patterns. Falls back to
                        median for unimodal distributions.
            - "custom"  : use the array provided via custom_tau. Raises an
                        error if custom_tau is None.
        store_key : str or None
            If provided, stores the abundance score matrix in
            adata.layers[store_key].  Columns correspond to adata.var_names,
            allowing direct use with scanpy plotting functions via the
            ``layer`` parameter.
        return_numpy : bool
            True = return numpy arrays; False = return torch.Tensors.
        custom_tau : array-like of shape (G,), scalar, or None
            Per-protein threshold values. Only used when tau_method="custom".
            Ignored for all other tau_method values.

        Returns
        -------
        abundance_scores : ndarray or Tensor, shape (N, G)
            Per-cell per-protein P(x > tau), in [0, 1].
        tau : ndarray or Tensor, shape (G,)
            Threshold values used.
        per_run : list of dict
            Raw distribution parameters from get_distribution_params, for
            further downstream analysis if needed.

        Notes
        -----
        Percent positivity for a cell group selected by group_mask:
            percent_pos = abundance_scores[group_mask].mean(axis=0)  # shape (G,)

        See Also
        --------
        select_reference_batch : choose ref_batch automatically from data coverage.
        get_distribution_params : return raw distribution parameters directly.
        """
        agg_loc, _, per_run = self.get_distribution_params(
            batch_index=batch_index,
            ref_batch=ref_batch,
            use_mean=use_mean,
            return_numpy=False,
        )

        # Determine threshold tau
        if tau_method == "median":
            tau_t = torch.as_tensor(
                np.median(agg_loc.numpy(), axis=0), dtype=agg_loc.dtype
            )
        elif tau_method == "bimodal":
            tau_np = find_bimodal_tau(agg_loc.numpy())
            tau_t = torch.as_tensor(tau_np, dtype=agg_loc.dtype)
        elif tau_method == "custom":
            if custom_tau is None:
                raise ValueError(
                    "tau_method='custom' requires custom_tau to be provided. "
                    "Pass an array of shape (G,) or a scalar."
                )
            tau_t = torch.as_tensor(custom_tau, dtype=agg_loc.dtype)
            if tau_t.dim() == 0:
                tau_t = tau_t.expand(agg_loc.shape[1])
        else:
            raise ValueError(
                f"Unknown tau_method: {tau_method!r}. "
                "Choose from 'median', 'bimodal', or 'custom'."
            )

        # Compute P(x > tau) per model, then average
        prob_list = []
        for r in per_run:
            loc = r["loc"]
            scale = r["scale"]
            df_safe = r["df"].clamp(min=2.01)
            tau_expanded = tau_t.unsqueeze(0).expand_as(loc)
            prob = 1.0 - studentt_cdf(tau_expanded, loc, scale, df_safe)
            prob_list.append(prob)

        abundance_scores = torch.stack(prob_list, dim=0).mean(dim=0)

        if store_key is not None:
            self.adata.layers[store_key] = (
                abundance_scores.numpy()
                if isinstance(abundance_scores, torch.Tensor)
                else abundance_scores
            )
            logger.info(
                "[GLANCE] Abundance scores stored in adata.layers['%s'] | shape: %s",
                store_key,
                self.adata.layers[store_key].shape,
            )

        if return_numpy:
            return abundance_scores.numpy(), tau_t.numpy(), per_run
        return abundance_scores, tau_t, per_run

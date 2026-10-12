import anndata as ad
import pandas as pd
import numpy as np
import torch
import json

from typing import Callable, Iterable, Literal, Union


class BioLabelEncoder:
    def __init__(
        self,
        label_col: str,
        sort: Union[None, bool, Callable] = None,
        mode: str = Literal["torch", "numpy"],
        device: str = "cpu",
    ):
        self.label_col = label_col
        self.sort = sort

        # 声明模式并校验
        if mode not in ["torch", "numpy"]:
            raise ValueError("mode must be 'torch' or 'numpy'")
        self.mode = mode

        # 仅在 torch 模式下保存 device
        self.device = device if mode == "torch" else None

        self.classes_ = None
        self.label2id = None
        self.id2label = None
        self.n_classes_ = None

    # =========================
    # FIT
    # =========================
    def fit(self, data: Union[ad.AnnData, Iterable]):

        if isinstance(data, ad.AnnData):
            labels = data.obs[self.label_col]
        else:
            labels = pd.Series(data)

        series = labels.astype("category")
        categories = list(series.cat.categories)

        # sorting
        if self.sort is True:
            categories = sorted(categories)
        elif callable(self.sort):
            categories = sorted(categories, key=self.sort)
        elif self.sort is None:
            pass
        else:
            raise ValueError("sort must be None, True or callable")

        self.classes_ = list(categories)
        self.n_classes_ = len(categories)

        self.label2id = {label: i for i, label in enumerate(self.classes_)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        return self

    # =========================
    # TRANSFORM
    # =========================
    def transform(self, data):

        if self.label2id is None:
            raise RuntimeError("Encoder not fitted yet.")

        if isinstance(data, ad.AnnData):
            labels = data.obs[self.label_col]
        else:
            labels = data

        ids = [self.label2id.get(x, -1) for x in labels]

        # 根据初始化的 mode 返回对应格式的数据
        if self.mode == "torch":
            return torch.tensor(ids, dtype=torch.long, device=self.device)
        else:
            return np.array(ids, dtype=np.int64)

    # =========================
    # FIT + TRANSFORM
    # =========================
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    # =========================
    # DECODE
    # =========================
    def inverse_transform(self, ids):

        if self.id2label is None:
            raise RuntimeError("Encoder not fitted yet.")

        # 兼容 torch tensor 或 numpy array 输入
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()

        return [self.id2label.get(int(i), None) for i in ids]

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path):

        state = {
            "label_col": self.label_col,
            "classes_": self.classes_,
            "mode": self.mode,
            "device": self.device,
        }

        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path, override_mode=None, override_device=None):

        with open(path) as f:
            state = json.load(f)

        # 允许在 load 时覆盖原本保存的 mode 和 device
        mode = (
            override_mode if override_mode is not None else state.get("mode", "torch")
        )
        device = (
            override_device
            if override_device is not None
            else state.get("device", "cpu")
        )

        obj = cls(label_col=state["label_col"], mode=mode, device=device)

        obj.classes_ = state["classes_"]
        obj.n_classes_ = len(obj.classes_)

        obj.label2id = {label: i for i, label in enumerate(obj.classes_)}
        obj.id2label = {i: label for label, i in obj.label2id.items()}

        return obj


# class BioLabelEncoder:
#     def __init__(
#         self,
#         label_col: str,
#         sort: Union[None, bool, Callable] = None,
#     ):
#         self.label_col = label_col
#         self.sort = sort

#         self.classes_ = None
#         self.label2id = None
#         self.id2label = None
#         self.n_classes_ = None

#     # =========================
#     # FIT
#     # =========================
#     def fit(self, data: Union[ad.AnnData, Iterable]):

#         if isinstance(data, ad.AnnData):
#             labels = data.obs[self.label_col]
#         else:
#             labels = pd.Series(data)

#         series = labels.astype("category")
#         categories = list(series.cat.categories)

#         # sorting
#         if self.sort is True:
#             categories = sorted(categories)
#         elif callable(self.sort):
#             categories = sorted(categories, key=self.sort)
#         elif self.sort is None:
#             pass
#         else:
#             raise ValueError("sort must be None, True or callable")

#         self.classes_ = list(categories)
#         self.n_classes_ = len(categories)

#         self.label2id = {label: i for i, label in enumerate(self.classes_)}
#         self.id2label = {i: label for label, i in self.label2id.items()}

#         return self

#     # =========================
#     # TRANSFORM
#     # =========================
#     def transform(self, data, device=None):

#         if self.label2id is None:
#             raise RuntimeError("Encoder not fitted yet.")

#         if isinstance(data, ad.AnnData):
#             labels = data.obs[self.label_col]
#         else:
#             labels = data

#         ids = [self.label2id.get(x, -1) for x in labels]
#         encoded = torch.tensor(ids, dtype=torch.long)

#         return encoded.to(device) if device is not None else encoded

#     # =========================
#     # FIT + TRANSFORM
#     # =========================
#     def fit_transform(self, data, device=None):
#         self.fit(data)
#         return self.transform(data, device)

#     # =========================
#     # DECODE
#     # =========================
#     def inverse_transform(self, ids):

#         if self.id2label is None:
#             raise RuntimeError("Encoder not fitted yet.")

#         return [self.id2label.get(int(i), None) for i in ids]

#     # =========================
#     # SAVE / LOAD
#     # =========================
#     def save(self, path):

#         state = {
#             "label_col": self.label_col,
#             "classes_": self.classes_,
#         }

#         with open(path, "w") as f:
#             json.dump(state, f)

#     @classmethod
#     def load(cls, path):

#         with open(path) as f:
#             state = json.load(f)

#         obj = cls(label_col=state["label_col"])

#         obj.classes_ = state["classes_"]
#         obj.n_classes_ = len(obj.classes_)

#         obj.label2id = {label: i for i, label in enumerate(obj.classes_)}
#         obj.id2label = {i: label for label, i in obj.label2id.items()}

#         return obj

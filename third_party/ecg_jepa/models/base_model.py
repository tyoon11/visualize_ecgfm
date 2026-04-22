"""
models/base_model.py  —  ABC + 모델 레지스트리

새 모델 추가:
  1. BaseModel 상속 + model_name 정의
  2. models/__init__.py 에 import 한 줄 추가
  → build_model('이름') 으로 즉시 사용 가능
"""

import abc
import torch.nn as nn

_REGISTRY: dict = {}


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list:
    return sorted(_REGISTRY.keys())


class BaseModel(nn.Module, metaclass=abc.ABCMeta):

    model_name: str = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.model_name is not None:
            _REGISTRY[cls.model_name] = cls

    @abc.abstractmethod
    def forward(self, x):
        """pretraining loss 반환.  x: (B, n_leads, T)"""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, x):
        """downstream feature 반환.  x: (B, n_leads, T) → (B, embed_dim)"""
        raise NotImplementedError

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
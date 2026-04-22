"""
models/__init__.py
모든 모델 모듈을 여기서 import → 레지스트리 자동 채움.
새 모델 추가 시 이 파일에 import 한 줄만 추가하면 됩니다.
"""
from .base_model import BaseModel, build_model, list_models  # noqa
from .ecg_jepa.model import ECGJepa                          # noqa

# 나중에 추가할 모델 예시:
# from models.st_mem.model   import STMEM   # noqa: F401
# from models.ecg_cpc.model  import ECGCPC  # noqa: F401
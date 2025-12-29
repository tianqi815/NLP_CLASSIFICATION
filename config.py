"""
NLP 意图识别服务配置
"""
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class IntentServiceConfig(BaseModel):
    """意图识别服务配置"""

    # 服务配置
    host: str = os.getenv("INTENT_SERVICE_HOST", "0.0.0.0")
    port: int = int(os.getenv("INTENT_SERVICE_PORT", "9001"))

    # 模型配置
    model_name: str = os.getenv(
        "INTENT_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model_cache_dir: str = os.getenv("INTENT_MODEL_CACHE_DIR", "./models_cache")

    # 分类配置
    confidence_threshold: float = float(
        os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.5")
    )
    max_length: int = int(os.getenv("INTENT_MAX_LENGTH", "512"))

    # 性能配置
    device: str = os.getenv("INTENT_DEVICE", "cpu")  # cpu, cuda, mps
    batch_size: int = int(os.getenv("INTENT_BATCH_SIZE", "8"))

    # 日志配置
    log_level: str = os.getenv("INTENT_LOG_LEVEL", "INFO")


# 全局配置实例
intent_config = IntentServiceConfig()


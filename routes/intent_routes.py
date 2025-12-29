"""
意图识别 API 路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from services.intent_service import IntentService
from config import intent_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intent", tags=["Intent"])

# 全局服务实例
_intent_service: Optional[IntentService] = None


def get_intent_service() -> IntentService:
    """获取意图识别服务实例（单例）"""
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentService(intent_config)
    return _intent_service


# ==================== 请求/响应模型 ====================

class ClassifyRequest(BaseModel):
    """分类请求"""
    question: str
    intent_type: str = "search"
    conversation_context: Optional[str] = None


class ClassifyResponse(BaseModel):
    """分类响应"""
    success: bool
    need_action: bool
    confidence: float
    reason: str
    query: str


class BatchClassifyRequest(BaseModel):
    """批量分类请求"""
    questions: List[str]
    intent_type: str = "search"


class BatchClassifyItem(BaseModel):
    """批量分类结果项"""
    question: str
    need_action: bool
    confidence: float
    reason: str
    query: str


class BatchClassifyResponse(BaseModel):
    """批量分类响应"""
    success: bool
    results: List[BatchClassifyItem]


# ==================== API 端点 ====================

@router.post("/classify", response_model=ClassifyResponse)
async def classify_intent(request: ClassifyRequest):
    """
    分类用户意图

    示例：
    ```json
    {
        "question": "今天天气怎么样？",
        "intent_type": "search",
        "conversation_context": null
    }
    ```
    """
    try:
        service = get_intent_service()
        result = service.classify(
            question=request.question,
            intent_type=request.intent_type,
            conversation_context=request.conversation_context
        )

        return ClassifyResponse(
            success=True,
            **result
        )
    except Exception as e:
        logger.error(f"意图分类失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"意图分类失败: {str(e)}"
        )


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_intent_batch(request: BatchClassifyRequest):
    """
    批量分类用户意图

    示例：
    ```json
    {
        "questions": [
            "今天天气怎么样？",
            "解释一下什么是机器学习"
        ],
        "intent_type": "search"
    }
    ```
    """
    try:
        service = get_intent_service()
        results = service.classify_batch(
            questions=request.questions,
            intent_type=request.intent_type
        )

        return BatchClassifyResponse(
            success=True,
            results=[
                BatchClassifyItem(**r) for r in results
            ]
        )
    except Exception as e:
        logger.error(f"批量分类失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"批量分类失败: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    健康检查

    示例：
    ```
    GET /api/v1/intent/health
    ```
    """
    try:
        service = get_intent_service()
        return {
            "status": "healthy",
            "service": "nlp-intent-classifier",
            "model": intent_config.model_name,
            "version": "1.0.0",
            "device": intent_config.device
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "nlp-intent-classifier",
            "error": str(e)
        }


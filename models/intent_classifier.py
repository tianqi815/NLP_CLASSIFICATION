"""
NLP 模型封装 - 意图分类器
"""
import logging
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class IntentClassifier:
    """意图分类器 - 使用零样本分类方法"""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        初始化意图分类器

        Args:
            model_name: 模型名称或路径
            device: 设备类型 (cpu, cuda, mps)
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")

            # 自动检测设备
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            # 加载 sentence-transformers 模型
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )

            logger.info(f"模型加载成功: {self.model_name}, 设备: {self.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"无法加载模型: {str(e)}")

    def predict(
        self,
        question: str,
        templates: List[str],
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        使用零样本分类方法预测意图

        Args:
            question: 用户问题
            templates: 意图模板列表（用于相似度计算）
            threshold: 置信度阈值

        Returns:
            预测结果字典，包含 need_action, confidence, reason
        """
        if not question or not question.strip():
            return {
                "need_action": False,
                "confidence": 0.0,
                "reason": "问题为空"
            }

        try:
            # 编码问题和模板
            question_embedding = self.model.encode(
                question,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            template_embeddings = self.model.encode(
                templates,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # 计算余弦相似度
            similarities = np.dot(template_embeddings, question_embedding)
            max_similarity = float(np.max(similarities))
            avg_similarity = float(np.mean(similarities))

            # 使用平均相似度和最大相似度的加权平均作为最终置信度
            confidence = (max_similarity * 0.7 + avg_similarity * 0.3)

            # 判断是否需要搜索
            need_action = confidence >= threshold

            # 生成原因
            if need_action:
                reason = (
                    f"问题与搜索意图模板相似度较高 "
                    f"(最大: {max_similarity:.2f}, 平均: {avg_similarity:.2f}, "
                    f"综合: {confidence:.2f})"
                )
            else:
                reason = (
                    f"问题与搜索意图模板相似度较低 "
                    f"(最大: {max_similarity:.2f}, 平均: {avg_similarity:.2f}, "
                    f"综合: {confidence:.2f})，不需要联网搜索"
                )

            return {
                "need_action": need_action,
                "confidence": confidence,
                "reason": reason
            }
        except Exception as e:
            logger.error(f"意图预测失败: {e}", exc_info=True)
            raise RuntimeError(f"意图预测失败: {str(e)}")

    def predict_batch(
        self,
        questions: List[str],
        templates: List[str],
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        批量预测意图

        Args:
            questions: 问题列表
            templates: 意图模板列表
            threshold: 置信度阈值

        Returns:
            预测结果列表
        """
        results = []
        for question in questions:
            try:
                result = self.predict(question, templates, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"批量预测失败 - 问题: {question}, 错误: {e}")
                results.append({
                    "need_action": False,
                    "confidence": 0.0,
                    "reason": f"预测失败: {str(e)}"
                })
        return results


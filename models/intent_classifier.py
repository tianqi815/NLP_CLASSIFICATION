"""
NLP 模型封装 - 意图分类器
"""
import logging
import os
from pathlib import Path
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

    def _find_local_model_path(self) -> Optional[str]:
        """
        查找本地缓存的模型路径
        
        Returns:
            本地模型路径，如果不存在则返回 None
        """
        if not self.cache_dir:
            return None
        
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return None
        
        # 构建模型缓存目录名
        model_cache_name = f"models--{self.model_name.replace('/', '--')}"
        model_cache_path = cache_path / model_cache_name
        
        if not model_cache_path.exists():
            logger.warning(f"模型缓存目录不存在: {model_cache_path}")
            return None
        
        # 查找快照目录
        snapshots_path = model_cache_path / "snapshots"
        if not snapshots_path.exists():
            logger.warning(f"快照目录不存在: {snapshots_path}")
            return None
        
        # 获取第一个快照（通常只有一个）
        snapshots = list(snapshots_path.iterdir())
        if not snapshots:
            logger.warning(f"未找到模型快照: {snapshots_path}")
            return None
        
        snapshot_path = snapshots[0]
        logger.info(f"找到本地模型快照: {snapshot_path}")
        return str(snapshot_path)

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

            # 尝试使用本地缓存的模型
            local_model_path = self._find_local_model_path()
            
            if local_model_path:
                logger.info(f"使用本地缓存的模型: {local_model_path}")
                try:
                    # 直接使用本地路径加载，避免网络请求
                    self.model = SentenceTransformer(
                        local_model_path,
                        device=self.device
                    )
                    logger.info(f"从本地路径加载模型成功")
                except Exception as e:
                    logger.warning(f"从本地路径加载失败: {e}")
                    logger.info(f"尝试使用模型名称加载（将使用缓存）: {self.model_name}")
                    # 如果本地路径加载失败，尝试使用模型名称（会使用缓存）
                    self.model = SentenceTransformer(
                        self.model_name,
                        cache_folder=self.cache_dir,
                        device=self.device
                    )
            else:
                logger.info(f"未找到本地缓存，尝试从缓存或网络加载: {self.model_name}")
                # 使用模型名称加载（会优先使用缓存，如果缓存不存在则从网络下载）
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
            # 优化：更依赖最大值（0.85:0.15），减少平均值的影响
            confidence = (max_similarity * 0.85 + avg_similarity * 0.15)

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


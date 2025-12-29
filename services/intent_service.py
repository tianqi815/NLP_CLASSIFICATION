"""
意图识别核心服务
"""
import logging
from typing import Dict, Any, List, Optional
from models.intent_classifier import IntentClassifier
from config import intent_config

logger = logging.getLogger(__name__)


class IntentService:
    """意图识别服务"""

    # 搜索意图模板 - 用于零样本分类
    SEARCH_INTENT_TEMPLATES = [
        "需要查询实时信息",
        "需要搜索最新数据",
        "需要联网获取信息",
        "查询当前时间相关信息",
        "获取最新动态",
        "搜索实时新闻",
        "查询天气信息",
        "查询股价价格",
        "查询最新事件"
    ]

    def __init__(self, config=None):
        """
        初始化意图识别服务

        Args:
            config: 配置对象，如果为 None 则使用全局配置
        """
        self.config = config or intent_config
        self.classifier = None
        self._load_classifier()

    def _load_classifier(self):
        """加载意图分类器"""
        try:
            logger.info("正在初始化意图分类器...")
            self.classifier = IntentClassifier(
                model_name=self.config.model_name,
                device=self.config.device,
                cache_dir=self.config.model_cache_dir
            )
            logger.info("意图分类器初始化成功")
        except Exception as e:
            logger.error(f"意图分类器初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"无法初始化意图分类器: {str(e)}")

    def classify(
        self,
        question: str,
        intent_type: str = "search",
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分类用户意图

        Args:
            question: 用户问题
            intent_type: 意图类型（默认 "search"）
            conversation_context: 对话上下文（预留）

        Returns:
            分类结果字典
        """
        if not question or not question.strip():
            return {
                "need_action": False,
                "confidence": 0.0,
                "reason": "问题为空",
                "query": ""
            }

        try:
            # 使用分类器进行预测
            result = self.classifier.predict(
                question=question,
                templates=self.SEARCH_INTENT_TEMPLATES,
                threshold=self.config.confidence_threshold
            )

            # 提取查询关键词（如果需要搜索）
            query = self._extract_query(question) if result["need_action"] else ""

            # 增强原因说明
            reason = self._enhance_reason(question, result)

            return {
                "need_action": result["need_action"],
                "confidence": result["confidence"],
                "reason": reason,
                "query": query
            }
        except Exception as e:
            logger.error(f"意图分类失败: {e}", exc_info=True)
            raise RuntimeError(f"意图分类失败: {str(e)}")

    def _enhance_reason(self, question: str, result: Dict[str, Any]) -> str:
        """
        增强分类原因说明

        Args:
            question: 用户问题
            result: 分类结果

        Returns:
            增强后的原因说明
        """
        base_reason = result["reason"]
        confidence = result["confidence"]

        # 简单的关键词匹配用于增强说明
        time_keywords = ["今天", "现在", "当前", "最新", "最近", "实时"]
        realtime_keywords = ["天气", "股价", "价格", "新闻", "事件", "行情"]

        has_time = any(kw in question for kw in time_keywords)
        has_realtime = any(kw in question for kw in realtime_keywords)

        if has_time and has_realtime:
            return (
                f"问题包含实时信息查询需求（时间+实时数据）。"
                f"{base_reason}"
            )
        elif has_time:
            return f"问题包含时间相关查询。{base_reason}"
        elif has_realtime:
            return f"问题包含实时数据查询需求。{base_reason}"
        else:
            return base_reason

    def _extract_query(self, question: str) -> str:
        """
        从问题中提取搜索查询

        Args:
            question: 用户问题

        Returns:
            提取的查询字符串
        """
        # 简单的查询提取逻辑
        # 可以后续优化为使用 NER 或关键词提取
        return question.strip()

    def classify_batch(
        self,
        questions: List[str],
        intent_type: str = "search"
    ) -> List[Dict[str, Any]]:
        """
        批量分类

        Args:
            questions: 问题列表
            intent_type: 意图类型（默认 "search"）

        Returns:
            分类结果列表
        """
        results = []
        for question in questions:
            try:
                result = self.classify(question, intent_type)
                results.append({
                    "question": question,
                    **result
                })
            except Exception as e:
                logger.error(f"批量分类失败 - 问题: {question}, 错误: {e}")
                results.append({
                    "question": question,
                    "need_action": False,
                    "confidence": 0.0,
                    "reason": f"分类失败: {str(e)}",
                    "query": ""
                })
        return results


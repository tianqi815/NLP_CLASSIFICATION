"""
测试意图识别改善效果
"""
import sys
import logging
from services.intent_service import IntentService
from config import intent_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_question(service: IntentService, question: str, expected_action: bool):
    """
    测试单个问题
    
    Args:
        service: 意图识别服务
        question: 测试问题
        expected_action: 期望是否需要搜索
    """
    result = service.classify(question)
    need_action = result["need_action"]
    confidence = result["confidence"]
    reason = result["reason"]
    
    status = "✅" if need_action == expected_action else "❌"
    print(f"\n{status} 问题: {question}")
    print(f"   需要搜索: {need_action} (期望: {expected_action})")
    print(f"   置信度: {confidence:.3f}")
    print(f"   原因: {reason}")
    
    return need_action == expected_action


def main():
    """主测试函数"""
    print("=" * 60)
    print("意图识别改善效果测试")
    print("=" * 60)
    
    try:
        # 初始化服务
        logger.info("正在初始化意图识别服务...")
        service = IntentService(intent_config)
        logger.info("服务初始化成功")
        
        # 测试用例
        test_cases = [
            # 问题案例，期望是否需要搜索
            ("今天天气如何？", True),  # 原有正确案例
            ("今天会下雨吗？", True),  # 原有误判案例（应该改善）
            ("明天天气怎么样？", True),  # 新增模板测试
            ("现在几点了？", True),  # 时间相关测试
            ("会下雨吗？", True),  # 疑问句测试
            ("温度多少？", True),  # 天气相关测试
            ("解释一下什么是机器学习", False),  # 不需要搜索的知识性问题
            ("什么是Python？", False),  # 不需要搜索的知识性问题
            ("今天适合出门吗？", True),  # 天气相关判断
        ]
        
        print(f"\n测试配置:")
        print(f"  模型: {intent_config.model_name}")
        print(f"  置信度阈值: {intent_config.confidence_threshold}")
        print(f"  设备: {intent_config.device}")
        
        # 执行测试
        passed = 0
        total = len(test_cases)
        
        for question, expected in test_cases:
            if test_question(service, question, expected):
                passed += 1
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print(f"测试结果: {passed}/{total} 通过")
        print("=" * 60)
        
        if passed == total:
            print("🎉 所有测试通过！")
            return 0
        else:
            print(f"⚠️  有 {total - passed} 个测试未通过")
            return 1
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


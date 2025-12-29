# NLP 意图识别微服务

## 项目简介

这是一个基于预训练中文 NLP 模型的意图识别微服务，主要用于判断用户问题是否需要联网搜索。服务通过 HTTP API 提供接口，可以集成到各种应用系统中，帮助智能判断用户查询是否需要实时信息检索。

### 核心功能

- **意图分类**：判断用户问题是否需要联网搜索
- **置信度评估**：提供分类结果的置信度分数
- **批量处理**：支持批量问题分类
- **RESTful API**：提供标准的 HTTP 接口

## 技术实现

### 实现方法

本项目采用**零样本分类（Zero-shot Classification）**方法实现意图识别：

1. **语义相似度计算**：使用预训练的多语言语义嵌入模型将用户问题和意图模板转换为向量表示
2. **相似度匹配**：计算问题向量与多个搜索意图模板向量的余弦相似度
3. **综合判断**：基于最大相似度和平均相似度的加权组合，判断是否需要搜索
4. **阈值过滤**：通过置信度阈值（默认 0.6）决定最终分类结果

### 技术架构

```
用户请求 → FastAPI 路由 → 意图识别服务 → 模型分类器 → 返回结果
```

- **路由层** (`routes/`): 处理 HTTP 请求和响应
- **服务层** (`services/`): 业务逻辑处理
- **模型层** (`models/`): NLP 模型封装和推理

## 使用的模型

### 主模型

**sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**

- **模型类型**：多语言语义嵌入模型（Sentence Transformers）
- **支持语言**：支持包括中文在内的多种语言
- **模型特点**：
  - 基于 MiniLM 架构，模型体积小，推理速度快
  - 专门针对语义相似度任务优化
  - 支持零样本分类，无需额外训练即可使用
- **模型大小**：约 420MB
- **性能**：在 CPU 上单次推理时间约 50-200ms

### 意图模板

服务使用以下 9 个搜索意图模板进行相似度匹配：

- 需要查询实时信息
- 需要搜索最新数据
- 需要联网获取信息
- 查询当前时间相关信息
- 获取最新动态
- 搜索实时新闻
- 查询天气信息
- 查询股价价格
- 查询最新事件

## 项目结构

```
nlp_intent_service/
├── main.py                 # FastAPI 应用入口
├── config.py               # 配置管理（使用 Pydantic）
├── requirements.txt        # 依赖包列表
├── environment.yml         # Conda 环境配置文件
├── .env.example           # 环境变量示例
├── services/
│   ├── __init__.py
│   └── intent_service.py   # 意图识别核心服务
├── routes/
│   ├── __init__.py
│   └── intent_routes.py    # FastAPI 路由
└── models/
    ├── __init__.py
    └── intent_classifier.py # NLP 模型封装
```

## 快速开始

### 1. 环境准备

使用 Conda 创建环境：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate nlp_classification
```

### 2. 配置环境变量（可选）

```bash
# 复制环境变量示例文件
cp .env.example .env

# 根据需要修改 .env 文件
```

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:9001` 启动。

### 4. 访问 API 文档

打开浏览器访问：http://localhost:9001/docs

## API 接口

### 1. 意图分类接口

**端点**: `POST /api/v1/intent/classify`

**请求示例**:
```json
{
  "question": "今天天气怎么样？",
  "intent_type": "search",
  "conversation_context": null
}
```

**响应示例**:
```json
{
  "success": true,
  "need_action": true,
  "confidence": 0.61,
  "reason": "问题包含实时信息查询需求(时间+实时数据)。问题与搜索意图模板相似度较高(最大:0.74,平均:0.30,综合:0.61)",
  "query": "今天天气怎么样？"
}
```

### 2. 批量分类接口

**端点**: `POST /api/v1/intent/classify/batch`

**请求示例**:
```json
{
  "questions": [
    "今天天气怎么样？",
    "解释一下什么是机器学习"
  ],
  "intent_type": "search"
}
```

### 3. 健康检查接口

**端点**: `GET /api/v1/intent/health`

**响应示例**:
```json
{
  "status": "healthy",
  "service": "nlp-intent-classifier",
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "version": "1.0.0",
  "device": "cpu"
}
```

## 测试结果

根据实际测试，意图识别功能正常工作：

- ✅ **API 响应正常**：HTTP 200 状态码
- ✅ **分类准确**：能够正确识别需要搜索的实时信息查询
- ✅ **置信度合理**：返回的置信度分数在合理范围内（0.6-0.8）
- ✅ **原因说明清晰**：提供了详细的分类原因和相似度分析

**测试案例**：
- 问题："今天天气怎么样？"
- 结果：`need_action: true`, `confidence: 0.61`
- 原因：正确识别为实时信息查询需求

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `INTENT_SERVICE_HOST` | 服务绑定地址 | `0.0.0.0` |
| `INTENT_SERVICE_PORT` | 服务端口 | `9001` |
| `INTENT_MODEL_NAME` | 模型名称 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `INTENT_CONFIDENCE_THRESHOLD` | 置信度阈值 | `0.6` |
| `INTENT_DEVICE` | 运行设备 | `cpu` (可选: `cuda`, `mps`, `auto`) |
| `INTENT_LOG_LEVEL` | 日志级别 | `INFO` |

## 后续优化改善方案

### 1. 模型优化

#### 方案 A：模型微调
- **目标**：提高分类准确率
- **方法**：收集标注数据，在预训练模型基础上进行微调
- **优势**：针对特定领域优化，准确率可提升 10-20%
- **实施**：
  - 收集 1000+ 标注样本（需要搜索/不需要搜索）
  - 使用 `hfl/chinese-roberta-wwm-ext` 作为基础模型
  - 进行二分类微调训练

#### 方案 B：模型升级
- **目标**：使用更强大的模型
- **候选模型**：
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`（更准确但更慢）
  - `BAAI/bge-large-zh-v1.5`（中文优化模型）
- **权衡**：准确率 vs 推理速度

### 2. 算法优化

#### 改进相似度计算
- **当前**：使用最大相似度和平均相似度的加权平均
- **优化**：
  - 引入加权模板（不同模板设置不同权重）
  - 使用 Top-K 相似度聚合
  - 考虑问题长度归一化

#### 增强特征提取
- **关键词匹配**：结合规则方法，识别时间、实时数据等关键词
- **NER 集成**：使用命名实体识别提取时间、地点等信息
- **上下文理解**：利用 `conversation_context` 参数进行上下文分析

### 3. 性能优化

#### 批处理优化
- **当前**：批量请求逐个处理
- **优化**：实现真正的批处理推理，提升吞吐量 3-5 倍

#### 缓存机制
- **问题缓存**：缓存常见问题的分类结果
- **模型缓存**：优化模型加载和内存使用

#### GPU 加速
- 支持 CUDA 设备，推理速度可提升 5-10 倍
- 支持批处理推理，充分利用 GPU 并行能力

### 4. 功能扩展

#### 多意图类型支持
- 当前仅支持"搜索"意图
- 扩展支持：问答、任务执行、闲聊等意图类型

#### 查询提取优化
- **当前**：简单返回原始问题
- **优化**：
  - 使用关键词提取算法
  - 去除停用词和冗余信息
  - 生成优化的搜索查询

#### 对话上下文利用
- 实现多轮对话的上下文理解
- 考虑历史对话对当前意图的影响

### 5. 工程优化

#### 监控和日志
- 添加 Prometheus 指标导出
- 记录分类准确率、响应时间等关键指标
- 实现日志聚合和分析

#### 错误处理
- 增强异常处理机制
- 添加重试逻辑和降级策略
- 实现优雅的错误响应

#### 部署优化
- 支持 Docker 容器化部署
- 添加 Kubernetes 部署配置
- 实现健康检查和自动重启

### 6. 数据驱动优化

#### A/B 测试
- 对比不同模型和算法的效果
- 收集用户反馈数据
- 持续优化阈值和参数

#### 数据收集
- 记录所有分类请求和结果
- 分析误分类案例
- 构建高质量训练数据集

## 依赖项

主要依赖包：

- `fastapi>=0.104.1` - Web 框架
- `uvicorn>=0.24.0` - ASGI 服务器
- `sentence-transformers>=2.2.2` - 语义嵌入模型
- `torch>=2.0.0` - 深度学习框架
- `transformers>=4.30.0` - Hugging Face 模型库
- `pydantic>=2.5.0` - 数据验证
- `numpy>=1.24.0` - 数值计算

完整依赖列表请查看 `requirements.txt`。

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 Issue 反馈。


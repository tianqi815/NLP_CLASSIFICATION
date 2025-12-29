"""
NLP 意图识别服务主应用
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from config import intent_config
from routes.intent_routes import router

# 配置日志
logging.basicConfig(
    level=getattr(logging, intent_config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="NLP Intent Classification Service",
    description="基于预训练中文 NLP 模型的意图识别服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "NLP Intent Classification Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/intent/health"
    }


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("=" * 50)
    logger.info("🚀 NLP Intent Classification Service v1.0.0")
    # 显示正确的访问地址
    if intent_config.host == "0.0.0.0":
        access_host = "localhost"
        logger.info(f"📝 服务地址: http://{access_host}:{intent_config.port} 或 http://127.0.0.1:{intent_config.port}")
        logger.info(f"📚 API文档: http://{access_host}:{intent_config.port}/docs")
    else:
        logger.info(f"📝 服务地址: http://{intent_config.host}:{intent_config.port}")
        logger.info(f"📚 API文档: http://{intent_config.host}:{intent_config.port}/docs")
    logger.info(f"🤖 模型: {intent_config.model_name}")
    logger.info(f"💻 设备: {intent_config.device}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("服务正在关闭...")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=intent_config.host,
        port=intent_config.port,
        reload=False,
        log_level=intent_config.log_level.lower()
    )


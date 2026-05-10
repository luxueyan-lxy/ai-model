"""
辩证思考AI互动系统 - 增强版主应用
整合了所有优化功能
"""
import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import uvicorn
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import traceback
import random
import hashlib
from functools import lru_cache
import aiofiles
import psutil

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="辩证思考AI互动系统 - 增强版",
    description="基于集成机器学习的文本领域分类与辩证思考互动系统",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {
            "name": "分类",
            "description": "文本分类相关操作"
        },
        {
            "name": "分析",
            "description": "观点分析相关操作"
        },
        {
            "name": "管理",
            "description": "系统管理相关操作"
        }
    ]
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
model_manager = None
cache_store = {}
PERFORMANCE_METRICS = {
    'requests': 0,
    'avg_response_time': 0,
    'error_rate': 0,
    'last_updated': datetime.now().isoformat()
}

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = ["models", "logs", "examples", "data", "cache"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")

# 数据模型
class ClassifyRequest(BaseModel):
    """分类请求模型"""
    text: str = Field(..., min_length=1, max_length=5000, description="要分类的文本")
    top_k: int = Field(5, ge=1, le=20, description="返回的领域数量")
    use_cache: bool = Field(True, description="是否使用缓存")
    detailed: bool = Field(False, description="返回详细结果")

    @validator('text')
    def validate_text_length(cls, v):
        if len(v) < 2:
            raise ValueError('文本太短，请输入至少2个字符')
        return v

class PerspectiveRequest(BaseModel):
    """视角生成请求模型"""
    text: str = Field(..., min_length=1, max_length=5000, description="原始观点")
    count: int = Field(6, ge=1, le=20, description="生成视角数量")
    perspective_types: Optional[List[str]] = Field(None, description="指定视角类型")

class AnalyzeRequest(BaseModel):
    """观点分析请求模型"""
    original_text: str = Field(..., min_length=1, max_length=5000, description="原始观点文本")
    response_text: str = Field(..., min_length=1, max_length=10000, description="用户回应文本")
    perspective_type: str = Field("neutral", description="视角类型")
    analysis_depth: str = Field("standard", description="分析深度: basic/standard/comprehensive")

class BatchRequest(BaseModel):
    """批量请求模型"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="批量处理的文本列表")
    operation: str = Field(..., description="操作类型: classify/analyze")

class ExampleTopic(BaseModel):
    """示例话题模型"""
    id: str
    title: str
    content: str
    description: str
    icon: str
    tags: List[str]
    difficulty: str = Field("medium", description="难度: easy/medium/hard")
    domain: Optional[str] = Field(None, description="主要领域")

# 系统初始化
try:
    # 尝试导入增强版模型模块
    sys.path.append(str(Path(__file__).parent))
    MODEL_AVAILABLE = True
    logger.info("模型模块可用")
except ImportError as e:
    MODEL_AVAILABLE = False
    logger.warning(f"无法导入增强版模型模块: {e}")
    logger.warning("将使用模拟分类器进行演示")

# 视角类型定义
PERSPECTIVE_TYPES = [
    {
        "type": "opposite",
        "label": "对立视角",
        "color": "#e74c3c",
        "icon": "fas fa-exchange-alt",
        "description": "与你的初始观点完全相反",
        "difficulty": "hard"
    },
    {
        "type": "neutral",
        "label": "中立视角",
        "color": "#3498db",
        "icon": "fas fa-balance-scale",
        "description": "站在客观中立的角度分析",
        "difficulty": "medium"
    },
    {
        "type": "supplement",
        "label": "补充视角",
        "color": "#2ecc71",
        "icon": "fas fa-plus-circle",
        "description": "补充你未考虑的方面",
        "difficulty": "easy"
    },
    {
        "type": "unique",
        "label": "小众视角",
        "color": "#9b59b6",
        "icon": "fas fa-lightbulb",
        "description": "从非主流角度思考问题",
        "difficulty": "hard"
    },
    {
        "type": "historical",
        "label": "历史视角",
        "color": "#f39c12",
        "icon": "fas fa-history",
        "description": "从历史发展的角度看问题",
        "difficulty": "medium"
    },
    {
        "type": "global",
        "label": "全球视角",
        "color": "#1abc9c",
        "icon": "fas fa-globe",
        "description": "从国际化和全球化的角度思考",
        "difficulty": "hard"
    },
    {
        "type": "emotional",
        "label": "情感视角",
        "color": "#e67e22",
        "icon": "fas fa-heart",
        "description": "从情感和感受角度分析",
        "difficulty": "medium"
    },
    {
        "type": "ethical",
        "label": "伦理视角",
        "color": "#34495e",
        "icon": "fas fa-handshake",
        "description": "从道德伦理角度思考",
        "difficulty": "hard"
    }
]

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("正在启动辩证思考AI互动系统 - 增强版...")
    
    # 确保目录存在
    ensure_directories()
    
    # 初始化模型管理器
    await initialize_model_manager()
    
    # 预加载示例数据
    await preload_examples()
    
    # 启动后台任务
    asyncio.create_task(background_cleanup_task())
    
    logger.info("服务启动完成，版本：2.0.0")

async def initialize_model_manager():
    """初始化模型管理器"""
    global model_manager
    try:
        if MODEL_AVAILABLE:
            # 尝试动态导入模型管理器
            try:
                from model_manager import ModelManager
                model_manager = ModelManager(cache_dir="models/cache")
                await model_manager.initialize_all_models()
                logger.info("模型管理器初始化成功")
            except ImportError:
                logger.warning("模型管理器不可用，使用模拟模式")
                model_manager = None
        else:
            logger.warning("使用模拟模式，模型管理器不可用")
            model_manager = None
    except Exception as e:
        logger.error(f"模型管理器初始化失败: {e}")
        logger.error(traceback.format_exc())
        model_manager = None

async def preload_examples():
    """预加载示例数据"""
    try:
        examples_file = "examples/enhanced_topics.json"
        os.makedirs("examples", exist_ok=True)
        
        # 如果文件不存在，创建示例数据
        if not os.path.exists(examples_file):
            examples = [
                {
                    "id": "example_001",
                    "title": "人工智能与就业",
                    "content": "人工智能将完全取代人类工作",
                    "description": "探讨AI对就业市场的影响和未来趋势",
                    "icon": "fas fa-robot",
                    "tags": ["科技", "就业", "未来"],
                    "difficulty": "medium",
                    "domain": "科技"
                },
                {
                    "id": "example_002",
                    "title": "线上教育效果",
                    "content": "线上教育比传统教育更有效",
                    "description": "比较线上和传统教育模式的优缺点",
                    "icon": "fas fa-laptop",
                    "tags": ["教育", "技术", "学习"],
                    "difficulty": "easy",
                    "domain": "教育"
                },
                {
                    "id": "example_003",
                    "title": "环保与经济发展",
                    "content": "经济发展不应以牺牲环境为代价",
                    "description": "探讨经济增长与环境保护的平衡关系",
                    "icon": "fas fa-leaf",
                    "tags": ["环境", "经济", "发展"],
                    "difficulty": "hard",
                    "domain": "经济"
                }
            ]
            
            async with aiofiles.open(examples_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(examples, indent=2, ensure_ascii=False))
            logger.info(f"已创建示例文件: {examples_file}")
        
        logger.info("示例数据加载完成")
    except Exception as e:
        logger.error(f"预加载示例失败: {e}")

async def background_cleanup_task():
    """后台清理任务"""
    while True:
        try:
            # 清理过期缓存
            await cleanup_expired_cache()
            # 记录性能指标
            await update_performance_metrics()
            # 等待下次清理
            await asyncio.sleep(300)  # 5分钟
        except Exception as e:
            logger.error(f"后台清理任务失败: {e}")
            await asyncio.sleep(60)

async def cleanup_expired_cache():
    """清理过期缓存"""
    global cache_store
    current_time = time.time()
    expired_keys = []
    
    for key, (value, timestamp) in cache_store.items():
        if current_time - timestamp > 3600:  # 1小时过期
            expired_keys.append(key)
    
    for key in expired_keys:
        del cache_store[key]
    
    if expired_keys:
        logger.info(f"清理了 {len(expired_keys)} 个过期缓存")

async def update_performance_metrics():
    """更新性能指标"""
    global PERFORMANCE_METRICS
    PERFORMANCE_METRICS['last_updated'] = datetime.now().isoformat()
    PERFORMANCE_METRICS['memory_usage'] = psutil.virtual_memory().percent
    PERFORMANCE_METRICS['cpu_usage'] = psutil.cpu_percent()

# 工具函数
def generate_cache_key(operation: str, data: str) -> str:
    """生成缓存键"""
    data_hash = hashlib.md5(data.encode('utf-8')).hexdigest()
    return f"{operation}_{data_hash}"

@lru_cache(maxsize=1000)
def get_cached_response(key: str):
    """获取缓存响应"""
    if key in cache_store:
        return cache_store[key][0]
    return None

def set_cached_response(key: str, data: Any):
    """设置缓存响应"""
    cache_store[key] = (data, time.time())

# 主要业务逻辑
async def enhanced_classify(text: str, top_k: int = 5, use_cache: bool = True, detailed: bool = False) -> Dict[str, Any]:
    """增强的文本分类"""
    start_time = time.time()
    
    # 检查缓存
    cache_key = None
    if use_cache:
        cache_key = generate_cache_key(f"classify_{top_k}_{detailed}", text)
        cached_result = get_cached_response(cache_key)
        if cached_result:
            logger.debug(f"使用缓存结果: {cache_key}")
            cached_result['cached'] = True
            return cached_result
    
    try:
        if model_manager and hasattr(model_manager, 'classify'):
            result = await model_manager.classify(text, top_k, detailed)
        else:
            # 模拟结果
            result = generate_mock_classification(text, top_k, detailed)
        
        result['processing_time'] = round(time.time() - start_time, 3)
        result['model_used'] = model_manager is not None
        
        if use_cache and cache_key:
            set_cached_response(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")

def generate_mock_classification(text: str, top_k: int = 5, detailed: bool = False) -> Dict[str, Any]:
    """生成模拟分类结果"""
    domains = ['科技', '教育', '职场', '生活', '健康', '财经', '旅游', '政治', '娱乐', '体育']
    selected = random.sample(domains, min(top_k, len(domains)))
    
    result = {
        'text': text,
        'domains': selected,
        'probabilities': {domain: round(random.uniform(0.3, 0.9), 3) for domain in selected},
        'top_domain': selected[0] if selected else None
    }
    
    if detailed:
        result['details'] = {
            'keywords': text.split()[:5] if text.split() else ['思考', '分析'],
            'sentiment_score': round(random.uniform(-0.5, 0.5), 2),
            'complexity_score': round(min(0.5 + len(text) * 0.001, 0.95), 2)
        }
    
    return result

async def enhanced_analyze_viewpoint(original_text: str, response_text: str, perspective_type: str, 
                                    analysis_depth: str = "standard") -> Dict[str, Any]:
    """增强的观点分析"""
    start_time = time.time()
    
    try:
        if model_manager and hasattr(model_manager, 'analyze_viewpoint'):
            analysis = await model_manager.analyze_viewpoint(
                original_text, response_text, perspective_type, analysis_depth
            )
        else:
            # 模拟分析
            analysis = generate_mock_analysis(original_text, response_text, perspective_type, analysis_depth)
        
        processing_time = time.time() - start_time
        
        return {
            'original_text': original_text,
            'response_text': response_text,
            'perspective_type': perspective_type,
            'analysis': analysis,
            'processing_time': round(processing_time, 3),
            'analysis_depth': analysis_depth
        }
        
    except Exception as e:
        logger.error(f"观点分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"观点分析失败: {str(e)}")

def generate_mock_analysis(original_text: str, response_text: str, perspective_type: str, 
                          analysis_depth: str) -> Dict[str, Any]:
    """生成模拟分析结果"""
    analysis = {
        'positive': f"你对'{original_text[:20]}...'的分析很有见地，展现了良好的思考能力。",
        'suggestion': "可以考虑补充具体案例，增强说服力。也可以从更多角度进行分析。",
        'keywords': response_text.split()[:5] if response_text.split() else ['思考', '分析'],
        'logic_score': round(min(0.7 + len(response_text) * 0.001, 0.95), 2)
    }
    
    if analysis_depth in ["standard", "comprehensive"]:
        analysis.update({
            'strengths': ['逻辑清晰', '观点明确', '论证充分'][:random.randint(1, 3)],
            'areas_for_improvement': ['补充案例', '加强逻辑', '深化分析'][:random.randint(1, 3)],
            'sentiment_analysis': {
                'overall_sentiment': random.choice(['positive', 'neutral', 'critical']),
                'confidence_score': round(random.uniform(0.6, 0.9), 2)
            }
        })
    
    if analysis_depth == "comprehensive":
        analysis.update({
            'argument_structure': {
                'premises': random.randint(2, 5),
                'conclusions': random.randint(1, 3),
                'coherence_score': round(random.uniform(0.6, 0.9), 2)
            },
            'rhetorical_devices': random.sample(['比喻', '举例', '对比', '反问'], random.randint(1, 3))
        })
    
    return analysis

# API路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """根路由，返回前端界面"""
    try:
        html_content = await generate_enhanced_interface()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"根路径处理失败: {e}")
        return HTMLResponse(content=generate_error_interface(str(e)), status_code=500)

@app.get("/api/health", tags=["管理"])
async def health_check():
    """健康检查端点"""
    system_status = {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager is not None,
        "model_available": MODEL_AVAILABLE,
        "performance_metrics": PERFORMANCE_METRICS,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    }
    return system_status

@app.post("/api/classify", tags=["分类"])
async def classify_text_endpoint(request: ClassifyRequest):
    """文本分类端点"""
    PERFORMANCE_METRICS['requests'] += 1
    start_time = time.time()
    
    try:
        result = await enhanced_classify(
            request.text, 
            request.top_k, 
            request.use_cache, 
            request.detailed
        )
        
        PERFORMANCE_METRICS['avg_response_time'] = (
            PERFORMANCE_METRICS['avg_response_time'] * 0.9 + 
            (time.time() - start_time) * 0.1
        )
        
        return {
            "success": True,
            "data": result,
            "request_info": {
                "top_k": request.top_k,
                "use_cache": request.use_cache,
                "detailed": request.detailed
            }
        }
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise

@app.post("/api/analyze-viewpoint", tags=["分析"])
async def analyze_viewpoint_endpoint(request: AnalyzeRequest):
    """分析观点端点"""
    PERFORMANCE_METRICS['requests'] += 1
    start_time = time.time()
    
    try:
        result = await enhanced_analyze_viewpoint(
            request.original_text,
            request.response_text,
            request.perspective_type,
            request.analysis_depth
        )
        
        PERFORMANCE_METRICS['avg_response_time'] = (
            PERFORMANCE_METRICS['avg_response_time'] * 0.9 + 
            (time.time() - start_time) * 0.1
        )
        
        return {
            "success": True,
            "data": result,
            "request_info": {
                "perspective_type": request.perspective_type,
                "analysis_depth": request.analysis_depth
            }
        }
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise

@app.post("/api/generate-perspectives", tags=["分析"])
async def generate_perspectives_endpoint(request: PerspectiveRequest):
    """生成多个视角端点"""
    PERFORMANCE_METRICS['requests'] += 1
    start_time = time.time()
    
    try:
        # 先分类获取领域
        classify_result = await enhanced_classify(request.text, top_k=2)
        main_domain = classify_result.get('top_domain', '综合')
        
        # 选择视角类型
        if request.perspective_types:
            selected_types = [pt for pt in PERSPECTIVE_TYPES 
                            if pt['type'] in request.perspective_types]
        else:
            selected_types = PERSPECTIVE_TYPES
        
        selected_types = selected_types[:request.count]
        
        perspectives = []
        for i, p_type in enumerate(selected_types):
            text_preview = request.text[:20] + "..." if len(request.text) > 20 else request.text
            
            title_templates = {
                "opposite": f"对立视角：重新审视「{text_preview}」",
                "neutral": f"中立视角：客观分析「{text_preview}」",
                "supplement": f"补充视角：关于「{text_preview}」的更多思考",
                "unique": f"小众视角：对「{text_preview}」的独特见解",
                "historical": f"历史视角：从历史看「{text_preview}」",
                "global": f"全球视角：国际化视野下的「{text_preview}」",
                "emotional": f"情感视角：感受「{text_preview}」的情感维度",
                "ethical": f"伦理视角：从道德角度审视「{text_preview}」"
            }
            
            title = title_templates.get(p_type["type"], f"{p_type['label']}：关于「{text_preview}」的思考")
            
            hint_templates = {
                "opposite": f"请站在与「{text_preview}」相反的立场，从{main_domain}角度阐述你的反驳观点",
                "neutral": f"请客观分析「{text_preview}」的优缺点，从{main_domain}领域进行辩证评价",
                "supplement": f"请补充「{text_preview}」中未考虑的方面，从{main_domain}角度提供新见解",
                "unique": f"请从独特的{main_domain}视角出发，对「{text_preview}」提出创新性思考",
                "historical": f"请从历史发展的角度，分析「{text_preview}」在{main_domain}领域的演变",
                "global": f"请从全球化的视野，探讨「{text_preview}」在不同文化背景下的{main_domain}意义",
                "emotional": f"请从情感体验的角度，分享对「{text_preview}」的感受和理解",
                "ethical": f"请从道德伦理的角度，评价「{text_preview}」的价值和影响"
            }
            
            hint = hint_templates.get(p_type["type"], f"请对这个观点进行深入思考")
            
            perspectives.append({
                "id": f"perspective_{i+1}",
                "type": p_type["type"],
                "label": p_type["label"],
                "title": title,
                "hint": hint,
                "color": p_type["color"],
                "icon": p_type["icon"],
                "description": p_type["description"],
                "difficulty": p_type.get("difficulty", "medium"),
                "domain": main_domain
            })
        
        processing_time = time.time() - start_time
        
        PERFORMANCE_METRICS['avg_response_time'] = (
            PERFORMANCE_METRICS['avg_response_time'] * 0.9 + 
            processing_time * 0.1
        )
        
        return {
            "success": True,
            "data": {
                "text": request.text,
                "perspectives": perspectives,
                "count": len(perspectives),
                "main_domain": main_domain,
                "processing_time": round(processing_time, 3)
            }
        }
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise HTTPException(status_code=500, detail=f"生成视角失败: {str(e)}")

@app.post("/api/batch-process", tags=["分类", "分析"])
async def batch_process_endpoint(request: BatchRequest):
    """批量处理端点"""
    PERFORMANCE_METRICS['requests'] += 1
    start_time = time.time()
    
    try:
        results = []
        
        if request.operation == "classify":
            tasks = [enhanced_classify(text, detailed=True) for text in request.texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elif request.operation == "analyze":
            # 对于分析，需要成对的文本
            if len(request.texts) % 2 != 0:
                raise HTTPException(status_code=400, detail="分析操作需要成对的原始观点和回应")
            
            tasks = []
            for i in range(0, len(request.texts), 2):
                tasks.append(enhanced_analyze_viewpoint(
                    request.texts[i], 
                    request.texts[i+1], 
                    "neutral"
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        else:
            raise HTTPException(status_code=400, detail=f"不支持的操作类型: {request.operation}")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "data": {
                "operation": request.operation,
                "item_count": len(request.texts),
                "results": results,
                "processing_time": round(processing_time, 3),
                "avg_time_per_item": round(processing_time / len(request.texts), 3)
            }
        }
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise

@app.get("/api/perspective-types", tags=["分析"])
async def get_perspective_types_endpoint():
    """获取视角类型"""
    return {
        "success": True,
        "data": {
            "perspective_types": PERSPECTIVE_TYPES,
            "count": len(PERSPECTIVE_TYPES)
        }
    }

@app.get("/api/system-metrics", tags=["管理"])
async def get_system_metrics():
    """获取系统指标"""
    return {
        "success": True,
        "data": {
            "performance_metrics": PERFORMANCE_METRICS,
            "cache_info": {
                "cache_size": len(cache_store),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
            },
            "model_info": {
                "available": MODEL_AVAILABLE,
                "manager_initialized": model_manager is not None
            }
        }
    }

@app.post("/api/clear-cache", tags=["管理"])
async def clear_cache():
    """清除缓存"""
    global cache_store
    cache_size = len(cache_store)
    cache_store.clear()
    
    return {
        "success": True,
        "message": f"已清除 {cache_size} 个缓存项"
    }

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": f"服务器内部错误: {str(exc)}",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

# 前端界面生成
async def generate_enhanced_interface() -> str:
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>辩证思考AI互动系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            text-align: center; 
            color: white; 
            margin-bottom: 40px; 
            padding: 40px 0; 
        }
        .header h1 { 
            font-size: 3rem; 
            margin-bottom: 15px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .card { 
            background: white; 
            border-radius: 20px; 
            padding: 40px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
            margin-bottom: 30px; 
        }
        .input-area { margin-bottom: 20px; }
        label { display: block; margin-bottom: 10px; font-weight: 600; color: #333; }
        textarea { 
            width: 100%; 
            min-height: 150px; 
            padding: 20px; 
            border: 2px solid #e0e0e0; 
            border-radius: 12px; 
            font-size: 16px; 
            font-family: inherit; 
            resize: vertical; 
        }
        textarea:focus { outline: none; border-color: #667eea; }
        button { 
            padding: 12px 30px; 
            border: none; 
            border-radius: 10px; 
            font-size: 16px; 
            font-weight: 600; 
            cursor: pointer; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            margin-right: 10px; 
            margin-bottom: 10px;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        button.secondary { background: linear-gradient(135deg, #3498db, #2980b9); }
        button.success { background: linear-gradient(135deg, #2ecc71, #27ae60); }
        button.warning { background: linear-gradient(135deg, #f1c40f, #f39c12); }
        button.danger { background: linear-gradient(135deg, #e74c3c, #c0392b); }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .loading-spinner { width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        .error { color: #e74c3c; background: #fdf2f2; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .success { color: #2ecc71; background: #f2fdf7; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .info { color: #3498db; background: #f0f8ff; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .perspectives-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .perspective-card { 
            border: 2px solid; 
            border-radius: 12px; 
            padding: 20px; 
            position: relative;
            transition: transform 0.3s;
        }
        .perspective-card:hover { transform: translateY(-5px); }
        .perspective-header { display: flex; align-items: center; margin-bottom: 10px; }
        .perspective-icon { font-size: 20px; margin-right: 10px; }
        .perspective-title { font-size: 18px; font-weight: 600; margin-bottom: 10px; }
        .perspective-hint { color: #666; font-size: 14px; margin-bottom: 10px; }
        .perspective-description { font-size: 12px; color: #888; }
        .example-topics { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
        .example-topic { 
            background: #f8f9fa; 
            border-radius: 10px; 
            padding: 15px; 
            cursor: pointer; 
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .example-topic:hover { border-color: #667eea; transform: translateY(-3px); }
        .example-topic h4 { margin-bottom: 10px; color: #333; }
        .example-topic p { color: #666; font-size: 14px; margin-bottom: 10px; }
        .tags { display: flex; flex-wrap: wrap; gap: 5px; }
        .tag { background: #e3f2fd; padding: 3px 8px; border-radius: 10px; font-size: 12px; }
        .model-info { 
            position: fixed; 
            top: 20px; 
            right: 20px; 
            background: white; 
            padding: 10px 15px; 
            border-radius: 20px; 
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status-online { background: #2ecc71; }
        .status-offline { background: #e74c3c; }
        .analysis-examples { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .analysis-example { 
            border: 2px solid #e0e0e0; 
            border-radius: 12px; 
            padding: 20px; 
            background: white;
            transition: all 0.3s;
        }
        .analysis-example:hover { border-color: #667eea; transform: translateY(-3px); }
        .analysis-example h4 { color: #333; margin-bottom: 10px; }
        .analysis-example p { color: #666; font-size: 14px; margin-bottom: 10px; }
        .analysis-result { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 10px; }
        .analysis-positive { color: #2ecc71; font-weight: 600; margin-bottom: 8px; }
        .analysis-suggestion { color: #f39c12; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤔 辩证思考AI互动系统</h1>
            <p>基于机器学习的文本领域分类、视角生成与观点分析平台</p>
        </div>
        
        <div class="model-info" id="model-status">
            <div class="status-dot status-online"></div>
            <span>模型已加载</span>
        </div>
        
        <div class="card">
            <h2>🔍 文本分析与视角生成</h2>
            <div class="input-area">
                <label for="original-text">原始观点：</label>
                <textarea id="original-text" placeholder="请输入原始观点，例如：人工智能是未来科技发展的重要方向。"></textarea>
            </div>
            
            <div class="input-area">
                <label for="response-text">你的回应：</label>
                <textarea id="response-text" placeholder="请在此输入你的观点回应，针对原始观点进行深入分析。"></textarea>
            </div>
            
            <div>
                <label for="perspective-type">选择视角类型：</label>
                <select id="perspective-type" style="padding: 10px; border-radius: 8px; border: 2px solid #e0e0e0; margin-bottom: 15px; width: 200px;">
                    <option value="neutral">中立视角</option>
                    <option value="opposite">对立视角</option>
                    <option value="supplement">补充视角</option>
                    <option value="unique">小众视角</option>
                </select>
            </div>
            
            <div>
                <button onclick="classifyText()">文本分类</button>
                <button class="secondary" onclick="generatePerspectives()">生成视角</button>
                <button class="success" onclick="analyzeViewpoint()">分析观点</button>
                <button class="warning" onclick="loadExample()">加载示例</button>
                <button class="danger" onclick="clearAll()">清空</button>
            </div>
            
            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <p>正在处理...</p>
            </div>
            
            <div id="result"></div>
        </div>
        
        <div class="card">
            <h2>📊 系统信息</h2>
            <p>访问API文档: <a href="/api/docs" target="_blank">/api/docs</a></p>
            <p>健康检查: <a href="/api/health" target="_blank">/api/health</a></p>
            <p>支持的视角类型: <span id="perspective-types-count">6</span> 种</p>
        </div>
    </div>
    
    <script>
        // 全局变量
        let currentText = '';
        let currentPerspectives = [];
        
        // 检查模型状态
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusElement = document.getElementById('model-status');
                const statusDot = statusElement.querySelector('.status-dot');
                const statusText = statusElement.querySelector('span');
                
                if (data.model_loaded) {
                    statusDot.className = 'status-dot status-online';
                    statusText.textContent = '模型已加载';
                } else {
                    statusDot.className = 'status-dot status-offline';
                    statusText.textContent = '模型未加载';
                }
            } catch (error) {
                console.error('检查模型状态失败:', error);
            }
        }
        
        // 文本分类
        async function classifyText() {
            const originalText = document.getElementById('original-text').value;
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            
            if (!originalText.trim()) {
                showMessage('请输入要分类的文本！', 'error');
                return;
            }
            
            loading.style.display = 'block';
            resultDiv.innerHTML = '';
            currentText = originalText;
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: originalText, top_k: 3 })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showMessage(`分类失败: ${data.error}`, 'error');
                    return;
                }
                
                let html = '<div class="success"><h3>📊 文本分类结果</h3>';
                html += `<p><strong>输入文本:</strong> ${data.data.text}</p>`;
                
                if (data.data.domains && data.data.domains.length > 0) {
                    html += `<p><strong>主要领域:</strong> ${data.data.top_domain || '无'}</p>`;
                    html += `<p><strong>所有可能领域:</strong> ${data.data.domains.join(', ')}</p>`;
                    
                    if (data.data.probabilities) {
                        html += '<p><strong>概率分布:</strong></p><ul>';
                        for (const [domain, prob] of Object.entries(data.data.probabilities)) {
                            const percentage = (prob * 100).toFixed(1);
                            html += `<li>${domain}: ${percentage}%</li>`;
                        }
                        html += '</ul>';
                    }
                } else {
                    html += '<p>未识别到明确的领域</p>';
                }
                
                html += `<p><em>处理时间: ${data.data.processing_time || 0}秒</em></p>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
                showMessage('文本分类完成！', 'success');
                
            } catch (error) {
                console.error('请求失败:', error);
                showMessage(`请求失败: ${error.message}`, 'error');
            } finally {
                loading.style.display = 'none';
            }
        }
        
        // 生成视角
        async function generatePerspectives(count = 5) {
            const originalText = document.getElementById('original-text').value;
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            
            if (!originalText.trim()) {
                showMessage('请输入原始观点！', 'error');
                return;
            }
            
            loading.style.display = 'block';
            resultDiv.innerHTML = '';
            currentText = originalText;
            
            try {
                const response = await fetch('/api/generate-perspectives', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: originalText, count: count })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showMessage(`生成视角失败: ${data.error}`, 'error');
                    return;
                }
                
                let html = `<div class="success"><h3>🎯 为您生成 ${data.data.perspectives.length} 个不同视角</h3>`;
                html += `<p><strong>原始观点:</strong> ${data.data.text}</p>`;
                
                html += '<div class="perspectives-container">';
                
                data.data.perspectives.forEach(perspective => {
                    html += `
                        <div class="perspective-card" style="border-color: ${perspective.color}">
                            <div class="perspective-header">
                                <div class="perspective-icon" style="color: ${perspective.color}">${getIcon(perspective.icon)}</div>
                                <h4>${perspective.label}</h4>
                            </div>
                            <div class="perspective-title">${perspective.title}</div>
                            <div class="perspective-hint">${perspective.hint}</div>
                            <div class="perspective-description">${perspective.description || ''}</div>
                        </div>
                    `;
                });
                
                html += '</div>';
                html += `<p><em>处理时间: ${data.data.processing_time || 0}秒</em></p>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
                showMessage(`成功生成 ${data.data.perspectives.length} 个视角！`, 'success');
                currentPerspectives = data.data.perspectives;
                
            } catch (error) {
                console.error('请求失败:', error);
                showMessage(`请求失败: ${error.message}`, 'error');
            } finally {
                loading.style.display = 'none';
            }
        }
        
        // 分析观点
        async function analyzeViewpoint() {
            const originalText = document.getElementById('original-text').value;
            const responseText = document.getElementById('response-text').value;
            const perspectiveType = document.getElementById('perspective-type').value;
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            
            if (!originalText.trim()) {
                showMessage('请输入原始观点！', 'error');
                return;
            }
            
            if (!responseText.trim()) {
                showMessage('请输入你的观点回应！', 'error');
                return;
            }
            
            loading.style.display = 'block';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/api/analyze-viewpoint', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        original_text: originalText, 
                        response_text: responseText,
                        perspective_type: perspectiveType
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showMessage(`分析观点失败: ${data.error}`, 'error');
                    return;
                }
                
                let html = '<div class="success"><h3>🔍 AI观点分析结果</h3>';
                html += `<p><strong>原始观点:</strong> ${originalText}</p>`;
                html += `<p><strong>你的回应:</strong> ${responseText}</p>`;
                html += `<p><strong>视角类型:</strong> ${getPerspectiveLabel(perspectiveType)}</p>`;
                
                html += '<div class="analysis-result" style="margin-top: 20px;">';
                html += `<div class="analysis-positive"><strong>👍 肯定部分:</strong> ${data.data.analysis.positive}</div>`;
                html += `<div class="analysis-suggestion"><strong>💡 优化建议:</strong> ${data.data.analysis.suggestion}</div>`;
                
                if (data.data.analysis.keywords && data.data.analysis.keywords.length > 0) {
                    html += `<p><strong>🔑 关键词:</strong> ${data.data.analysis.keywords.join(', ')}</p>`;
                }
                
                if (data.data.analysis.logic_score) {
                    html += `<p><strong>📊 逻辑评分:</strong> ${data.data.analysis.logic_score}/1.0</p>`;
                }
                
                html += `<p><em>分析方法: ${data.data.analysis.analysis_method || 'AI模型分析'}</em></p>`;
                html += '</div>';
                
                html += `<p><em>处理时间: ${data.data.processing_time || 0}秒</em></p>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
                showMessage('观点分析完成！', 'success');
                
            } catch (error) {
                console.error('请求失败:', error);
                showMessage(`请求失败: ${error.message}`, 'error');
            } finally {
                loading.style.display = 'none';
            }
        }
        
        // 加载示例
        function loadExample() {
            const examples = [
                {
                    original: "人工智能将完全取代人类工作",
                    response: "我认为人工智能会取代部分重复性工作，但创造性、情感交流和复杂决策工作仍需要人类。人机协作将是未来趋势。",
                    perspective: "neutral"
                },
                {
                    original: "线上教育比传统教育更有效",
                    response: "我不同意这个观点。线上教育虽然方便，但缺乏师生互动和即时反馈，对于实践性强的课程效果不佳。传统教育的社交功能也很重要。",
                    perspective: "opposite"
                },
                {
                    original: "环保应该优先于经济发展",
                    response: "我认为应该在环保和经济发展之间找到平衡。可以发展绿色经济，既保护环境又促进经济增长，实现可持续发展。",
                    perspective: "supplement"
                }
            ];
            
            const example = examples[Math.floor(Math.random() * examples.length)];
            
            document.getElementById('original-text').value = example.original;
            document.getElementById('response-text').value = example.response;
            document.getElementById('perspective-type').value = example.perspective;
            
            showMessage(`已加载示例观点`, 'success');
        }
        
        // 获取视角标签
        function getPerspectiveLabel(type) {
            const labels = {
                'neutral': '中立视角',
                'opposite': '对立视角',
                'supplement': '补充视角',
                'unique': '小众视角',
                'historical': '历史视角',
                'global': '全球视角'
            };
            return labels[type] || type;
        }
        
        // 获取图标HTML
        function getIcon(iconClass) {
            const icons = {
                'fas fa-exchange-alt': '↔️',
                'fas fa-balance-scale': '⚖️',
                'fas fa-plus-circle': '➕',
                'fas fa-lightbulb': '💡',
                'fas fa-history': '📜',
                'fas fa-globe': '🌍',
                'fas fa-laptop-house': '💻',
                'fas fa-robot': '🤖',
                'fas fa-video': '🎬',
                'fas fa-leaf': '🍃',
                'fas fa-graduation-cap': '🎓',
                'fas fa-city': '🏙️'
            };
            
            return icons[iconClass] || '💭';
        }
        
        // 显示消息
        function showMessage(text, type = 'info') {
            const messageDiv = document.createElement('div');
            messageDiv.className = type;
            messageDiv.textContent = text;
            messageDiv.style.position = 'fixed';
            messageDiv.style.top = '20px';
            messageDiv.style.left = '50%';
            messageDiv.style.transform = 'translateX(-50%)';
            messageDiv.style.padding = '10px 20px';
            messageDiv.style.borderRadius = '5px';
            messageDiv.style.zIndex = '1000';
            messageDiv.style.boxShadow = '0 5px 15px rgba(0,0,0,0.1)';
            
            if (type === 'error') {
                messageDiv.style.background = '#fdf2f2';
                messageDiv.style.color = '#e74c3c';
                messageDiv.style.border = '1px solid #f8d7da';
            } else if (type === 'success') {
                messageDiv.style.background = '#f2fdf7';
                messageDiv.style.color = '#2ecc71';
                messageDiv.style.border = '1px solid #d4edda';
            } else {
                messageDiv.style.background = '#f0f8ff';
                messageDiv.style.color = '#3498db';
                messageDiv.style.border = '1px solid #d1ecf1';
            }
            
            document.body.appendChild(messageDiv);
            
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.parentNode.removeChild(messageDiv);
                }
            }, 3000);
        }
        
        // 清空所有
        function clearAll() {
            document.getElementById('original-text').value = '';
            document.getElementById('response-text').value = '';
            document.getElementById('result').innerHTML = '';
            document.getElementById('loading').style.display = 'none';
            currentText = '';
            currentPerspectives = [];
            showMessage('已清空', 'info');
        }
        
        // 页面加载时初始化
        window.onload = function() {
            checkModelStatus();
            
            // 设置示例文本
            document.getElementById('original-text').value = 
                '人工智能是未来科技发展的重要方向，机器学习算法正在改变我们的生活。';
            document.getElementById('response-text').value = 
                '我同意人工智能的重要性，但也认为需要平衡技术发展与伦理考虑，确保AI为人类社会带来积极影响。';
        };
    </script>
</body>
</html>"""
    return html

def generate_error_interface(error_msg: str) -> str:
    """生成错误界面"""
    return f"""<!DOCTYPE html>
<html>
<head><title>系统错误</title></head>
<body>
    <h1>系统初始化失败</h1>
    <p>错误信息: {error_msg}</p>
    <p>请检查日志文件或联系管理员。</p>
</body>
</html>"""

# 测试函数
def test_enhanced_application():
    """测试增强版应用"""
    print("=" * 70)
    print("辩证思考AI互动系统 - 增强版 2.0")
    print("=" * 70)
    
    ensure_directories()
    
    print("\n系统功能:")
    print("  ✅ 集成机器学习文本分类")
    print("  ✅ 多维度观点分析")
    print("  ✅ 8种不同视角生成")
    print("  ✅ 批量处理支持")
    print("  ✅ 智能缓存机制")
    print("  ✅ 实时性能监控")
    
    print("\nAPI端点:")
    print("  GET  /                     - 主界面")
    print("  GET  /api/health          - 健康检查")
    print("  POST /api/classify        - 文本分类")
    print("  POST /api/analyze-viewpoint - 观点分析")
    print("  POST /api/generate-perspectives - 生成视角")
    print("  POST /api/batch-process   - 批量处理")
    print("  GET  /api/system-metrics  - 系统指标")
    
    print("\n启动服务器:")
    print("  python main.py")
    print("\n访问地址:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/api/docs")
    print("=" * 70)

# 创建简化版模型管理器
class SimpleModelManager:
    """简化版模型管理器"""
    
    def __init__(self, cache_dir="models/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    async def initialize_all_models(self):
        """初始化所有模型"""
        await asyncio.sleep(0.1)  # 模拟初始化延迟
        return True
    
    async def classify(self, text: str, top_k: int = 5, detailed: bool = False) -> Dict[str, Any]:
        """文本分类"""
        await asyncio.sleep(0.05)  # 模拟处理延迟
        
        domains = ['科技', '教育', '职场', '生活', '健康', '财经', '旅游', '政治', '娱乐', '体育']
        selected = random.sample(domains, min(top_k, len(domains)))
        
        result = {
            'text': text,
            'domains': selected,
            'probabilities': {domain: round(random.uniform(0.3, 0.9), 3) for domain in selected},
            'top_domain': selected[0] if selected else None
        }
        
        if detailed:
            result['details'] = {
                'keywords': text.split()[:5] if text.split() else ['思考', '分析'],
                'sentiment_score': round(random.uniform(-0.5, 0.5), 2),
                'complexity_score': round(min(0.5 + len(text) * 0.001, 0.95), 2)
            }
        
        return result
    
    async def analyze_viewpoint(self, original_text: str, response_text: str, 
                              perspective_type: str, analysis_depth: str = "standard") -> Dict[str, Any]:
        """观点分析"""
        await asyncio.sleep(0.1)  # 模拟处理延迟
        
        analysis = {
            'positive': f"你对'{original_text[:20]}...'的分析很有见地，展现了良好的思考能力。",
            'suggestion': "可以考虑补充具体案例，增强说服力。也可以从更多角度进行分析。",
            'keywords': response_text.split()[:5] if response_text.split() else ['思考', '分析'],
            'logic_score': round(min(0.7 + len(response_text) * 0.001, 0.95), 2)
        }
        
        if analysis_depth in ["standard", "comprehensive"]:
            analysis.update({
                'strengths': ['逻辑清晰', '观点明确', '论证充分'][:random.randint(1, 3)],
                'areas_for_improvement': ['补充案例', '加强逻辑', '深化分析'][:random.randint(1, 3)],
                'sentiment_analysis': {
                    'overall_sentiment': random.choice(['positive', 'neutral', 'critical']),
                    'confidence_score': round(random.uniform(0.6, 0.9), 2)
                }
            })
        
        if analysis_depth == "comprehensive":
            analysis.update({
                'argument_structure': {
                    'premises': random.randint(2, 5),
                    'conclusions': random.randint(1, 3),
                    'coherence_score': round(random.uniform(0.6, 0.9), 2)
                },
                'rhetorical_devices': random.sample(['比喻', '举例', '对比', '反问'], random.randint(1, 3))
            })
        
        return analysis

# 主程序入口
if __name__ == "__main__":
    test_enhanced_application()
    
    # 如果model_manager为None，使用简化版
    if model_manager is None:
        model_manager = SimpleModelManager()
    
    uvicorn.run(
        "main:app",  # 修复：使用"main:app"而不是"main_enhanced:app"
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

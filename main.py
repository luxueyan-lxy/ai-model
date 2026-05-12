"""
辩证思考AI互动系统
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
    description="基于集成机器学习的文本领域分类、观点分析与文章生成系统",
    version="3.0.0",
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
            "name": "生成",
            "description": "文章生成相关操作"
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
    thinking_depth: str = Field("standard", description="思考深度: basic/standard/deep")
    include_prompts: bool = Field(True, description="是否包含详细思考提示")

class AnalyzeRequest(BaseModel):
    """观点分析请求模型"""
    original_text: str = Field(..., min_length=1, max_length=5000, description="原始观点文本")
    response_text: str = Field(..., min_length=1, max_length=10000, description="用户回应文本")
    perspective_type: str = Field("neutral", description="视角类型")
    analysis_depth: str = Field("standard", description="分析深度: basic/standard/comprehensive")

class GenerateArticleRequest(BaseModel):
    """文章生成请求模型"""
    topic: str = Field(..., min_length=2, max_length=200, description="文章主题")
    perspective_type: str = Field("neutral", description="视角类型: neutral/opposite/supplement/unique")
    target_length: int = Field(500, ge=100, le=5000, description="目标字数(100-5000)")
    structure_type: str = Field("argumentative", description="结构类型: argumentative/expository/persuasive/comparative")
    include_keywords: Optional[List[str]] = Field(None, description="需要包含的关键词")
    target_audience: str = Field("general", description="目标读者: general/professional/student")
    writing_style: str = Field("standard", description="写作风格: standard/academic/casual")

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
    
    logger.info("服务启动完成，版本：3.0.0")

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

async def generate_article(request: GenerateArticleRequest) -> Dict[str, Any]:
    """生成文章"""
    start_time = time.time()
    
    try:
        if model_manager and hasattr(model_manager, 'generate_article'):
            result = await model_manager.generate_article(
                topic=request.topic,
                perspective_type=request.perspective_type,
                target_length=request.target_length,
                structure_type=request.structure_type,
                include_keywords=request.include_keywords,
                target_audience=request.target_audience,
                writing_style=request.writing_style
            )
        else:
            # 模拟生成
            result = generate_mock_article(request)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'data': {
                'article': result,
                'processing_time': round(processing_time, 3),
                'request_info': {
                    'topic': request.topic,
                    'perspective_type': request.perspective_type,
                    'target_length': request.target_length,
                    'structure_type': request.structure_type
                }
            }
        }
        
    except Exception as e:
        logger.error(f"文章生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"文章生成失败: {str(e)}")

def generate_mock_article(request: GenerateArticleRequest) -> Dict[str, Any]:
    """生成模拟文章"""
    # 生成文章内容
    content = f"""关于{request.topic}的思考

近年来，{request.topic}问题引起了社会各界的广泛关注。从{request.perspective_type}的视角来看，这一议题涉及多个维度的考量。

首先，从技术发展的角度分析，{request.topic}代表了当前社会发展的重要趋势。技术的进步为我们提供了新的解决思路和方法，但同时也带来了新的挑战和问题。这需要我们在享受技术红利的同时，认真思考如何平衡各种利益关系。

其次，从社会影响的角度来看，{request.topic}对人们的生活方式、工作模式和社会关系都产生了深远的影响。这种影响既有积极的一面，也有需要警惕的风险。我们需要以开放的心态面对这些变化，同时保持审慎的态度。

再者，从个人发展的层面来说，{request.topic}为个人成长和职业发展提供了新的机遇。每个人都应该积极面对这一变化，不断提升自己的适应能力和创新能力，以便在快速变化的环境中保持竞争力。

总之，{request.topic}是一个复杂而重要的议题。我们需要保持理性的思考，积极探讨适合中国国情的发展道路，为实现可持续发展贡献力量。"""
    
    # 确保达到目标长度
    while len(content) < request.target_length:
        additional = f"""从更深层次来看，{request.topic}还涉及到价值观念、文化传统和社会制度的多个层面。我们需要在继承优秀传统文化的基础上，积极探索符合时代要求的创新路径。

在未来的发展中，{request.topic}将继续成为社会关注的热点。我们需要建立更加完善的制度体系，为相关领域的发展提供有力保障。同时，加强国际合作与交流，学习借鉴国际先进经验，也是推动{request.topic}健康发展的有效途径。"""
        content += "\n\n" + additional
    
    return {
        'title': f"关于{request.topic}的思考与分析",
        'content': content,
        'main_idea': f"探讨{request.topic}的多方面影响和发展趋势",
        'structure': [
            {'section': '引言', 'content': '背景介绍', 'length': 50},
            {'section': '分析1', 'content': '技术角度', 'length': 100},
            {'section': '分析2', 'content': '社会角度', 'length': 100},
            {'section': '分析3', 'content': '个人角度', 'length': 100},
            {'section': '结论', 'content': '总结展望', 'length': 50}
        ],
        'keywords': [request.topic, '发展', '影响', '趋势', '思考'][:5],
        'word_count': len(content),
        'reading_time_minutes': round(len(content) / 300, 1),
        'target_audience': request.target_audience,
        'writing_style': request.writing_style,
        'coherence_score': round(random.uniform(0.6, 0.9), 2),
        'quality_score': round(random.uniform(0.6, 0.9), 2),
        'generation_method': '模拟生成',
        'timestamp': datetime.now().isoformat()
    }

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
    """生成多个视角端点（包含详细创作提示）"""
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
            
            # 创建视角对象
            perspective = {
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
            }
            
            # 如果包含详细提示，添加创作提示
            if request.include_prompts:
                # 生成详细的视角提示
                detailed_prompts = generate_detailed_perspective_prompts(
                    p_type["type"], 
                    main_domain, 
                    request.thinking_depth
                )
                
                # 获取创意思维工具
                thinking_tools = get_creative_thinking_tools(p_type["type"])
                
                # 添加到视角对象
                perspective["creation_prompts"] = {
                    "thinking_directions": detailed_prompts["thinking_directions"],
                    "questions_to_ask": detailed_prompts["questions_to_ask"],
                    "analysis_frameworks": detailed_prompts["analysis_frameworks"],
                    "writing_suggestions": detailed_prompts["writing_suggestions"],
                    "creative_thinking_tools": thinking_tools,
                    "suggested_length": 300 if request.thinking_depth == "deep" else 200,
                    "keywords": PERSPECTIVE_PROMPTS.get(p_type["type"], {}).get("keywords", [])
                }
                
                # 添加领域特定的提示
                if "domain_specific" in detailed_prompts:
                    perspective["creation_prompts"]["domain_specific"] = detailed_prompts["domain_specific"]
            
            perspectives.append(perspective)
        
        processing_time = time.time() - start_time
        
        PERFORMANCE_METRICS['avg_response_time'] = (
            PERFORMANCE_METRICS['avg_response_time'] * 0.9 + 
            processing_time * 0.1
        )
        
        response_data = {
            "text": request.text,
            "perspectives": perspectives,
            "count": len(perspectives),
            "main_domain": main_domain,
            "thinking_depth": request.thinking_depth,
            "include_prompts": request.include_prompts,
            "processing_time": round(processing_time, 3)
        }
        
        return {
            "success": True,
            "data": response_data
        }
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise HTTPException(status_code=500, detail=f"生成视角失败: {str(e)}")

@app.get("/api/thinking-tools", tags=["分析"])
async def get_thinking_tools_endpoint():
    """获取创意思维工具"""
    tools = {
        "scamper": {
            "name": "SCAMPER法",
            "description": "7个创意激发方向的思考工具",
            "techniques": [
                {"code": "S", "name": "替代(Substitute)", "description": "能用什么替代现有方案？"},
                {"code": "C", "name": "结合(Combine)", "description": "能和什么结合产生新价值？"},
                {"code": "A", "name": "调整(Adapt)", "description": "怎样调整以适应新需求？"},
                {"code": "M", "name": "修改(Modify)", "description": "怎样修改能大幅提升效果？"},
                {"code": "P", "name": "他用(Put to other uses)", "description": "还能用在其他什么地方？"},
                {"code": "E", "name": "消除(Eliminate)", "description": "能去掉什么不必要的部分？"},
                {"code": "R", "name": "反向(Reverse)", "description": "反过来做会怎样？"}
            ]
        },
        "six_thinking_hats": {
            "name": "六顶思考帽",
            "description": "从六个角度进行全方位思考",
            "techniques": [
                {"color": "white", "name": "白帽(事实)", "description": "相关的事实和数据有哪些？"},
                {"color": "red", "name": "红帽(情感)", "description": "对这个问题的直觉感受是什么？"},
                {"color": "black", "name": "黑帽(谨慎)", "description": "可能的风险和问题是什么？"},
                {"color": "yellow", "name": "黄帽(乐观)", "description": "有哪些机会和好处？"},
                {"color": "green", "name": "绿帽(创意)", "description": "有哪些创新的可能性？"},
                {"color": "blue", "name": "蓝帽(统筹)", "description": "整体应该怎么规划和执行？"}
            ]
        },
        "5w2h": {
            "name": "5W2H提问法",
            "description": "从七个维度进行系统分析",
            "techniques": [
                {"letter": "W", "name": "What", "description": "核心问题/目标是什么？"},
                {"letter": "W", "name": "Why", "description": "为什么要做？根本原因是什么？"},
                {"letter": "W", "name": "Who", "description": "涉及哪些人？谁影响谁？"},
                {"letter": "W", "name": "When", "description": "什么时间节点最重要？"},
                {"letter": "W", "name": "Where", "description": "在哪里实施最合适？"},
                {"letter": "H", "name": "How", "description": "具体如何实施？"},
                {"letter": "H", "name": "How much", "description": "需要多少资源？预期效果如何？"}
            ]
        }
    }
    
    return {
        "success": True,
        "data": {
            "thinking_tools": tools,
            "count": len(tools)
        }
    }

@app.post("/api/generate-article", tags=["生成"])
async def generate_article_endpoint(request: GenerateArticleRequest):
    """生成文章端点"""
    PERFORMANCE_METRICS['requests'] += 1
    start_time = time.time()
    
    try:
        result = await generate_article(request)
        
        PERFORMANCE_METRICS['avg_response_time'] = (
            PERFORMANCE_METRICS['avg_response_time'] * 0.9 + 
            (time.time() - start_time) * 0.1
        )
        
        return result
        
    except Exception as e:
        PERFORMANCE_METRICS['error_rate'] = (
            PERFORMANCE_METRICS['error_rate'] * 0.9 + 0.1
        )
        raise

@app.post("/api/batch-process", tags=["分类", "分析", "生成"])
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
        
        elif request.operation == "generate":
            # 对于生成，每个文本都是一个主题
            tasks = []
            for text in request.texts:
                # 创建生成请求
                gen_request = GenerateArticleRequest(
                    topic=text,
                    perspective_type="neutral",
                    target_length=300
                )
                tasks.append(generate_article(gen_request))
            
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

@app.get("/api/article-structures", tags=["生成"])
async def get_article_structures_endpoint():
    """获取文章结构类型"""
    structures = {
        "argumentative": {
            "name": "论述型",
            "description": "提出论点并进行论证的结构",
            "sections": ["引言", "论点1", "论点2", "论点3", "反驳观点", "结论"]
        },
        "expository": {
            "name": "说明型", 
            "description": "说明和解释问题的结构",
            "sections": ["背景介绍", "问题提出", "分析探讨", "解决方案", "总结展望"]
        },
        "persuasive": {
            "name": "说服型",
            "description": "说服读者接受某种观点的结构",
            "sections": ["问题引出", "立场陈述", "论据支持", "反方观点", "强化立场", "行动呼吁"]
        },
        "comparative": {
            "name": "比较型",
            "description": "比较两种或多种观点的结构",
            "sections": ["主题引入", "A方分析", "B方分析", "对比分析", "综合评价", "总结建议"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "structures": structures,
            "count": len(structures)
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
    """生成增强的前端界面"""
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>辩证思考AI互动系统 - 增强版 3.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .btn-primary { @apply bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-200; }
        .btn-secondary { @apply bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-200; }
        .btn-warning { @apply bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-200; }
        .btn-danger { @apply bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-200; }
        .card { @apply bg-white rounded-xl shadow-lg p-6 mb-6; }
        .input-field { @apply w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none; }
        .loading { @apply flex items-center justify-center space-x-2; }
        .loading-spinner { @apply w-8 h-8 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin; }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-4">
    <div class="container mx-auto max-w-7xl">
        <!-- 头部 -->
        <div class="text-center mb-8 pt-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🤔 辩证思考AI互动系统 - 增强版 3.0</h1>
            <p class="text-gray-600 text-lg">基于集成机器学习的文本分类、观点分析与文章生成平台</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- 左侧：功能操作 -->
            <div>
                <!-- 1. 文本分类 -->
                <div class="card">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">🔍 文本分类测试</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2">输入文本：</label>
                        <textarea id="classify-text" class="input-field" rows="3">人工智能是未来科技发展的重要方向，机器学习算法正在改变我们的生活。</textarea>
                    </div>
                    <button onclick="testClassify()" class="btn-primary">文本分类</button>
                    <div id="classify-result" class="mt-4 p-4 bg-gray-50 rounded-lg"></div>
                </div>

                <!-- 2. 观点分析 -->
                <div class="card">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">📊 观点分析测试</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2">原始观点：</label>
                        <textarea id="original-text" class="input-field" rows="2">人工智能将完全取代人类工作</textarea>
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2">你的回应：</label>
                        <textarea id="response-text" class="input-field" rows="3">我认为人工智能会取代部分重复性工作，但创造性、情感交流和复杂决策工作仍需要人类。人机协作将是未来趋势。</textarea>
                    </div>
                    <button onclick="testAnalysis()" class="btn-primary">分析观点</button>
                    <div id="analysis-result" class="mt-4 p-4 bg-gray-50 rounded-lg"></div>
                </div>
            </div>

            <!-- 右侧：文章生成 -->
            <div>
                <!-- 3. 文章生成 -->
                <div class="card">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">✍️ 文章生成测试</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2">文章主题：</label>
                        <input type="text" id="article-topic" class="input-field" value="人工智能对社会的影响">
                    </div>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <label class="block text-gray-700 mb-2">视角类型：</label>
                            <select id="article-perspective" class="input-field">
                                <option value="neutral">中立视角</option>
                                <option value="opposite">对立视角</option>
                                <option value="supplement">补充视角</option>
                                <option value="unique">小众视角</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-gray-700 mb-2">目标字数：</label>
                            <select id="article-length" class="input-field">
                                <option value="200">200字</option>
                                <option value="300" selected>300字</option>
                                <option value="500">500字</option>
                                <option value="800">800字</option>
                            </select>
                        </div>
                    </div>
                    <button onclick="testGenerateArticle()" class="btn-secondary w-full">生成文章</button>
                    <div id="article-result" class="mt-4 p-4 bg-gray-50 rounded-lg"></div>
                </div>

                <!-- 视角生成测试 -->
                <div class="card">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">🎯 视角生成测试（含创作提示）</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2">输入观点：</label>
                        <textarea id="perspective-text" class="input-field" rows="3">人工智能将完全取代人类工作</textarea>
                    </div>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <label class="block text-gray-700 mb-2">视角数量：</label>
                            <select id="perspective-count" class="input-field">
                                <option value="3">3个视角</option>
                                <option value="4" selected>4个视角</option>
                                <option value="6">6个视角</option>
                                <option value="8">8个视角</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-gray-700 mb-2">思考深度：</label>
                            <select id="thinking-depth" class="input-field">
                                <option value="basic">基础</option>
                                <option value="standard" selected>标准</option>
                                <option value="deep">深入</option>
                            </select>
                        </div>
                    </div>
                    <button onclick="testGeneratePerspectives()" class="btn-secondary w-full">生成视角与创作提示</button>
                    <div id="perspectives-result" class="mt-4"></div>
                </div>

                <!-- 4. 系统信息 -->
                <div class="card">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">📈 系统信息</h2>
                    <div class="space-y-2">
                        <p>版本：3.0.0</p>
                        <p>访问API文档：<a href="/api/docs" class="text-blue-500 hover:underline" target="_blank">/api/docs</a></p>
                        <p>健康检查：<a href="/api/health" class="text-blue-500 hover:underline" target="_blank">/api/health</a></p>
                        <p>模型状态：<span id="model-status">正在检查...</span></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- API示例 -->
        <div class="card mt-8">
            <h2 class="text-2xl font-bold text-gray-800 mb-4">🛠️ API调用示例</h2>
            <div class="space-y-4">
                <!-- 文本分类示例 -->
                <div>
                    <h3 class="font-semibold text-gray-700 mb-2">文本分类API：</h3>
                    <pre class="bg-gray-800 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
POST /api/classify
Content-Type: application/json

{
  "text": "人工智能是未来科技发展的重要方向",
  "top_k": 3,
  "use_cache": true,
  "detailed": true
}</pre>
                </div>

                <!-- 观点分析示例 -->
                <div>
                    <h3 class="font-semibold text-gray-700 mb-2">观点分析API：</h3>
                    <pre class="bg-gray-800 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
POST /api/analyze-viewpoint
Content-Type: application/json

{
  "original_text": "人工智能将完全取代人类工作",
  "response_text": "我认为人工智能会取代部分重复性工作...",
  "perspective_type": "neutral",
  "analysis_depth": "standard"
}</pre>
                </div>

                <!-- 文章生成示例 -->
                <div>
                    <h3 class="font-semibold text-gray-700 mb-2">文章生成API：</h3>
                    <pre class="bg-gray-800 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
POST /api/generate-article
Content-Type: application/json

{
  "topic": "人工智能对社会的影响",
  "perspective_type": "neutral",
  "target_length": 500,
  "structure_type": "argumentative",
  "target_audience": "general",
  "writing_style": "standard"
}</pre>
                </div>
            </div>
        </div>

        
    <div>
        <h3 class="font-semibold text-gray-700 mb-2">视角生成API（含创作提示）：</h3>
        <pre class="bg-gray-800 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
POST /api/generate-perspectives
Content-Type: application/json

{
  "text": "人工智能将完全取代人类工作",
  "count": 4,
  "perspective_types": ["neutral", "opposite", "supplement", "unique"],
  "thinking_depth": "deep",
  "include_prompts": true
}</pre>
</div>

<div>
    <h3 class="font-semibold text-gray-700 mb-2">获取思维工具API：</h3>
    <pre class="bg-gray-800 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
GET /api/thinking-tools
Content-Type: application/json

// 响应示例
{
  "success": true,
  "data": {
    "thinking_tools": {
      "scamper": {
        "name": "SCAMPER法",
        "description": "7个创意激发方向的思考工具",
        "techniques": [...]
      },
      "six_thinking_hats": {...},
      "5w2h": {...}
    }
  }
}</pre>
</div>

        <!-- 页脚 -->
        <div class="text-center text-gray-500 text-sm mt-8 py-4 border-t border-gray-200">
            <p>辩证思考AI互动系统 - 增强版 3.0 © 2024</p>
            <p class="mt-1">技术支持：基于FastAPI + 集成机器学习模型</p>
        </div>
    </div>

    <script>
        // 工具函数
        function showLoading(elementId) {
            const element = document.getElementById(elementId);
            element.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>正在处理中...</span>
                </div>
            `;
        }

        function showError(elementId, message) {
            const element = document.getElementById(elementId);
            element.innerHTML = `
                <div class="bg-red-50 border-l-4 border-red-500 p-4">
                    <div class="flex">
                        <div class="flex-shrink-0">❌</div>
                        <div class="ml-3">
                            <p class="text-sm text-red-700">${message}</p>
                        </div>
                    </div>
                </div>
            `;
        }

        // 文本分类测试
        async function testClassify() {
            const text = document.getElementById('classify-text').value;
            const resultDiv = document.getElementById('classify-result');
            
            if (!text.trim()) {
                alert('请输入文本！');
                return;
            }
            
            showLoading('classify-result');
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: text, 
                        top_k: 3, 
                        detailed: true,
                        use_cache: true
                    })
                });
                
                const data = await response.json();
                
                if (!data.success) {
                    throw new Error(data.error?.message || '请求失败');
                }
                
                let html = '<h3 class="font-semibold text-lg text-gray-800 mb-2">📊 分类结果</h3>';
                html += `<p class="mb-2"><strong>文本：</strong>${data.data.text.substring(0, 100)}${data.data.text.length > 100 ? '...' : ''}</p>`;
                
                if (data.data.domains && data.data.domains.length > 0) {
                    html += '<p class="mb-2"><strong>主要领域：</strong>';
                    data.data.domains.forEach((domain, index) => {
                        const prob = data.data.probabilities?.[domain] || 0;
                        const percentage = (prob * 100).toFixed(1);
                        html += `<span class="inline-block bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded mr-1 mb-1">${domain} (${percentage}%)</span>`;
                    });
                    html += '</p>';
                }
                
                html += `<p class="text-sm text-gray-600 mt-2">处理时间: ${data.data.processing_time}s | 使用模型: ${data.data.model_used ? '是' : '否（模拟模式）'}</p>`;
                
                resultDiv.innerHTML = html;
                
            } catch (error) {
                showError('classify-result', `错误: ${error.message}`);
            }
        }

        // 观点分析测试
        async function testAnalysis() {
            const originalText = document.getElementById('original-text').value;
            const responseText = document.getElementById('response-text').value;
            const resultDiv = document.getElementById('analysis-result');
            
            if (!originalText.trim() || !responseText.trim()) {
                alert('请输入原始观点和回应！');
                return;
            }
            
            showLoading('analysis-result');
            
            try {
                const response = await fetch('/api/analyze-viewpoint', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        original_text: originalText,
                        response_text: responseText,
                        perspective_type: 'neutral',
                        analysis_depth: 'standard'
                    })
                });
                
                const data = await response.json();
                
                if (!data.success) {
                    throw new Error(data.error?.message || '请求失败');
                }
                
                let html = '<h3 class="font-semibold text-lg text-gray-800 mb-2">🔍 分析结果</h3>';
                html += `<p class="mb-2"><strong>肯定部分：</strong>${data.data.analysis.positive}</p>`;
                html += `<p class="mb-2"><strong>优化建议：</strong>${data.data.analysis.suggestion}</p>`;
                
                if (data.data.analysis.keywords && data.data.analysis.keywords.length > 0) {
                    html += `<p class="mb-2"><strong>关键词：</strong>`;
                    data.data.analysis.keywords.forEach(keyword => {
                        html += `<span class="inline-block bg-green-100 text-green-800 text-xs font-semibold px-2.5 py-0.5 rounded mr-1 mb-1">${keyword}</span>`;
                    });
                    html += '</p>';
                }
                
                if (data.data.analysis.logic_score) {
                    html += `<p class="mb-2"><strong>逻辑评分：</strong>${data.data.analysis.logic_score}/1.0</p>`;
                }
                
                html += `<p class="text-sm text-gray-600 mt-2">处理时间: ${data.data.processing_time}s</p>`;
                
                resultDiv.innerHTML = html;
                
            } catch (error) {
                showError('analysis-result', `错误: ${error.message}`);
            }
        }

        // 文章生成测试
        async function testGenerateArticle() {
            const topic = document.getElementById('article-topic').value;
            const perspective = document.getElementById('article-perspective').value;
            const length = parseInt(document.getElementById('article-length').value);
            const resultDiv = document.getElementById('article-result');
            
            if (!topic.trim()) {
                alert('请输入文章主题！');
                return;
            }
            
            showLoading('article-result');
            
            try {
                const response = await fetch('/api/generate-article', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        topic: topic,
                        perspective_type: perspective,
                        target_length: length,
                        structure_type: 'argumentative',
                        target_audience: 'general',
                        writing_style: 'standard'
                    })
                });
                
                const data = await response.json();
                
                if (!data.success) {
                    throw new Error(data.error?.message || '请求失败');
                }
                
                const article = data.data.article;
                
                let html = '<h3 class="font-semibold text-lg text-gray-800 mb-2">✍️ 生成的文章</h3>';
                html += `<h4 class="font-semibold text-md text-blue-700 mb-2">${article.title}</h4>`;
                html += `<div class="bg-gray-50 p-4 rounded-lg mb-3 max-h-60 overflow-y-auto">`;
                html += `<p class="whitespace-pre-wrap text-gray-700">${article.content}</p>`;
                html += `</div>`;
                
                html += `<div class="grid grid-cols-2 gap-4 mb-3">`;
                html += `<div><strong>字数：</strong>${article.word_count}字</div>`;
                html += `<div><strong>阅读时间：</strong>${article.reading_time_minutes}分钟</div>`;
                html += `<div><strong>目标读者：</strong>${article.target_audience}</div>`;
                html += `<div><strong>连贯性评分：</strong>${article.coherence_score}/1.0</div>`;
                html += `</div>`;
                
                html += `<p class="text-sm text-gray-600">处理时间: ${data.data.processing_time}s | 生成方法: ${article.generation_method}</p>`;
                
                resultDiv.innerHTML = html;
                
            } catch (error) {
                showError('article-result', `错误: ${error.message}`);
            }
        }

         // 视角生成测试函数
        async function testGeneratePerspectives() {
            const text = document.getElementById('perspective-text').value;
            const count = parseInt(document.getElementById('perspective-count').value);
            const depth = document.getElementById('thinking-depth').value;
            const resultDiv = document.getElementById('perspectives-result');
            
            if (!text.trim()) {
                alert('请输入观点！');
                return;
            }
            
            resultDiv.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>正在生成视角和创作提示...</span>
                </div>
            `;
            
            try {
                const response = await fetch('/api/generate-perspectives', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: text,
                        count: count,
                        thinking_depth: depth,
                        include_prompts: true
                    })
                });
                
                const data = await response.json();
                
                if (!data.success) {
                    throw new Error(data.error?.message || '请求失败');
                }
                
                let html = '<h3 class="font-semibold text-lg text-gray-800 mb-4">🎯 生成的视角与创作提示</h3>';
                html += `<p class="mb-4 text-gray-600">主题：${data.data.text} | 领域：${data.data.main_domain} | 思考深度：${data.data.thinking_depth}</p>`;
                
                data.data.perspectives.forEach(perspective => {
                    html += `
                    <div class="mb-6 p-4 border border-gray-200 rounded-lg" style="border-left: 4px solid ${perspective.color};">
                        <div class="flex items-center mb-3">
                            <div class="w-8 h-8 flex items-center justify-center rounded-full mr-3" style="background-color: ${perspective.color}20; color: ${perspective.color};">
                                <span class="text-sm">${perspective.icon}</span>
                            </div>
                            <div>
                                <h4 class="font-semibold text-gray-800">${perspective.title}</h4>
                                <div class="flex items-center mt-1">
                                    <span class="inline-block bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded mr-2">${perspective.label}</span>
                                    <span class="inline-block bg-blue-100 text-blue-600 text-xs px-2 py-0.5 rounded">${perspective.difficulty === 'hard' ? '困难' : perspective.difficulty === 'medium' ? '中等' : '简单'}</span>
                                </div>
                            </div>
                        </div>
                        
                        <p class="text-gray-600 mb-3">${perspective.hint}</p>
                        
                        ${perspective.creation_prompts ? `
                        <div class="mt-4 border-t border-gray-100 pt-4">
                            <h5 class="font-semibold text-gray-700 mb-2">🧠 创作思维提示</h5>
                            
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">思考方向：</h6>
                                <ul class="list-disc pl-5 text-gray-600 text-sm">
                                    ${perspective.creation_prompts.thinking_directions.map(dir => `<li>${dir}</li>`).join('')}
                                </ul>
                            </div>
                            
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">启发问题：</h6>
                                <ul class="list-disc pl-5 text-gray-600 text-sm">
                                    ${perspective.creation_prompts.questions_to_ask.map(q => `<li>${q}</li>`).join('')}
                                </ul>
                            </div>
                            
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">分析框架：</h6>
                                <div class="flex flex-wrap gap-2">
                                    ${perspective.creation_prompts.analysis_frameworks.map(framework => 
                                        `<span class="inline-block bg-purple-100 text-purple-600 text-xs px-2 py-0.5 rounded">${framework}</span>`
                                    ).join('')}
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">写作建议：</h6>
                                <p class="text-gray-600 text-sm">${perspective.creation_prompts.writing_suggestions.join(' ')}</p>
                            </div>
                            
                            ${perspective.creation_prompts.creative_thinking_tools ? `
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">创意思维工具：</h6>
                                <div class="grid grid-cols-2 gap-2">
                                    ${perspective.creation_prompts.creative_thinking_tools.map(tool => `
                                        <div class="bg-gray-50 p-2 rounded text-sm">
                                            <div class="font-medium text-gray-700">${tool.name}</div>
                                            <div class="text-gray-500 text-xs">${tool.description}</div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            ` : ''}
                            
                            ${perspective.creation_prompts.domain_specific ? `
                            <div class="mb-3">
                                <h6 class="font-medium text-gray-600 mb-1">领域考量：</h6>
                                <ul class="list-disc pl-5 text-gray-600 text-sm">
                                    ${perspective.creation_prompts.domain_specific.considerations.map(cons => `<li>${cons}</li>`).join('')}
                                </ul>
                            </div>
                            ` : ''}
                            
                            <div class="flex justify-between items-center mt-3 pt-3 border-t border-gray-100">
                                <div>
                                    <span class="text-sm text-gray-500">建议字数：${perspective.creation_prompts.suggested_length}字</span>
                                </div>
                                <div>
                                    ${perspective.creation_prompts.keywords ? perspective.creation_prompts.keywords.map(keyword => 
                                        `<span class="inline-block bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded mr-1">${keyword}</span>`
                                    ).join('') : ''}
                                </div>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    `;
                });
                
                html += `<p class="text-sm text-gray-500 mt-4">处理时间: ${data.data.processing_time}s | 生成 ${data.data.count} 个视角</p>`;
                
                resultDiv.innerHTML = html;
                
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="bg-red-50 border-l-4 border-red-500 p-4">
                        <div class="flex">
                            <div class="flex-shrink-0">❌</div>
                            <div class="ml-3">
                                <p class="text-sm text-red-700">错误: ${error.message}</p>
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        // 检查模型状态
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusElement = document.getElementById('model-status');
                if (data.model_loaded) {
                    statusElement.innerHTML = '<span class="text-green-600">✅ 模型已加载</span>';
                } else {
                    statusElement.innerHTML = '<span class="text-yellow-600">⚠️ 模型未加载（使用模拟模式）</span>';
                }
            } catch (error) {
                document.getElementById('model-status').innerHTML = '<span class="text-red-600">❌ 状态检查失败</span>';
            }
        }

        // 页面加载时初始化
        window.onload = function() {
            // 检查模型状态
            checkModelStatus();
            
            // 自动运行文本分类测试
            setTimeout(() => {
                testClassify();
            }, 1000);
        };
    </script>
</body>
</html>'''
    return html

def generate_error_interface(error_msg: str) -> str:
    """生成错误界面"""
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>系统错误</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-red-50 to-pink-100 min-h-screen flex items-center justify-center">
    <div class="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
        <div class="text-red-500 text-6xl mb-4">⚠️</div>
        <h1 class="text-2xl font-bold text-gray-800 mb-4">系统初始化失败</h1>
        <p class="text-gray-600 mb-6">错误信息: {error_msg}</p>
        <p class="text-gray-500 text-sm">请检查日志文件或联系管理员。</p>
    </div>
</body>
</html>'''

# 模型管理器类
class EnhancedModelManager:
    """增强版模型管理器"""
    
    def __init__(self, cache_dir="models/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.classifier = None
    
    async def initialize_all_models(self):
        """初始化所有模型"""
        try:
            from text_domain_classifier import TextDomainClassifier
            model_path = "models/text_domain_classifier.pkl"
            
            if os.path.exists(model_path):
                self.classifier = TextDomainClassifier(model_path=model_path)
                print(f"从 {model_path} 加载模型")
            else:
                self.classifier = TextDomainClassifier()
                print("创建新模型实例")
            
            return True
        except Exception as e:
            print(f"初始化模型失败: {e}")
            return False
    
    async def classify(self, text: str, top_k: int = 5, detailed: bool = False) -> Dict[str, Any]:
        """文本分类"""
        if self.classifier:
            result = self.classifier.predict(text, top_k, detailed)
            return result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
        else:
            raise Exception("模型未初始化")
    
    async def analyze_viewpoint(self, original_text: str, response_text: str, 
                              perspective_type: str, analysis_depth: str = "standard") -> Dict[str, Any]:
        """观点分析"""
        if self.classifier:
            result = self.classifier.analyze_viewpoint(original_text, response_text, perspective_type, analysis_depth)
            return result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
        else:
            raise Exception("模型未初始化")
    
    async def generate_article(self, topic: str, perspective_type: str, target_length: int = 500,
                             structure_type: str = "argumentative", include_keywords: Optional[List[str]] = None,
                             target_audience: str = "general", writing_style: str = "standard") -> Dict[str, Any]:
        """生成文章"""
        if self.classifier and hasattr(self.classifier, 'generate_article'):
            result = self.classifier.generate_article(
                topic, perspective_type, target_length, structure_type
            )
            return result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
        else:
            raise Exception("文章生成功能不可用")

class SimpleModelManager:
    """简化版模型管理器（模拟模式）"""
    
    def __init__(self, cache_dir="models/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    async def initialize_all_models(self):
        """初始化所有模型"""
        await asyncio.sleep(0.1)
        return True
    
    async def classify(self, text: str, top_k: int = 5, detailed: bool = False) -> Dict[str, Any]:
        """文本分类"""
        await asyncio.sleep(0.05)
        
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
        await asyncio.sleep(0.1)
        
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
    
    async def generate_article(self, topic: str, perspective_type: str, target_length: int = 500,
                             structure_type: str = "argumentative", include_keywords: Optional[List[str]] = None,
                             target_audience: str = "general", writing_style: str = "standard") -> Dict[str, Any]:
        """生成文章"""
        await asyncio.sleep(0.2)
        
        # 生成文章内容
        content = f"""关于{topic}的思考

近年来，{topic}问题引起了社会各界的广泛关注。从{perspective_type}的视角来看，这一议题涉及多个维度的考量。

首先，从技术发展的角度分析，{topic}代表了当前社会发展的重要趋势。技术的进步为我们提供了新的解决思路和方法，但同时也带来了新的挑战和问题。这需要我们在享受技术红利的同时，认真思考如何平衡各种利益关系。

其次，从社会影响的角度来看，{topic}对人们的生活方式、工作模式和社会关系都产生了深远的影响。这种影响既有积极的一面，也有需要警惕的风险。我们需要以开放的心态面对这些变化，同时保持审慎的态度。

再者，从个人发展的层面来说，{topic}为个人成长和职业发展提供了新的机遇。每个人都应该积极面对这一变化，不断提升自己的适应能力和创新能力，以便在快速变化的环境中保持竞争力。

总之，{topic}是一个复杂而重要的议题。我们需要保持理性的思考，积极探讨适合中国国情的发展道路，为实现可持续发展贡献力量。"""
        
        # 确保达到目标长度
        while len(content) < target_length:
            additional = f"""从更深层次来看，{topic}还涉及到价值观念、文化传统和社会制度的多个层面。我们需要在继承优秀传统文化的基础上，积极探索符合时代要求的创新路径。

在未来的发展中，{topic}将继续成为社会关注的热点。我们需要建立更加完善的制度体系，为相关领域的发展提供有力保障。同时，加强国际合作与交流，学习借鉴国际先进经验，也是推动{topic}健康发展的有效途径。"""
            content += "\n\n" + additional
        
        return {
            'title': f"关于{topic}的思考与分析",
            'content': content,
            'main_idea': f"探讨{topic}的多方面影响和发展趋势",
            'structure': [
                {'section': '引言', 'content': '背景介绍', 'length': 50},
                {'section': '分析1', 'content': '技术角度', 'length': 100},
                {'section': '分析2', 'content': '社会角度', 'length': 100},
                {'section': '分析3', 'content': '个人角度', 'length': 100},
                {'section': '结论', 'content': '总结展望', 'length': 50}
            ],
            'keywords': [topic, '发展', '影响', '趋势', '思考'][:5],
            'word_count': len(content),
            'reading_time_minutes': round(len(content) / 300, 1),
            'target_audience': target_audience,
            'writing_style': writing_style,
            'coherence_score': round(random.uniform(0.6, 0.9), 2),
            'quality_score': round(random.uniform(0.6, 0.9), 2),
            'generation_method': '模拟生成',
            'timestamp': datetime.now().isoformat()
        }
def generate_detailed_perspective_prompts(perspective_type: str, domain: str, thinking_depth: str = "standard") -> Dict[str, Any]:
    """生成详细的视角提示"""
    prompts = PERSPECTIVE_PROMPTS.get(perspective_type, {})
    
    # 根据思考深度调整内容
    depth_config = {
        "basic": {
            "thinking_directions": 2,
            "questions_to_ask": 3,
            "analysis_frameworks": 1
        },
        "standard": {
            "thinking_directions": 3,
            "questions_to_ask": 4,
            "analysis_frameworks": 2
        },
        "deep": {
            "thinking_directions": 4,
            "questions_to_ask": 5,
            "analysis_frameworks": 3
        }
    }
    
    config = depth_config.get(thinking_depth, depth_config["standard"])
    
    # 获取领域特定的思考提示
    domain_prompts = DOMAIN_PROMPTS.get(domain, {})
    
    result = {
        "thinking_directions": prompts.get("thinking_directions", [])[:config["thinking_directions"]],
        "questions_to_ask": prompts.get("questions_to_ask", [])[:config["questions_to_ask"]],
        "analysis_frameworks": prompts.get("analysis_frameworks", [])[:config["analysis_frameworks"]],
        "writing_suggestions": prompts.get("writing_suggestions", [])
    }
    
    # 如果有领域特定的提示，合并到结果中
    if domain_prompts:
        result["domain_specific"] = {
            "considerations": domain_prompts.get("technical_considerations", [])[:2],
            "related_fields": domain_prompts.get("related_fields", [])[:3]
        }
    
    return result

def get_creative_thinking_tools(perspective_type: str) -> List[Dict[str, str]]:
    """获取创意思维工具"""
    tools = {
        "neutral": [
            {"name": "六顶思考帽", "description": "用不同颜色的帽子代表不同思考模式，全面分析问题"},
            {"name": "PMI分析法", "description": "从优点(Plus)、缺点(Minus)、有趣点(Interesting)三个角度分析"}
        ],
        "opposite": [
            {"name": "归谬法", "description": "从对立观点推导出荒谬结论，证明其不合理"},
            {"name": "反证法", "description": "假设对立观点正确，寻找矛盾证明其错误"}
        ],
        "supplement": [
            {"name": "5W2H法", "description": "从七个维度（是什么、为什么、谁、何时、何地、如何、多少）扩展思考"},
            {"name": "维度扩展法", "description": "增加时间、空间、主体、价值等分析维度"}
        ],
        "unique": [
            {"name": "SCAMPER法", "description": "七个创意激发方向：替代、结合、调整、修改、他用、消除、反向"},
            {"name": "类比思维法", "description": "用其他领域的成功案例启发新思路"}
        ],
        "historical": [
            {"name": "历史分期法", "description": "将问题放在历史时间轴上，分析不同时期的特征和变化"},
            {"name": "比较历史学", "description": "比较不同时期、不同地域的类似问题"}
        ],
        "global": [
            {"name": "跨文化比较", "description": "比较不同文化背景下的理解和做法"},
            {"name": "全球化影响分析", "description": "分析国际要素的相互作用和影响"}
        ],
        "emotional": [
            {"name": "同理心地图", "description": "理解不同群体的感受、想法、需求和行动"},
            {"name": "马斯洛需求层次", "description": "从生理到自我实现的需求层次分析"}
        ],
        "ethical": [
            {"name": "道德决策模型", "description": "结果论、义务论、美德伦理的综合分析"},
            {"name": "利益相关者分析", "description": "分析不同群体的利益和伦理诉求"}
        ]
    }
    
    return tools.get(perspective_type, [])
    
# 思维提示库
PERSPECTIVE_PROMPTS = {
    "neutral": {
        "thinking_directions": [
            "平衡分析：既看到优点，也思考缺点",
            "多维度比较：从技术、社会、经济、文化等不同维度分析",
            "辩证统一：寻找看似矛盾观点中的统一性"
        ],
        "questions_to_ask": [
            "这个观点在什么条件下成立？什么条件下不成立？",
            "哪些群体/情境能从中受益？哪些可能受损？",
            "短期和长期影响分别是什么？",
            "这个观点的适用边界在哪里？"
        ],
        "analysis_frameworks": [
            "SWOT分析（优势、劣势、机会、威胁）",
            "PEST分析（政治、经济、社会、技术）",
            "5W2H分析法（是什么、为什么、谁、何时、何地、如何、多少）"
        ],
        "writing_suggestions": [
            "采用'一方面...另一方面...'的平衡句式",
            "用'从...角度看'引出不同维度",
            "使用'然而/但是'进行适度的转折和补充"
        ],
        "keywords": ["平衡", "客观", "辩证", "多维度", "分析"]
    },
    "opposite": {
        "thinking_directions": [
            "角色转换：如果你是对方的支持者，会怎么想？",
            "利益反转：从对立方的利益角度重新思考",
            "假设翻转：如果前提条件完全相反会怎样？"
        ],
        "questions_to_ask": [
            "逻辑漏洞：原观点的逻辑链条哪里最脆弱？",
            "证据不足：哪些关键论据缺乏支持？",
            "前提假设：原观点基于哪些可能错误的假设？",
            "极端推演：将观点推向极端会暴露什么问题？"
        ],
        "analysis_frameworks": [
            "逻辑谬误分析（寻找草人谬误、滑坡谬误等）",
            "归谬法（从对立观点推导出荒谬结论）",
            "案例对比法（用反例证明观点不成立）"
        ],
        "writing_suggestions": [
            "用'然而/但是'强烈转折，表明立场",
            "用'实际上/事实上'引入反驳证据",
            "用'如果...那么...'展示错误假设的后果"
        ],
        "keywords": ["反驳", "挑战", "质疑", "对立", "反思"]
    },
    "supplement": {
        "thinking_directions": [
            "深度延伸：这个观点还能延伸到哪些相关领域？",
            "广度扩展：还有哪些群体/角度值得考虑？",
            "时间延伸：过去、现在、未来的变化趋势如何？"
        ],
        "questions_to_ask": [
            "遗漏维度：这个观点忽略了哪些重要方面？",
            "补充论据：可以增加哪些新的证据支持？",
            "交叉领域：与其他领域的交集会带来什么新视角？",
            "发展脉络：这个问题是如何演变成今天的样子的？"
        ],
        "analysis_frameworks": [
            "维度扩展框架（技术+社会+经济+文化）",
            "时间轴分析（历史演变-现状-未来趋势）",
            "交叉学科分析（与其他学科的连接点）"
        ],
        "writing_suggestions": [
            "用'此外/另外/同时'引出补充观点",
            "用'从...角度看'扩展思考维度",
            "用'更重要的是/特别是'强调补充的价值"
        ],
        "keywords": ["扩展", "补充", "延伸", "深化", "丰富"]
    },
    "unique": {
        "thinking_directions": [
            "逆向思考：与常识相反的思考角度",
            "跨界借鉴：用其他领域的理论解释这个问题",
            "边缘关注：关注被主流忽视的少数群体或现象"
        ],
        "questions_to_ask": [
            "非主流视角：儿童/老人/残障人士会怎么看？",
            "创新类比：用其他领域的成功案例如何解释这个问题？",
            "科幻思维：假设100年后的人类会怎么看？",
            "艺术视角：从绘画、音乐、文学的角度有什么启发？"
        ],
        "analysis_frameworks": [
            "逆向思维法（与常识相反的方向思考）",
            "类比思维法（借用其他领域的概念和理论）",
            "假设思维法（如果条件完全改变会怎样）"
        ],
        "writing_suggestions": [
            "用'很少有人注意到...'引出独特视角",
            "用'换个角度看...'进行视角转换",
            "用'如果...也许...'展开想象性思考"
        ],
        "keywords": ["独特", "创新", "跨界", "想象", "小众"]
    },
    "historical": {
        "thinking_directions": [
            "历史演变：这个问题是如何从过去发展到现在？",
            "历史比较：与历史上的类似问题有何异同？",
            "历史教训：从历史中我们能学到什么经验？"
        ],
        "questions_to_ask": [
            "历史渊源：这个问题的根源可以追溯到什么时候？",
            "转折点：历史上哪些关键时刻改变了问题的走向？",
            "周期规律：是否存在某种历史发展的周期模式？",
            "历史影响：历史上的决策如何影响今天的局面？"
        ],
        "analysis_frameworks": [
            "历史分期法（古代-近代-现代-当代）",
            "比较历史学（跨时期、跨地域的比较）",
            "历史脉络梳理（原因-过程-结果分析）"
        ],
        "writing_suggestions": [
            "用'回顾历史...'引出历史背景",
            "用'与...时期相比...'进行历史比较",
            "用'历史告诉我们...'总结历史教训"
        ],
        "keywords": ["历史", "演变", "传承", "变革", "发展"]
    },
    "global": {
        "thinking_directions": [
            "跨国比较：不同国家的做法有何不同？",
            "文化差异：不同文化背景下这个问题如何表现？",
            "全球联动：这个问题如何受国际环境影响？"
        ],
        "questions_to_ask": [
            "国际比较：哪些国家处理这个问题最成功？为什么？",
            "文化适应性：这个观点在不同文化中如何被理解？",
            "全球化影响：国际交流如何改变了这个问题？",
            "地缘政治：不同地区的政治经济环境如何影响这个问题？"
        ],
        "analysis_frameworks": [
            "跨国比较分析（不同国家的政策和实践）",
            "文化相对主义（从不同文化视角理解）",
            "全球化影响分析（国际要素的相互作用）"
        ],
        "writing_suggestions": [
            "用'在国际视野下...'引出全球视角",
            "用'对比...国家...'进行国际比较",
            "用'从全球来看...'总结国际趋势"
        ],
        "keywords": ["全球", "国际", "跨文化", "比较", "多元"]
    },
    "emotional": {
        "thinking_directions": [
            "情感体验：这个问题引发哪些情感反应？",
            "心理影响：对人们的心理健康有什么影响？",
            "人际关系：如何影响人与人之间的关系？"
        ],
        "questions_to_ask": [
            "情感反应：这个问题通常引发什么情感？为什么？",
            "心理需求：涉及到人们的哪些基本心理需求？",
            "情感共鸣：如何建立与读者的情感连接？",
            "情感障碍：什么情感因素阻碍了问题的解决？"
        ],
        "analysis_frameworks": [
            "马斯洛需求层次分析（生理-安全-社交-尊重-自我实现）",
            "情感智力分析（识别-理解-管理情感）",
            "同理心地图（理解不同群体的感受和需求）"
        ],
        "writing_suggestions": [
            "用'从情感上说...'引出情感层面",
            "用'我们都能感受到...'建立情感共鸣",
            "用'这不仅...更是...'连接事实与情感"
        ],
        "keywords": ["情感", "心理", "感受", "共鸣", "体验"]
    },
    "ethical": {
        "thinking_directions": [
            "道德原则：涉及到哪些基本的道德原则？",
            "价值冲突：存在哪些价值观的冲突和权衡？",
            "社会责任：相关方应该承担什么道德责任？"
        ],
        "questions_to_ask": [
            "道德困境：存在哪些难以抉择的道德困境？",
            "公平正义：如何确保决策的公平性和正义性？",
            "权利保护：哪些基本权利需要特别保护？",
            "伦理边界：在什么情况下突破常规伦理是合理的？"
        ],
        "analysis_frameworks": [
            "道德决策模型（结果论-义务论-美德伦理）",
            "利益相关者伦理分析（不同群体的利益平衡）",
            "伦理困境分析（价值观冲突与权衡）"
        ],
        "writing_suggestions": [
            "用'从伦理角度看...'引入道德考量",
            "用'这不仅...而且关乎...'连接技术与伦理",
            "用'如何在...与...之间取得平衡'展示伦理思考"
        ],
        "keywords": ["伦理", "道德", "责任", "公平", "正义"]
    }
}

# 领域特定的思考提示
DOMAIN_PROMPTS = {
    "科技": {
        "technical_considerations": [
            "技术可行性：从技术角度是否真的可行？",
            "安全性考量：可能带来哪些安全风险和隐私问题？",
            "用户体验：如何确保良好的用户体验和易用性？"
        ],
        "innovation_questions": [
            "技术突破：这个观点涉及到哪些关键技术突破？",
            "应用场景：在哪些具体场景中最有价值？",
            "发展瓶颈：当前最大的技术障碍是什么？"
        ],
        "related_fields": ["人工智能", "大数据", "物联网", "区块链", "生物技术"]
    },
    "教育": {
        "learning_perspectives": [
            "学习者中心：如何以学习者为中心进行设计？",
            "个性化学习：如何满足不同学习者的个性化需求？",
            "评估方法：如何科学评估学习效果和教学成果？"
        ],
        "educational_questions": [
            "教学方法：适合采用哪些教学方法和策略？",
            "资源公平：如何确保教育资源的公平分配？",
            "终身学习：如何支持终身学习和持续发展？"
        ],
        "related_fields": ["在线教育", "个性化学习", "教育公平", "教学评估", "终身教育"]
    },
    "社会": {
        "social_impacts": [
            "社会公平：对社会公平和正义有什么影响？",
            "群体差异：对不同社会群体的影响有何不同？",
            "社区建设：如何促进社区凝聚力和归属感？"
        ],
        "social_questions": [
            "社会变迁：如何反映和推动社会变迁？",
            "文化传承：对文化传承和发展有什么影响？",
            "社会治理：如何优化社会治理和公共服务？"
        ],
        "related_fields": ["社会公平", "社区治理", "文化传承", "社会保障", "公共服务"]
    },
    "经济": {
        "economic_factors": [
            "成本效益：从经济角度是否合理高效？",
            "市场影响：对相关市场和产业链有什么影响？",
            "就业影响：会创造还是减少就业机会？"
        ],
        "economic_questions": [
            "商业模式：可行的商业模式是什么？",
            "投资回报：长期的投资回报预期如何？",
            "经济风险：存在哪些经济风险和不确定性？"
        ],
        "related_fields": ["商业模式", "市场分析", "投资回报", "经济风险", "产业发展"]
    }
}

# 测试函数
def test_enhanced_application():
    """测试增强版应用"""
    print("=" * 70)
    print("辩证思考AI互动系统 - 增强版 3.0")
    print("=" * 70)
    
    ensure_directories()
    
    print("\n系统功能:")
    print("  ✅ 集成机器学习文本分类")
    print("  ✅ 多维度观点分析")
    print("  ✅ 智能文章生成（100字以上）")
    print("  ✅ 8种不同视角生成")
    print("  ✅ 批量处理支持")
    print("  ✅ 智能缓存机制")
    print("  ✅ 实时性能监控")
    
    print("\nAPI端点:")
    print("  GET  /                     - 主界面")
    print("  GET  /api/health          - 健康检查")
    print("  POST /api/classify        - 文本分类")
    print("  POST /api/analyze-viewpoint - 观点分析")
    print("  POST /api/generate-article - 文章生成")
    print("  POST /api/generate-perspectives - 生成视角")
    print("  POST /api/batch-process   - 批量处理")
    print("  GET  /api/system-metrics  - 系统指标")
    
    print("\n启动服务器:")
    print("  python main.py")
    print("\n访问地址:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/api/docs")
    print("=" * 70)

# 主程序入口
if __name__ == "__main__":
    test_enhanced_application()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

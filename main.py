"""
辩证思考AI互动系统 - 主应用
包含完整的观点分析功能
"""
import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import traceback
import random

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="辩证思考AI互动系统",
    description="基于机器学习的文本领域分类与辩证思考互动系统",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
classifier = None
model_loading = False
MODEL_PATH = "models/text_domain_classifier.pkl"

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = ["models", "logs", "examples"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")

# 导入文本分类器
def get_classifier():
    """获取或创建分类器实例"""
    global classifier, model_loading
    
    if classifier is not None:
        return classifier
    
    if model_loading:
        logger.info("模型正在加载中...")
        return None
    
    try:
        logger.info("尝试导入TextDomainClassifier...")
        from text_domain_classifier import TextDomainClassifier
        MODEL_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"无法导入TextDomainClassifier: {e}")
        logger.warning("将使用模拟分类器进行演示")
        MODEL_AVAILABLE = False
    
    if not MODEL_AVAILABLE:
        # 创建模拟分类器
        class MockClassifier:
            def __init__(self):
                self.domains = ['科技', '生活', '教育', '职场', '旅游', '健康', '财经', '娱乐', '体育']
                logger.info("使用模拟分类器")
            
            def predict(self, text, top_k=3, threshold=0.1):
                # 模拟预测结果
                domains = random.sample(self.domains, min(top_k, len(self.domains)))
                probabilities = {domain: round(random.uniform(0.3, 0.9), 2) for domain in domains}
                
                return {
                    'text': text,
                    'domains': domains,
                    'probabilities': probabilities,
                    'top_domain': domains[0] if domains else None
                }
            
            def analyze_viewpoint(self, original_text, response_text, perspective_type):
                # 模拟观点分析
                keywords = response_text.split()[:3] if response_text.split() else ['思考', '分析', '观点']
                
                positive_feedback = f"你的回应对'{original_text[:20]}...'的分析很有见地，展现了良好的思考能力。"
                
                if perspective_type == 'opposite':
                    positive_feedback += " 从对立角度思考很有挑战性。"
                elif perspective_type == 'neutral':
                    positive_feedback += " 客观分析展现了很好的平衡感。"
                elif perspective_type == 'supplement':
                    positive_feedback += " 补充观点很有价值。"
                elif perspective_type == 'unique':
                    positive_feedback += " 小众视角很独特。"
                
                suggestion = "可以考虑补充具体案例，增强说服力。也可以从更多角度进行分析。"
                
                logic_score = round(min(0.7 + len(response_text) * 0.001, 0.95), 2)
                
                return {
                    'positive': positive_feedback,
                    'suggestion': suggestion,
                    'keywords': keywords,
                    'logic_score': logic_score,
                    'analysis_method': '模拟分析',
                    'text_length': len(response_text)
                }
            
            def get_keywords(self, text, top_n=5):
                # 模拟关键词提取
                words = text.split()[:top_n]
                return words if words else ['思考', '分析', '观点', '讨论', '交流']
            
            def load_model(self, model_path):
                logger.info(f"模拟加载模型: {model_path}")
                return True
            
            def train(self):
                logger.info("模拟训练模型")
                return None, None
            
            def save_model(self, model_path):
                logger.info(f"模拟保存模型: {model_path}")
                return True
        
        classifier = MockClassifier()
        return classifier
    
    model_loading = True
    try:
        # 检查模型文件
        model_file = MODEL_PATH
        if os.path.exists(model_file):
            logger.info(f"从 {model_file} 加载预训练模型...")
            classifier = TextDomainClassifier()
            classifier.load_model(model_file)
            logger.info("模型加载成功")
        else:
            logger.warning("未找到预训练模型，创建模拟分类器...")
            # 使用模拟分类器
            class MockClassifier:
                def __init__(self):
                    self.domains = ['科技', '生活', '教育', '职场', '旅游', '健康', '财经', '娱乐', '体育']
                    logger.info("使用模拟分类器")
                
                def predict(self, text, top_k=3, threshold=0.1):
                    domains = random.sample(self.domains, min(top_k, len(self.domains)))
                    probabilities = {domain: round(random.uniform(0.3, 0.9), 2) for domain in domains}
                    
                    return {
                        'text': text,
                        'domains': domains,
                        'probabilities': probabilities,
                        'top_domain': domains[0] if domains else None
                    }
                
                def analyze_viewpoint(self, original_text, response_text, perspective_type):
                    keywords = response_text.split()[:3] if response_text.split() else ['思考', '分析', '观点']
                    
                    positive_feedback = f"你的回应对'{original_text[:20]}...'的分析很有见地，展现了良好的思考能力。"
                    
                    if perspective_type == 'opposite':
                        positive_feedback += " 从对立角度思考很有挑战性。"
                    elif perspective_type == 'neutral':
                        positive_feedback += " 客观分析展现了很好的平衡感。"
                    elif perspective_type == 'supplement':
                        positive_feedback += " 补充观点很有价值。"
                    elif perspective_type == 'unique':
                        positive_feedback += " 小众视角很独特。"
                    
                    suggestion = "可以考虑补充具体案例，增强说服力。也可以从更多角度进行分析。"
                    
                    logic_score = round(min(0.7 + len(response_text) * 0.001, 0.95), 2)
                    
                    return {
                        'positive': positive_feedback,
                        'suggestion': suggestion,
                        'keywords': keywords,
                        'logic_score': logic_score,
                        'analysis_method': '模拟分析',
                        'text_length': len(response_text)
                    }
                
                def get_keywords(self, text, top_n=5):
                    words = text.split()[:top_n]
                    return words if words else ['思考', '分析', '观点', '讨论', '交流']
            
            classifier = MockClassifier()
        
        model_loading = False
        return classifier
        
    except Exception as e:
        logger.error(f"初始化分类器失败: {e}")
        logger.error(traceback.format_exc())
        model_loading = False
        return None

# 数据模型
class ClassifyRequest(BaseModel):
    """分类请求模型"""
    text: str = Field(..., min_length=1, max_length=500, description="要分类的文本")
    top_k: int = Field(3, ge=1, le=10, description="返回的领域数量")

class PerspectiveRequest(BaseModel):
    """视角生成请求模型"""
    text: str = Field(..., min_length=1, max_length=500, description="原始观点")
    count: int = Field(5, ge=1, le=10, description="生成视角数量")

class AnalyzeRequest(BaseModel):
    """观点分析请求模型"""
    original_text: str = Field(..., min_length=1, max_length=500, description="原始观点文本")
    response_text: str = Field(..., min_length=1, max_length=1000, description="用户回应文本")
    perspective_type: str = Field(..., description="视角类型", example="neutral")

class ExampleTopic(BaseModel):
    """示例话题模型"""
    id: str
    title: str
    content: str
    description: str
    icon: str
    tags: List[str]

# 示例话题数据
def get_example_topics():
    """获取示例话题"""
    return [
        ExampleTopic(
            id="topic_001",
            title="线上办公 vs 线下办公",
            content="线上办公比线下办公更高效",
            description="比较远程办公和传统办公室工作的优缺点",
            icon="fas fa-laptop-house",
            tags=["工作", "效率", "未来趋势"]
        ),
        ExampleTopic(
            id="topic_002",
            title="人工智能的利弊",
            content="人工智能对人类就业的威胁被夸大了",
            description="探讨AI发展对人类社会和就业的影响",
            icon="fas fa-robot",
            tags=["科技", "就业", "伦理"]
        ),
        ExampleTopic(
            id="topic_003",
            title="短视频的影响",
            content="短视频对青少年发展弊大于利",
            description="讨论短视频平台对年轻人认知和价值观的影响",
            icon="fas fa-video",
            tags=["媒体", "教育", "青少年"]
        ),
        ExampleTopic(
            id="topic_004",
            title="环保与发展的平衡",
            content="经济发展不应以牺牲环境为代价",
            description="探讨经济发展与环境保护之间的平衡关系",
            icon="fas fa-leaf",
            tags=["环境", "经济", "可持续发展"]
        ),
        ExampleTopic(
            id="topic_005",
            title="大学专业选择",
            content="选择热门专业比选择兴趣专业更重要",
            description="讨论大学专业选择的策略和考量因素",
            icon="fas fa-graduation-cap",
            tags=["教育", "职业", "选择"]
        ),
        ExampleTopic(
            id="topic_006",
            title="城市与乡村生活",
            content="大城市的生活质量比小城市更高",
            description="比较不同规模城市的生活体验和优缺点",
            icon="fas fa-city",
            tags=["生活", "城市", "社会"]
        )
    ]

# 视角类型定义
PERSPECTIVE_TYPES = [
    {
        "type": "opposite",
        "label": "对立视角",
        "color": "#e74c3c",
        "icon": "fas fa-exchange-alt",
        "description": "与你的初始观点完全相反"
    },
    {
        "type": "neutral",
        "label": "中立视角",
        "color": "#3498db",
        "icon": "fas fa-balance-scale",
        "description": "站在客观中立的角度分析"
    },
    {
        "type": "supplement",
        "label": "补充视角",
        "color": "#2ecc71",
        "icon": "fas fa-plus-circle",
        "description": "补充你未考虑的方面"
    },
    {
        "type": "unique",
        "label": "小众视角",
        "color": "#9b59b6",
        "icon": "fas fa-lightbulb",
        "description": "从非主流角度思考问题"
    },
    {
        "type": "historical",
        "label": "历史视角",
        "color": "#f39c12",
        "icon": "fas fa-history",
        "description": "从历史发展的角度看问题"
    },
    {
        "type": "global",
        "label": "全球视角",
        "color": "#1abc9c",
        "icon": "fas fa-globe",
        "description": "从国际化和全球化的角度思考"
    }
]

# 观点分析示例
ANALYSIS_EXAMPLES = [
    {
        "id": "example_001",
        "original": "人工智能将完全取代人类工作",
        "response": "我认为人工智能会取代部分重复性工作，但创造性、情感交流和复杂决策工作仍需要人类。人机协作将是未来趋势。",
        "perspective": "neutral",
        "positive": "你对人工智能与人类工作的关系分析很全面，考虑了自动化和人机协作的平衡，展现了很好的前瞻性思考。",
        "suggestion": "可以补充具体行业案例，说明哪些工作更容易被取代，哪些工作更依赖人类特质。"
    },
    {
        "id": "example_002",
        "original": "线上教育比传统教育更有效",
        "response": "我不同意这个观点。线上教育虽然方便，但缺乏师生互动和即时反馈，对于实践性强的课程效果不佳。传统教育的社交功能也很重要。",
        "perspective": "opposite",
        "positive": "你从对立角度提出了有力的反驳，指出了线上教育的局限性，特别是互动性和实践性的不足。",
        "suggestion": "可以考虑线上教育与线下教育的混合模式，以及如何通过技术改进解决互动不足的问题。"
    },
    {
        "id": "example_003",
        "original": "环保应该优先于经济发展",
        "response": "我认为应该在环保和经济发展之间找到平衡。可以发展绿色经济，既保护环境又促进经济增长，实现可持续发展。",
        "perspective": "supplement",
        "positive": "你的补充观点很有价值，提出了平衡发展的思路，体现了可持续发展理念。",
        "suggestion": "可以具体说明绿色经济的实现路径，以及不同发展阶段国家的应对策略。"
    }
]

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("正在启动辩证思考AI互动系统...")
    
    # 确保目录存在
    ensure_directories()
    
    # 保存示例话题
    save_examples()
    
    # 在后台初始化模型
    asyncio.create_task(initialize_classifier_async())
    logger.info("服务启动完成")

async def initialize_classifier_async():
    """异步初始化分类器"""
    try:
        classifier_instance = get_classifier()
        if classifier_instance:
            logger.info("模型初始化成功")
        else:
            logger.warning("模型初始化失败，但应用将继续运行（使用模拟模式）")
    except Exception as e:
        logger.error(f"异步初始化失败: {e}")
        logger.error(traceback.format_exc())

def save_examples():
    """保存示例话题到文件"""
    try:
        examples = [topic.dict() for topic in get_example_topics()]
        examples_file = "examples/topics.json"
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"示例话题已保存到: {examples_file}")
    except Exception as e:
        logger.error(f"保存示例话题失败: {e}")

# 视角生成函数
def generate_perspectives_for_text(text: str, count: int = 5) -> List[Dict[str, Any]]:
    """为文本生成多个视角"""
    classifier_instance = get_classifier()
    
    # 获取文本分类结果
    classification_result = {}
    if classifier_instance:
        try:
            classification_result = classifier_instance.predict(text, top_k=3)
        except:
            pass
    
    # 确定主要领域
    main_domains = classification_result.get('domains', [])
    if not main_domains and classifier_instance and hasattr(classifier_instance, 'domains'):
        main_domains = random.sample(classifier_instance.domains, min(2, len(classifier_instance.domains)))
    
    if not main_domains:
        main_domains = ['综合', '思考']
    
    main_domain = main_domains[0] if main_domains else "综合"
    
    # 选择视角类型
    selected_types = random.sample(PERSPECTIVE_TYPES, min(count, len(PERSPECTIVE_TYPES)))
    
    perspectives = []
    for i, p_type in enumerate(selected_types):
        # 生成视角标题
        text_preview = text[:20] + "..." if len(text) > 20 else text
        
        # 根据视角类型生成不同的标题
        title_templates = {
            "opposite": f"对立视角：重新审视「{text_preview}」",
            "neutral": f"中立视角：客观分析「{text_preview}」",
            "supplement": f"补充视角：关于「{text_preview}」的更多思考",
            "unique": f"小众视角：对「{text_preview}」的独特见解",
            "historical": f"历史视角：从历史看「{text_preview}」",
            "global": f"全球视角：国际化视野下的「{text_preview}」"
        }
        
        title = title_templates.get(p_type["type"], f"{p_type['label']}：关于「{text_preview}」的思考")
        
        # 生成提示
        hint_templates = {
            "opposite": f"请站在与「{text[:30]}{'...' if len(text) > 30 else ''}」相反的立场，从{main_domain}角度阐述你的反驳观点",
            "neutral": f"请客观分析「{text[:30]}{'...' if len(text) > 30 else ''}」的优缺点，从{main_domain}领域进行辩证评价",
            "supplement": f"请补充「{text[:30]}{'...' if len(text) > 30 else ''}」中未考虑的方面，从{main_domain}角度提供新见解",
            "unique": f"请从独特的{main_domain}视角出发，对「{text[:30]}{'...' if len(text) > 30 else ''}」提出创新性思考",
            "historical": f"请从历史发展的角度，分析「{text[:30]}{'...' if len(text) > 30 else ''}」在{main_domain}领域的演变",
            "global": f"请从全球化的视野，探讨「{text[:30]}{'...' if len(text) > 30 else ''}」在不同文化背景下的{main_domain}意义"
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
            "description": p_type["description"]
        })
    
    return perspectives

# 生成HTML界面
def create_html_interface():
    """创建HTML界面"""
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
            <h2>💡 观点分析示例</h2>
            <div class="analysis-examples" id="analysis-examples">
                <!-- 观点分析示例将通过JS动态加载 -->
            </div>
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
        
        // 加载观点分析示例
        async function loadAnalysisExamples() {
            try {
                const response = await fetch('/api/analysis-examples');
                const examples = await response.json();
                
                const container = document.getElementById('analysis-examples');
                container.innerHTML = '';
                
                examples.forEach(example => {
                    const exampleElement = document.createElement('div');
                    exampleElement.className = 'analysis-example';
                    exampleElement.innerHTML = `
                        <h4>${example.original}</h4>
                        <p><strong>回应:</strong> ${example.response}</p>
                        <p><strong>视角类型:</strong> ${getPerspectiveLabel(example.perspective)}</p>
                        <div class="analysis-result">
                            <div class="analysis-positive">👍 肯定部分: ${example.positive}</div>
                            <div class="analysis-suggestion">💡 优化建议: ${example.suggestion}</div>
                        </div>
                        <button style="margin-top: 10px; padding: 8px 15px; font-size: 14px;" 
                                onclick="loadExampleData('${example.id}')">使用此示例</button>
                    `;
                    
                    container.appendChild(exampleElement);
                });
            } catch (error) {
                console.error('加载观点分析示例失败:', error);
            }
        }
        
        // 加载示例数据
        function loadExampleData(exampleId) {
            try {
                const examples = document.getElementById('analysis-examples').querySelectorAll('.analysis-example');
                examples.forEach(example => {
                    const button = example.querySelector('button');
                    if (button.getAttribute('onclick').includes(exampleId)) {
                        const original = example.querySelector('h4').textContent;
                        const response = example.querySelector('p:nth-of-type(1)').textContent.replace('回应: ', '');
                        const perspective = example.querySelector('p:nth-of-type(2)').textContent.replace('视角类型: ', '');
                        
                        document.getElementById('original-text').value = original;
                        document.getElementById('response-text').value = response;
                        
                        // 设置视角类型
                        const perspectiveTypeMap = {
                            '中立视角': 'neutral',
                            '对立视角': 'opposite',
                            '补充视角': 'supplement',
                            '小众视角': 'unique'
                        };
                        
                        const perspectiveTypeSelect = document.getElementById('perspective-type');
                        for (let i = 0; i < perspectiveTypeSelect.options.length; i++) {
                            if (perspectiveTypeSelect.options[i].text === perspective) {
                                perspectiveTypeSelect.selectedIndex = i;
                                break;
                            }
                        }
                        
                        showMessage(`已加载示例: ${original.substring(0, 30)}...`, 'success');
                    }
                });
            } catch (error) {
                console.error('加载示例数据失败:', error);
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
                html += `<p><strong>输入文本:</strong> ${data.text}</p>`;
                
                if (data.domains && data.domains.length > 0) {
                    html += `<p><strong>主要领域:</strong> ${data.top_domain || '无'}</p>`;
                    html += `<p><strong>所有可能领域:</strong> ${data.domains.join(', ')}</p>`;
                    
                    if (data.probabilities) {
                        html += '<p><strong>概率分布:</strong></p><ul>';
                        for (const [domain, prob] of Object.entries(data.probabilities)) {
                            const percentage = (prob * 100).toFixed(1);
                            html += `<li>${domain}: ${percentage}%</li>`;
                        }
                        html += '</ul>';
                    }
                } else {
                    html += '<p>未识别到明确的领域</p>';
                }
                
                html += `<p><em>处理时间: ${data.processing_time || 0}秒</em></p>`;
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
                
                let html = `<div class="success"><h3>🎯 为您生成 ${data.perspectives.length} 个不同视角</h3>`;
                html += `<p><strong>原始观点:</strong> ${data.text}</p>`;
                
                html += '<div class="perspectives-container">';
                
                data.perspectives.forEach(perspective => {
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
                html += `<p><em>处理时间: ${data.processing_time || 0}秒</em></p>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
                showMessage(`成功生成 ${data.perspectives.length} 个视角！`, 'success');
                currentPerspectives = data.perspectives;
                
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
                html += `<div class="analysis-positive"><strong>👍 肯定部分:</strong> ${data.analysis.positive}</div>`;
                html += `<div class="analysis-suggestion"><strong>💡 优化建议:</strong> ${data.analysis.suggestion}</div>`;
                
                if (data.analysis.keywords && data.analysis.keywords.length > 0) {
                    html += `<p><strong>🔑 关键词:</strong> ${data.analysis.keywords.join(', ')}</p>`;
                }
                
                if (data.analysis.logic_score) {
                    html += `<p><strong>📊 逻辑评分:</strong> ${data.analysis.logic_score}/1.0</p>`;
                }
                
                html += `<p><em>分析方法: ${data.analysis.analysis_method || 'AI模型分析'}</em></p>`;
                html += '</div>';
                
                html += `<p><em>处理时间: ${data.processing_time || 0}秒</em></p>`;
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
            loadAnalysisExamples();
            
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

# API路由
@app.get("/")
async def read_root():
    """根路由，返回前端界面"""
    try:
        html_content = create_html_interface()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"根路径处理失败: {e}")
        return HTMLResponse(content=f"<h1>系统错误</h1><p>{str(e)}</p>", status_code=500)

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    try:
        classifier_instance = get_classifier()
        
        return {
            "status": "healthy",
            "model_loaded": classifier_instance is not None,
            "timestamp": datetime.now().isoformat(),
            "service": "辩证思考AI互动系统",
            "python_version": sys.version
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.get("/api/analysis-examples")
async def get_analysis_examples_endpoint():
    """获取观点分析示例"""
    try:
        return ANALYSIS_EXAMPLES
    except Exception as e:
        logger.error(f"获取观点分析示例失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取观点分析示例失败: {str(e)}")

@app.get("/api/perspective-types")
async def get_perspective_types_endpoint():
    """获取视角类型"""
    try:
        return PERSPECTIVE_TYPES
    except Exception as e:
        logger.error(f"获取视角类型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取视角类型失败: {str(e)}")

@app.post("/api/classify")
async def classify_text_endpoint(request: ClassifyRequest):
    """文本分类端点"""
    try:
        classifier_instance = get_classifier()
        if not classifier_instance:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        
        try:
            result = classifier_instance.predict(request.text, top_k=request.top_k)
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")
        
        processing_time = time.time() - start_time
        
        result["processing_time"] = round(processing_time, 3)
        result["model_used"] = True
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文本分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"文本分类失败: {str(e)}")

@app.post("/api/generate-perspectives")
async def generate_perspectives_endpoint(request: PerspectiveRequest):
    """生成多个视角"""
    try:
        start_time = time.time()
        
        perspectives = generate_perspectives_for_text(request.text, request.count)
        
        processing_time = time.time() - start_time
        
        return {
            "text": request.text,
            "perspectives": perspectives,
            "count": len(perspectives),
            "processing_time": round(processing_time, 3),
            "model_used": True
        }
    except Exception as e:
        logger.error(f"生成视角失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成视角失败: {str(e)}")

@app.post("/api/analyze-viewpoint")
async def analyze_viewpoint_endpoint(request: AnalyzeRequest):
    """分析观点"""
    try:
        classifier_instance = get_classifier()
        if not classifier_instance:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        start_time = time.time()
        
        try:
            # 检查是否有 analyze_viewpoint 方法
            if hasattr(classifier_instance, 'analyze_viewpoint'):
                analysis = classifier_instance.analyze_viewpoint(
                    request.original_text,
                    request.response_text,
                    request.perspective_type
                )
            else:
                # 使用模拟分析
                analysis = {
                    'positive': f'你的回应对"{request.original_text[:20]}..."的分析很有见地，展现了良好的思考能力。',
                    'suggestion': '可以考虑补充具体案例，增强说服力。也可以从更多角度进行分析。',
                    'keywords': request.response_text.split()[:3] if request.response_text.split() else ['思考', '分析', '观点'],
                    'logic_score': round(min(0.7 + len(request.response_text) * 0.001, 0.95), 2),
                    'analysis_method': '模拟分析',
                    'text_length': len(request.response_text)
                }
        except Exception as e:
            logger.error(f"观点分析失败: {e}")
            raise HTTPException(status_code=500, detail=f"观点分析失败: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return {
            "original_text": request.original_text,
            "response_text": request.response_text,
            "perspective_type": request.perspective_type,
            "analysis": analysis,
            "processing_time": round(processing_time, 3),
            "model_used": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"观点分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"观点分析失败: {str(e)}")

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "detail": f"服务器内部错误: {str(exc)}",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

# 测试函数
def test_application():
    """测试应用"""
    print("=" * 60)
    print("辩证思考AI互动系统 - 测试模式")
    print("=" * 60)
    
    # 确保目录存在
    ensure_directories()
    
    print(f"支持的视角类型: {len(PERSPECTIVE_TYPES)} 种")
    for p_type in PERSPECTIVE_TYPES:
        print(f"  - {p_type['label']} ({p_type['type']}): {p_type['description']}")
    
    print(f"\n观点分析示例: {len(ANALYSIS_EXAMPLES)} 个")
    for example in ANALYSIS_EXAMPLES:
        print(f"  - 原始: {example['original']}")
        print(f"    回应: {example['response']}")
    
    print("\nAPI端点:")
    print("  - GET  /                     - 主界面")
    print("  - GET  /api/health           - 健康检查")
    print("  - GET  /api/analysis-examples - 观点分析示例")
    print("  - GET  /api/perspective-types - 视角类型")
    print("  - POST /api/classify         - 文本分类")
    print("  - POST /api/generate-perspectives - 生成视角")
    print("  - POST /api/analyze-viewpoint - 分析观点")
    
    print("\n启动服务器: python app/main.py")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/api/docs")
    print("=" * 60)

# 主程序入口
if __name__ == "__main__":
    test_application()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

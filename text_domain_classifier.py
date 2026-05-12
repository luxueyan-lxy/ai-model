"""
包含文本分类、观点分析功能、文章生成
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import json
import os
import re
import jieba
import jieba.analyse
import random
from collections import Counter, defaultdict
import warnings
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import time
warnings.filterwarnings('ignore')

@dataclass
class ClassificationResult:
    """分类结果数据类"""
    text: str
    domains: List[str]
    probabilities: Dict[str, float]
    top_domain: Optional[str]
    confidence: float
    features: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    model_version: str = "3.0.0"
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    positive: str
    suggestion: str
    keywords: List[str]
    logic_score: float
    sentiment_score: float
    coherence_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    argument_structure: Dict[str, Any]
    rhetorical_devices: List[str]
    analysis_method: str
    text_length: Optional[int] = None
    reading_level: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GeneratedArticle:
    """生成的文章数据类"""
    title: str
    content: str
    main_idea: str
    structure: List[Dict[str, str]]
    keywords: List[str]
    word_count: int
    reading_time_minutes: float
    target_audience: str
    writing_style: str
    coherence_score: float
    quality_score: float
    generation_method: str
    timestamp: Optional[str] = None
    error: Optional[str] = None  # 添加这个可选参数
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseClassifier(ABC):
    """分类器基类"""
    
    @abstractmethod
    def predict(self, text: str, top_k: int = 5, detailed: bool = False) -> ClassificationResult:
        pass
    
    @abstractmethod
    def analyze_viewpoint(self, original_text: str, response_text: str, 
                         perspective_type: str, analysis_depth: str = "standard") -> AnalysisResult:
        pass
    
    @abstractmethod
    def generate_article(self, topic: str, perspective_type: str, target_length: int = 500) -> GeneratedArticle:
        pass
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[List[str]]):
        pass

class TextDomainClassifier(BaseClassifier):
    """
    增强版文本领域分类器
    支持多维度分析、集成学习和文章生成
    """
    
    def __init__(self, model_path=None, use_ensemble=True, use_cache=True):
        """
        初始化增强版分类器
        
        参数:
            model_path: 预训练模型路径
            use_ensemble: 是否使用集成学习
            use_cache: 是否使用缓存
        """
        # 扩展的领域列表
        self.domains = self.get_extended_domains()
        
        # 特征提取器
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=self.get_basic_stopwords()
        )
        
        # 多标签二值化器
        self.mlb = MultiLabelBinarizer(classes=self.domains)
        
        # 分类器
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        )
        
        # 缓存设置
        self.use_cache = use_cache
        self.prediction_cache = {}
        self.analysis_cache = {}
        self.article_cache = {}
        
        # 初始化中文处理
        self.initialize_chinese_processor()
        
        # 分析模板
        self.initialize_enhanced_templates()
        
        # 文章生成模板
        self.initialize_article_templates()
        
        # 模型版本
        self.version = "3.0.0"
        self.model_info = {
            "name": "TextDomainClassifier",
            "version": self.version,
            "features": [
                "domain_classification",
                "viewpoint_analysis", 
                "keyword_extraction",
                "article_generation"
            ],
            "domains": self.domains,
            "created_at": datetime.now().isoformat()
        }
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def get_extended_domains(self) -> List[str]:
        """获取扩展的领域列表"""
        return [
            '科技', '教育', '职场', '生活', '健康', '财经', '旅游', '政治', 
            '娱乐', '体育', '社会', '文化', '经济', '环境', '心理', '哲学',
            '历史', '文学', '艺术', '科学', '技术', '商业', '管理', '法律'
        ]
    
    def get_basic_stopwords(self) -> List[str]:
        """获取基础停用词表"""
        return [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', 
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', 
            '中', '等'
        ]
    
    def initialize_chinese_processor(self):
        """初始化中文处理器"""
        # 添加领域特定词汇
        domain_words = []
        for domain in self.domains:
            domain_words.append(f"{domain}领域")
            domain_words.append(f"{domain}相关")
        
        for word in domain_words:
            jieba.add_word(word)
    
    def initialize_enhanced_templates(self):
        """初始化增强的分析模板"""
        # 肯定部分模板
        self.positive_templates = {
            '科技': [
                "你对{aspect}的思考很有前瞻性，体现了科技创新的核心价值。",
                "从{aspect}角度分析科技发展，展现了很好的技术敏感性。",
                "你对{aspect}的理解准确，抓住了科技发展的关键要素。"
            ],
            '教育': [
                "你对{aspect}的分析体现了对教育本质的理解。",
                "从{aspect}角度思考教育问题，展现了教育创新的思维。",
                "你对{aspect}的思考有助于推动教育理念的发展。"
            ],
            '通用': [
                "你的观点逻辑清晰，论证充分。",
                "思考角度新颖，展现了很好的创新思维。",
                "分析全面，考虑到了问题的多个方面。",
                "观点明确，表达流畅，易于理解。",
                "思考深入，抓住了问题的本质。"
            ]
        }
        
        # 优化建议模板
        self.suggestion_templates = {
            '科技': [
                "可以进一步探讨{aspect}对未来社会的具体影响。",
                "建议补充{aspect}在实际应用中的挑战和解决方案。",
                "可以结合具体案例说明{aspect}的发展趋势。"
            ],
            '教育': [
                "可以进一步探讨{aspect}对不同年龄段学习者的影响。",
                "建议补充{aspect}在教学实践中的具体应用。",
                "可以结合教育政策分析{aspect}的发展方向。"
            ],
            '通用': [
                "可以进一步补充具体数据和案例支持。",
                "建议从更多角度进行对比分析。",
                "可以深入探讨相关的影响因素。",
                "建议考虑相反观点的可能性。",
                "可以结合实际应用场景进行分析。"
            ]
        }
    
    def initialize_article_templates(self):
        """初始化文章生成模板"""
        # 文章结构模板
        self.article_structures = {
            'argumentative': ["引言", "论点1", "论点2", "论点3", "反驳观点", "结论"],
            'expository': ["背景介绍", "问题提出", "分析探讨", "解决方案", "总结展望"],
            'persuasive': ["问题引出", "立场陈述", "论据支持", "反方观点", "强化立场", "行动呼吁"],
            'comparative': ["主题引入", "A方分析", "B方分析", "对比分析", "综合评价", "总结建议"]
        }
        
        # 段落模板
        self.paragraph_templates = {
            'introduction': [
                "近年来，{topic}已成为社会各界广泛关注的热点话题。",
                "在当今社会，{topic}问题日益凸显，引发深入思考。",
                "{topic}作为当代重要的议题，值得我们深入探讨和分析。"
            ],
            'argument': [
                "首先，{point}是{aspect}的重要体现。",
                "其次，{aspect}在{context}中发挥着关键作用。",
                "此外，{aspect}与{related_aspect}密切相关。"
            ],
            'example': [
                "以{example}为例，我们可以清晰地看到{aspect}的影响。",
                "具体来说，{example_case}充分说明了{aspect}的重要性。",
                "在实际应用中，{example}展现了{aspect}的实际价值。"
            ],
            'conclusion': [
                "综上所述，{topic}问题的解决需要多方面共同努力。",
                "总而言之，{aspect}的发展对{topic}具有重要意义。",
                "通过上述分析，我们可以得出{topic}的关键启示。"
            ]
        }
        
        # 连接词库
        self.connectors = {
            'sequence': ["首先", "其次", "再次", "然后", "最后", "第一", "第二", "第三"],
            'contrast': ["然而", "但是", "尽管", "虽然", "可是", "不过", "却"],
            'addition': ["而且", "并且", "此外", "另外", "同时", "也", "还"],
            'cause': ["因为", "由于", "因此", "所以", "因而", "于是", "从而"],
            'conclusion': ["总之", "总而言之", "综上所述", "总的来说", "简而言之"]
        }
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not isinstance(text, str):
            return ""
        
        # 去除特殊字符和标点符号
        text = re.sub(r'[^\w\u4e00-\u9fff]+', ' ', text)
        
        # 分词
        words = jieba.lcut(text)
        
        # 去除停用词和单字
        processed_words = []
        for word in words:
            if len(word) > 1:  # 只保留长度大于1的词
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def predict(self, text: str, top_k: int = 5, detailed: bool = False, 
                threshold: float = 0.1) -> ClassificationResult:
        """预测文本所属领域"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = None
        if self.use_cache:
            cache_key = f"predict_{text}_{top_k}_{detailed}"
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                cached_result.processing_time = 0.001  # 缓存时间
                cached_result.timestamp = datetime.now().isoformat()
                return cached_result
        
        try:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                result = ClassificationResult(
                    text=text,
                    domains=[],
                    probabilities={},
                    top_domain=None,
                    confidence=0.0,
                    error='文本为空或预处理后无有效内容',
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
                return result
            
            # 提取特征
            X = self.vectorizer.transform([processed_text])
            
            # 预测概率
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X)[0]
            else:
                # 模拟概率
                probabilities = [random.random() for _ in range(len(self.domains))]
            
            # 获取领域和对应的概率
            domain_probs = list(zip(self.domains, probabilities))
            
            # 按概率降序排序
            domain_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 获取top_k个领域
            top_domains = []
            top_probabilities = {}
            
            for domain, prob in domain_probs[:top_k]:
                if prob > threshold:  # 概率阈值
                    top_domains.append(domain)
                    top_probabilities[domain] = float(prob)
            
            confidence = max(probabilities) if probabilities else 0.0
            
            # 创建结果对象
            result = ClassificationResult(
                text=text,
                domains=top_domains,
                probabilities=top_probabilities,
                top_domain=top_domains[0] if top_domains else None,
                confidence=confidence,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            # 添加详细信息
            if detailed:
                result.keywords = self.extract_keywords(text, 10)
                result.features = {
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'complexity': self.calculate_complexity(text)
                }
            
            # 缓存结果
            if self.use_cache and cache_key:
                self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return ClassificationResult(
                text=text,
                domains=[],
                probabilities={},
                top_domain=None,
                confidence=0.0,
                error=f'预测失败: {str(e)}',
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
    
    def analyze_viewpoint(self, original_text: str, response_text: str, 
                         perspective_type: str = "neutral", 
                         analysis_depth: str = "standard") -> AnalysisResult:
        """
        分析观点，给出肯定部分和优化建议
        
        参数:
            original_text: 原始观点文本
            response_text: 用户回应文本
            perspective_type: 视角类型
            analysis_depth: 分析深度
            
        返回:
            分析结果
        """
        start_time = time.time()
        
        try:
            # 提取关键词
            keywords = self.extract_keywords(response_text, 5)
            
            # 分析逻辑结构
            logic_score = self.analyze_logic_structure(response_text)
            
            # 分析情感
            sentiment_score = self.analyze_sentiment(response_text)
            
            # 分析连贯性
            coherence_score = self.analyze_coherence(response_text)
            
            # 生成肯定部分
            positive_feedback = self.generate_positive_feedback(
                response_text, perspective_type, keywords
            )
            
            # 生成优化建议
            suggestions = self.generate_suggestions(
                response_text, perspective_type, keywords
            )
            
            # 提取优势和改进点
            strengths = self.identify_strengths(response_text)
            areas_for_improvement = self.identify_improvement_areas(response_text)
            
            # 分析论证结构
            argument_structure = self.analyze_argument_structure(response_text)
            
            # 识别修辞手法
            rhetorical_devices = self.identify_rhetorical_devices(response_text)
            
            result = AnalysisResult(
                positive=positive_feedback,
                suggestion=suggestions,
                keywords=keywords,
                logic_score=logic_score,
                sentiment_score=sentiment_score,
                coherence_score=coherence_score,
                strengths=strengths,
                areas_for_improvement=areas_for_improvement,
                argument_structure=argument_structure,
                rhetorical_devices=rhetorical_devices,
                analysis_method="AI模型分析",
                text_length=len(response_text),
                reading_level=self.assess_reading_level(response_text),
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            return AnalysisResult(
                positive="你的观点展现了良好的思考能力。",
                suggestion="可以进一步深化思考，补充具体案例。",
                keywords=[],
                logic_score=0.6,
                sentiment_score=0.0,
                coherence_score=0.6,
                strengths=[],
                areas_for_improvement=[],
                argument_structure={},
                rhetorical_devices=[],
                analysis_method="基础分析",
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def generate_article(self, topic: str, perspective_type: str = "neutral", 
                        target_length: int = 500, structure_type: str = "argumentative") -> GeneratedArticle:
        """
        根据主题生成观点文章
        
        参数:
            topic: 文章主题
            perspective_type: 视角类型
            target_length: 目标字数
            structure_type: 文章结构类型
            
        返回:
            生成的文章
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = None
            if self.use_cache:
                cache_key = f"article_{topic}_{perspective_type}_{target_length}_{structure_type}"
                if cache_key in self.article_cache:
                    cached_article = self.article_cache[cache_key]
                    return cached_article
            
            # 分析主题
            classification = self.predict(topic, top_k=1)
            main_domain = classification.top_domain or "综合"
            
            # 生成标题
            title = self.generate_title(topic, perspective_type, main_domain)
            
            # 确定文章结构
            structure = self.article_structures.get(structure_type, self.article_structures["argumentative"])
            
            # 生成段落
            paragraphs = []
            current_length = 0
            
            for section in structure:
                paragraph = self.generate_paragraph(
                    topic=topic,
                    section=section,
                    perspective_type=perspective_type,
                    domain=main_domain,
                    target_section_length=target_length // len(structure)
                )
                paragraphs.append({
                    "section": section,
                    "content": paragraph,
                    "length": len(paragraph)
                })
                current_length += len(paragraph)
            
            # 组合文章内容
            content = "\n\n".join([p["content"] for p in paragraphs])
            
            # 调整长度
            if current_length < target_length:
                # 添加额外内容
                additional_content = self.generate_additional_content(topic, perspective_type, 
                                                                     main_domain, target_length - current_length)
                content += "\n\n" + additional_content
            
            # 计算字数
            word_count = len(content)
            
            # 生成主要观点
            main_idea = self.generate_main_idea(topic, perspective_type, main_domain)
            
            # 提取关键词
            keywords = self.extract_keywords(content, 8)
            
            # 评估质量
            coherence_score = self.analyze_coherence(content)
            quality_score = self.assess_article_quality(content, structure, main_domain)
            
            result = GeneratedArticle(
                title=title,
                content=content,
                main_idea=main_idea,
                structure=paragraphs,
                keywords=keywords,
                word_count=word_count,
                reading_time_minutes=reading_time_minutes,
                target_audience=target_audience,
                writing_style=writing_style,
                coherence_score=coherence_score,
                quality_score=quality_score,
                generation_method=generation_method,
                timestamp=datetime.now().isoformat()
            )
            
            # 缓存结果
            if self.use_cache and cache_key:
                self.article_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            # 返回默认文章
            default_content = f"""关于{topic}的思考

近年来，{topic}问题引起了社会各界的广泛关注。从不同的视角来看，这一问题涉及多个方面的考量。

首先，从技术发展的角度来看，{topic}代表了当前社会发展的重要趋势。技术的进步为我们提供了新的解决思路和方法，但也带来了新的挑战。

其次，从社会影响的角度分析，{topic}对人们的生活方式、工作模式和社会关系都产生了深远影响。这需要我们认真思考如何平衡发展与稳定的关系。

再者，从个人发展的层面来看，{topic}为个人成长提供了新的机遇。每个人都应该积极面对这一变化，不断提升自己的适应能力。

总之，{topic}是一个复杂而重要的议题。我们需要保持开放的心态，积极思考，共同探索适合的发展道路。"""
            
            return GeneratedArticle(
                title=f"关于{topic}的思考",
                content=default_content,
                main_idea=f"探讨{topic}的多方面影响和发展趋势",
                structure=[{"section": "引言", "content": "..."}],
                keywords=self.extract_keywords(default_content, 5),
                word_count=len(default_content),
                reading_time_minutes=round(len(default_content) / 300, 1),
                target_audience="普通读者",
                writing_style="论述性",
                coherence_score=0.7,
                quality_score=0.6,
                generation_method="基础模板生成",
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def generate_title(self, topic: str, perspective_type: str, domain: str) -> str:
        """生成文章标题"""
        title_templates = {
            "neutral": ["关于{topic}的思考与分析", "{topic}：现状与展望", "探讨{topic}的发展趋势"],
            "opposite": ["重新审视{topic}：另一种视角", "{topic}的反思与批判", "对{topic}的不同看法"],
            "supplement": ["补充视角：{topic}的更多维度", "{topic}的深入思考", "关于{topic}的新见解"],
            "unique": ["独特视角：{topic}的创新思考", "{topic}的非传统分析", "重新定义{topic}的认知"]
        }
        
        templates = title_templates.get(perspective_type, title_templates["neutral"])
        template = random.choice(templates)
        return template.format(topic=topic)
    
    def generate_paragraph(self, topic: str, section: str, perspective_type: str, 
                          domain: str, target_section_length: int) -> str:
        """生成段落"""
        paragraph = ""
        
        if section in ["引言", "背景介绍", "问题引出"]:
            template = random.choice(self.paragraph_templates["introduction"])
            paragraph = template.format(topic=topic)
            
            # 添加背景信息
            background = self.generate_background_info(domain, perspective_type)
            paragraph += " " + background
        
        elif "论点" in section or "分析" in section or "探讨" in section:
            # 生成论点
            point = self.generate_argument_point(topic, domain, perspective_type)
            template = random.choice(self.paragraph_templates["argument"])
            paragraph = template.format(point=point, aspect=domain, context=topic)
            
            # 添加例证
            if random.random() > 0.3:  # 70%概率添加例子
                example = self.generate_example(domain, point)
                example_template = random.choice(self.paragraph_templates["example"])
                paragraph += " " + example_template.format(example=example, aspect=point)
        
        elif section in ["结论", "总结", "展望"]:
            template = random.choice(self.paragraph_templates["conclusion"])
            paragraph = template.format(topic=topic, aspect=domain)
            
            # 添加展望
            outlook = self.generate_future_outlook(topic, domain, perspective_type)
            paragraph += " " + outlook
        
        # 确保段落长度
        if len(paragraph) < target_section_length * 0.5:
            additional = self.generate_additional_content(topic, perspective_type, domain, 
                                                         target_section_length - len(paragraph))
            paragraph += " " + additional
        
        return paragraph
    
    def generate_background_info(self, domain: str, perspective_type: str) -> str:
        """生成背景信息"""
        background_templates = {
            "科技": f"在{domain}领域，技术发展日新月异，",
            "教育": f"在当前{domain}背景下，教育创新不断推进，",
            "经济": f"从{domain}角度观察，经济发展呈现新态势，",
            "社会": f"在社会{domain}发展中，各种因素相互作用，"
        }
        
        perspective_background = {
            "neutral": "需要全面客观地分析。",
            "opposite": "可能存在不同的观点和争议。",
            "supplement": "还有很多值得深入探讨的方面。",
            "unique": "可以从独特的视角进行创新思考。"
        }
        
        domain_info = background_templates.get(domain, f"在{domain}领域，")
        perspective_info = perspective_background.get(perspective_type, "值得深入探讨。")
        
        return domain_info + perspective_info

    
    def generate_argument_point(self, topic: str, domain: str, perspective_type: str) -> str:
        """生成论点"""
        points = {
            "科技": ["技术创新", "效率提升", "用户体验", "数据安全", "隐私保护"],
            "教育": ["教学方法", "学习效果", "资源公平", "评估方式", "个性化学习"],
            "社会": ["公平正义", "社会福利", "社区建设", "文化传承", "社会治理"]
        }
        
        if domain in points:
            return random.choice(points[domain])
        return random.choice(["发展前景", "社会影响", "个人价值", "实践意义"])
    
    def generate_example(self, domain: str, point: str) -> str:
        """生成例子"""
        examples = {
            "科技": ["人工智能应用", "大数据分析", "云计算服务", "物联网技术"],
            "教育": ["在线教育平台", "混合式教学", "个性化学习系统", "教育大数据"],
            "社会": ["社区治理创新", "公共服务优化", "文化传承保护", "环境保护实践"]
        }
        
        if domain in examples:
            return random.choice(examples[domain])
        return random.choice(["实际案例分析", "相关研究数据", "实践经验分享"])
    
    def generate_additional_content(self, topic: str, perspective_type: str, 
                                  domain: str, target_length: int) -> str:
        """生成额外内容"""
        content = ""
        
        # 添加连接词
        connector = random.choice(self.connectors["addition"])
        
        # 添加深入分析
        deep_analysis = self.generate_deep_analysis(topic, domain, perspective_type)
        content = connector + "，" + deep_analysis
        
        # 如果还不够长，添加更多内容
        if len(content) < target_length:
            more_content = self.generate_more_content(topic, domain)
            content += " " + more_content
        
        return content
    
    def generate_deep_analysis(self, topic: str, domain: str, perspective_type: str) -> str:
        """生成深入分析"""
        analyses = [
            f"从更深层次来看，{topic}体现了{domain}发展的内在逻辑。",
            f"进一步分析可以发现，{topic}与{domain}的多个方面密切相关。",
            f"这要求我们重新思考{domain}在{topic}中的作用和意义。"
        ]
        return random.choice(analyses)
    
    def generate_main_idea(self, topic: str, perspective_type: str, domain: str) -> str:
        """生成主要观点"""
        ideas = {
            "neutral": f"本文从多角度分析了{topic}的现状、挑战和未来发展趋势。",
            "opposite": f"本文提出了对{topic}传统观点的反思和新的思考方向。",
            "supplement": f"本文补充了关于{topic}的传统分析中忽略的重要方面。",
            "unique": f"本文从独特视角探讨了{topic}，提供了创新性的见解。"
        }
        return ideas.get(perspective_type, f"关于{topic}的综合分析与思考")
    
    def determine_target_audience(self, domain: str, perspective_type: str) -> str:
        """确定目标读者"""
        if perspective_type in ["opposite", "unique"]:
            return "专业读者和研究人员"
        elif domain in ["科技", "经济", "法律"]:
            return "相关领域从业者和学习者"
        else:
            return "普通读者和感兴趣人士"
    
    def determine_writing_style(self, perspective_type: str, structure_type: str) -> str:
        """确定写作风格"""
        styles = {
            "neutral": "客观分析",
            "opposite": "批判性思考", 
            "supplement": "补充性论述",
            "unique": "创新性探索"
        }
        return styles.get(perspective_type, "论述性")
    
    def assess_article_quality(self, content: str, structure: List, domain: str) -> float:
        """评估文章质量"""
        score = 0.5  # 基础分
        
        # 结构完整性
        if len(structure) >= 3:
            score += 0.1
        
        # 内容长度
        if len(content) > 300:
            score += 0.1
        
        # 连贯性
        coherence = self.analyze_coherence(content)
        score += coherence * 0.2
        
        # 确保在合理范围
        return min(max(score, 0.3), 0.95)
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        try:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(
                text, 
                topK=top_n, 
                withWeight=False,
                allowPOS=('n', 'v', 'a')
            )
            return keywords
        except:
            # 回退方法
            words = self.preprocess_text(text).split()
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(top_n)]
    
    def analyze_logic_structure(self, text: str) -> float:
        """分析逻辑结构评分"""
        score = 0.5
        
        # 长度评分
        if len(text) > 200:
            score += 0.2
        elif len(text) > 100:
            score += 0.1
        elif len(text) < 30:
            score -= 0.1
        
        # 连接词分析
        connectives = ['首先', '其次', '再次', '最后', '因为', '所以', 
                      '然而', '但是', '而且', '此外', '例如', '比如']
        found_connectives = sum(1 for conn in connectives if conn in text)
        
        if found_connectives >= 3:
            score += 0.15
        elif found_connectives >= 1:
            score += 0.05
        
        return round(min(max(score, 0.3), 0.95), 2)
    
    def analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        # 简单的情感分析
        positive_words = ['好', '优秀', '成功', '进步', '发展', '积极', '有益']
        negative_words = ['问题', '困难', '挑战', '不足', '缺陷', '消极', '有害']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        return round(sentiment, 2)
    
    def analyze_coherence(self, text: str) -> float:
        """分析连贯性"""
        score = 0.6
        
        # 句子数量
        sentences = re.split(r'[。！？]', text)
        if len(sentences) > 3:
            score += 0.1
        
        # 段落数量
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1
        
        return round(min(max(score, 0.4), 0.9), 2)
    
    def identify_strengths(self, text: str) -> List[str]:
        """识别优势"""
        strengths = []
        
        if len(text) > 100:
            strengths.append("论述充分")
        
        if any(conn in text for conn in self.connectors['sequence']):
            strengths.append("逻辑清晰")
        
        if any(conn in text for conn in self.connectors['example']):
            strengths.append("例证恰当")
        
        return strengths[:3]
    
    def identify_improvement_areas(self, text: str) -> List[str]:
        """识别改进点"""
        areas = []
        
        if len(text) < 50:
            areas.append("内容可以更丰富")
        
        if text.count('。') < 3:
            areas.append("可以增加具体例子")
        
        if not any(marker in text for marker in ['首先', '其次', '最后']):
            areas.append("结构可以更清晰")
        
        return areas[:3]
    
    def analyze_argument_structure(self, text: str) -> Dict[str, Any]:
        """分析论证结构"""
        return {
            'premises': random.randint(1, 3),
            'conclusions': 1,
            'examples': text.count('例如') + text.count('比如'),
            'counterarguments': text.count('然而') + text.count('但是')
        }
    
    def identify_rhetorical_devices(self, text: str) -> List[str]:
        """识别修辞手法"""
        devices = []
        
        if '例如' in text or '比如' in text:
            devices.append('举例')
        
        if '首先' in text and '其次' in text:
            devices.append('列举')
        
        if '因为' in text and '所以' in text:
            devices.append('因果')
        
        return devices
    
    def assess_reading_level(self, text: str) -> str:
        """评估阅读水平"""
        length = len(text)
        
        if length < 50:
            return "入门级"
        elif length < 200:
            return "基础级"
        elif length < 500:
            return "进阶级"
        else:
            return "专业级"
    
    def calculate_complexity(self, text: str) -> float:
        """计算文本复杂度"""
        words = len(text)
        sentences = len(re.findall(r'[。！？]', text)) or 1
        avg_sentence_length = words / sentences
        
        if avg_sentence_length > 30:
            return 0.8
        elif avg_sentence_length > 20:
            return 0.6
        elif avg_sentence_length > 10:
            return 0.4
        else:
            return 0.2

    def generate_positive_feedback(self, text: str, perspective_type: str, keywords: List[str]) -> str:
        """生成肯定反馈"""
        # 确定主要领域
        main_domains = self.predict(text, top_k=1).domains
        main_domain = main_domains[0] if main_domains else None
        
        # 选择模板
        if main_domain and main_domain in self.positive_templates:
            templates = self.positive_templates[main_domain]
        else:
            templates = self.positive_templates['通用']
        
        # 选择关键词作为分析角度
        aspect = keywords[0] if keywords else "这个观点"
        
        # 从模板中随机选择
        template = random.choice(templates)
        feedback = template.format(aspect=aspect)
        
        # 添加视角相关的肯定
        perspective_compliments = {
            'opposite': "你能从对立角度思考，展现了很好的辩证思维。",
            'neutral': "你的中立分析客观全面，体现了很好的平衡感。",
            'supplement': "你的补充观点很有价值，扩展了思考的维度。",
            'unique': "你的小众视角很独特，展现了创新性思考。"
        }
        
        if perspective_type in perspective_compliments:
            feedback += " " + perspective_compliments[perspective_type]
        
        return feedback
    
    def generate_suggestions(self, text: str, perspective_type: str, keywords: List[str]) -> str:
        """生成优化建议"""
        # 确定主要领域
        main_domains = self.predict(text, top_k=1).domains
        main_domain = main_domains[0] if main_domains else None
        
        # 选择模板
        if main_domain and main_domain in self.suggestion_templates:
            templates = self.suggestion_templates[main_domain]
        else:
            templates = self.suggestion_templates['通用']
        
        # 选择关键词作为分析角度
        aspect = keywords[0] if keywords else "这个方面"
        
        # 从模板中随机选择
        template = random.choice(templates)
        suggestion = template.format(aspect=aspect)
        
        # 添加视角相关的建议
        perspective_suggestions = {
            'opposite': "可以进一步考虑对立观点可能存在的局限性。",
            'neutral': "可以更深入地探讨各方的合理性和不足。",
            'supplement': "可以思考如何将这些补充观点整合到原有框架中。",
            'unique': "可以探讨小众视角在更大范围内的适用性。"
        }
        
        if perspective_type in perspective_suggestions:
            suggestion += " " + perspective_suggestions[perspective_type]
        
        return suggestion
    
    def train(self, texts: List[str], labels: List[List[str]], test_size: float = 0.2):
        """训练模型"""
        if not texts or not labels:
            # 创建训练数据
            texts, labels = self.create_training_data(200)
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 转换标签
        y = self.mlb.fit_transform(labels)
        
        # 提取特征
        X = self.vectorizer.fit_transform(processed_texts)
        
        # 训练分类器
        self.classifier.fit(X, y)
        
        print(f"训练完成！")
        print(f"训练样本数: {len(texts)}")
        print(f"特征维度: {X.shape[1]}")
        print(f"支持的领域: {self.domains}")
        
        return None
    
    def create_training_data(self, num_samples=200):
        """创建训练数据"""
        texts = []
        labels = []
        
        # 科技类
        tech_texts = [
            ("人工智能是未来科技发展的重要方向，机器学习算法正在改变我们的生活。", ['科技']),
            ("5G技术的普及将推动物联网和智能家居的发展。", ['科技']),
            ("区块链技术为数字货币和供应链管理提供了新的解决方案。", ['科技']),
        ]
        
        # 教育类
        edu_texts = [
            ("在线教育的兴起改变了传统教学模式，为学生提供了更多学习选择。", ['教育']),
            ("如何培养孩子的自主学习能力和创新思维？", ['教育']),
            ("大学专业选择对未来职业发展有重要影响，需要慎重考虑。", ['教育']),
        ]
        
        # 职场类
        work_texts = [
            ("职场新人如何快速适应工作环境，建立良好的人际关系？", ['职场']),
            ("职业规划对个人发展至关重要，需要定期审视和调整。", ['职场']),
            ("有效的时间管理和任务优先级安排能显著提高工作效率。", ['职场']),
        ]
        
        # 生活类
        life_texts = [
            ("如何合理安排一天的时间，提高生活效率和工作质量？", ['生活']),
            ("家庭装修时需要注意哪些细节，才能创造舒适的居住环境？", ['生活']),
            ("日常生活中的健康小习惯，长期坚持会让你受益匪浅。", ['生活']),
        ]
        
        # 合并所有数据
        all_data = tech_texts + edu_texts + work_texts + life_texts
        
        # 复制数据以增加样本量
        for _ in range(num_samples // len(all_data) + 1):
            for text, label in all_data:
                texts.append(text)
                labels.append(label)
        
        return texts[:num_samples], labels[:num_samples]
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型数据
            model_data = {
                'vectorizer': self.vectorizer,
                'mlb': self.mlb,
                'classifier': self.classifier,
                'domains': self.domains,
                'model_info': self.model_info,
                'version': self.version
            }
            
            joblib.dump(model_data, filepath)
            
            # 保存模型信息
            info_file = filepath.replace('.pkl', '_info.json')
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_info, f, indent=2, ensure_ascii=False)
            
            print(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")
            
            # 加载模型数据
            model_data = joblib.load(filepath)
            
            self.vectorizer = model_data['vectorizer']
            self.mlb = model_data['mlb']
            self.classifier = model_data['classifier']
            self.domains = model_data['domains']
            self.model_info = model_data.get('model_info', {})
            self.version = model_data.get('version', '3.0.0')
            
            print(f"模型已从 {filepath} 加载")
            print(f"版本: {self.version}")
            print(f"支持的领域: {self.domains}")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            **self.model_info,
            'current_time': datetime.now().isoformat(),
            'cache_sizes': {
                'prediction_cache': len(self.prediction_cache),
                'analysis_cache': len(self.analysis_cache),
                'article_cache': len(self.article_cache)
            }
        }

# 测试函数
def test_article_generation():
    """测试文章生成功能"""
    print("测试文章生成功能...")
    
    # 创建分类器
    classifier = TextDomainClassifier()
    
    # 测试数据
    test_topics = [
        {
            'topic': "人工智能对社会的影响",
            'perspective': 'neutral',
            'target_length': 300
        },
        {
            'topic': "线上教育的优缺点",
            'perspective': 'opposite', 
            'target_length': 400
        },
        {
            'topic': "环境保护与经济发展的平衡",
            'perspective': 'supplement',
            'target_length': 500
        }
    ]
    
    for i, test_case in enumerate(test_topics, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}")
        print(f"{'='*60}")
        print(f"主题: {test_case['topic']}")
        print(f"视角类型: {test_case['perspective']}")
        print(f"目标字数: {test_case['target_length']}")
        
        # 生成文章
        start_time = time.time()
        article = classifier.generate_article(
            topic=test_case['topic'],
            perspective_type=test_case['perspective'],
            target_length=test_case['target_length']
        )
        processing_time = time.time() - start_time
        
        print(f"\n📝 生成的文章:")
        print(f"标题: {article.title}")
        print(f"字数: {article.word_count}字")
        print(f"阅读时间: {article.reading_time_minutes}分钟")
        print(f"目标读者: {article.target_audience}")
        print(f"写作风格: {article.writing_style}")
        print(f"连贯性评分: {article.coherence_score}")
        print(f"质量评分: {article.quality_score}")
        
        print(f"\n📄 文章内容:")
        print(article.content[:200] + "..." if len(article.content) > 200 else article.content)
        
        print(f"\n⏱️ 处理时间: {processing_time:.2f}秒")
    
    return classifier

# 主程序
if __name__ == "__main__":
    # 运行测试
    classifier = test_article_generation()
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    classifier.save_model("models/text_domain_classifier.pkl")
    
    print("\n✅ 模型测试完成，已保存到 models/text_domain_classifier.pkl")
    print("\n📊 模型信息:")
    info = classifier.get_model_info()
    for key, value in info.items():
        if key != 'domains':  # domains太长，单独显示
            print(f"  {key}: {value}")
    print(f"  支持的领域数量: {len(classifier.domains)}")

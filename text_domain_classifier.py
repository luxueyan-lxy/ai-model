"""
增强版文本领域分类器
包含观点分析功能，可以给出肯定部分和优化建议
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import joblib
import json
import os
import re
import jieba
import random
from collections import Counter
import warnings
from typing import List, Dict, Any, Optional, Tuple
warnings.filterwarnings('ignore')

class TextDomainClassifier:
    def __init__(self, model_path=None, domains=None):
        """
        初始化文本领域分类器
        
        参数:
            model_path: 预训练模型路径
            domains: 预定义的领域列表
        """
        self.domains = domains or [
            '科技', '生活', '教育', '职场', '旅游', 
            '健康', '财经', '娱乐', '体育', '政治', 
            '美食', '艺术', '汽车', '房产', '游戏'
        ]
        
        # 特征提取器
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=['的', '了', '在', '是', '我', '有', '和', '就',
                       '不', '人', '都', '一', '一个', '上', '也', '很',
                       '到', '说', '要', '去', '你', '会', '着', '没有',
                       '看', '好', '自己', '这', '那', '中', '等']
        )
        
        # 多标签二值化器
        self.mlb = MultiLabelBinarizer(classes=self.domains)
        
        # 分类器
        self.classifier = OneVsRestClassifier(
            MultinomialNB(alpha=0.1)
        )
        
        # 初始化中文分词器
        self.initialize_jieba()
        
        # 观点分析模板
        self.initialize_analysis_templates()
        
        # 如果提供了模型路径，尝试加载
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def initialize_jieba(self):
        """初始化中文分词器"""
        domain_words = []
        for domain in self.domains:
            domain_words.append(f"{domain}领域")
            domain_words.append(f"{domain}相关")
        
        # 添加一些常见词汇
        common_words = [
            '人工智能', '机器学习', '深度学习', '大数据', '云计算',
            '区块链', '物联网', '5G', '虚拟现实', '增强现实',
            '在线教育', '远程办公', '自主学习', '职业规划',
            '自由行', '自驾游', '深度游', '周末游',
            '健康饮食', '健身运动', '心理压力', '睡眠质量',
            '投资理财', '股票基金', '房地产', '数字货币',
            '电影音乐', '综艺节目', '明星八卦', '网络游戏',
            '篮球足球', '健身跑步', '奥运会', '世界杯',
            '政策法规', '国际关系', '选举投票', '政府工作'
        ]
        
        for word in domain_words + common_words:
            jieba.add_word(word)
    
    def initialize_analysis_templates(self):
        """初始化观点分析模板"""
        # 肯定部分模板
        self.positive_templates = {
            '科技': [
                "你对{aspect}的思考很有前瞻性，体现了科技创新的核心价值。",
                "从{aspect}角度分析科技发展，展现了很好的技术敏感性。",
                "你对{aspect}的理解准确，抓住了科技发展的关键要素。"
            ],
            '生活': [
                "你对{aspect}的观察很细致，体现了对生活质量的关注。",
                "从{aspect}出发思考生活问题，展现了很好的生活智慧。",
                "你对{aspect}的见解贴近实际，具有很好的实用性。"
            ],
            '教育': [
                "你对{aspect}的分析体现了对教育本质的理解。",
                "从{aspect}角度思考教育问题，展现了教育创新的思维。",
                "你对{aspect}的思考有助于推动教育理念的发展。"
            ],
            '职场': [
                "你对{aspect}的分析很专业，具有很好的职场指导意义。",
                "从{aspect}思考职场发展，展现了良好的职业素养。",
                "你对{aspect}的理解有助于职业规划和能力提升。"
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
            '生活': [
                "可以进一步探讨{aspect}对不同人群的影响差异。",
                "建议补充{aspect}在日常生活中的实用建议。",
                "可以结合个人经验说明{aspect}的重要性。"
            ],
            '教育': [
                "可以进一步探讨{aspect}对不同年龄段学习者的影响。",
                "建议补充{aspect}在教学实践中的具体应用。",
                "可以结合教育政策分析{aspect}的发展方向。"
            ],
            '职场': [
                "可以进一步探讨{aspect}对职业发展的长期影响。",
                "建议补充{aspect}在不同行业中的具体表现。",
                "可以结合职业规划说明{aspect}的重要性。"
            ],
            '通用': [
                "可以进一步补充具体数据和案例支持。",
                "建议从更多角度进行对比分析。",
                "可以深入探讨相关的影响因素。",
                "建议考虑相反观点的可能性。",
                "可以结合实际应用场景进行分析。"
            ]
        }
    
    def preprocess_text(self, text):
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
    
    def analyze_viewpoint(self, original_text: str, response_text: str, perspective_type: str) -> Dict[str, Any]:
        """
        分析观点，给出肯定部分和优化建议
        
        参数:
            original_text: 原始观点文本
            response_text: 用户回应文本
            perspective_type: 视角类型
            
        返回:
            分析结果字典
        """
        try:
            # 提取关键词
            keywords = self.get_keywords(response_text, top_n=5)
            
            # 分析逻辑结构
            logic_score = self.analyze_logic_structure(response_text)
            
            # 生成肯定部分
            positive_feedback = self.generate_positive_feedback(
                response_text, 
                perspective_type,
                keywords
            )
            
            # 生成优化建议
            suggestions = self.generate_suggestions(
                response_text,
                perspective_type,
                keywords
            )
            
            return {
                'positive': positive_feedback,
                'suggestion': suggestions,
                'keywords': keywords,
                'logic_score': logic_score,
                'analysis_method': 'AI模型分析',
                'text_length': len(response_text)
            }
            
        except Exception as e:
            return {
                'positive': '你的观点展现了很好的思考能力。',
                'suggestion': '可以进一步深化思考，补充具体案例。',
                'keywords': [],
                'logic_score': 0.6,
                'analysis_method': '基础分析',
                'error': str(e)
            }
    
    def generate_positive_feedback(self, text: str, perspective_type: str, keywords: List[str]) -> str:
        """生成肯定反馈"""
        # 确定主要领域
        main_domains = self.predict(text, top_k=1)['domains']
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
        
        # 填充模板
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
        
        # 根据文本长度添加额外肯定
        if len(text) > 100:
            feedback += " 论证充分，思考深入。"
        elif len(text) > 50:
            feedback += " 观点明确，表达清晰。"
        
        return feedback
    
    def generate_suggestions(self, text: str, perspective_type: str, keywords: List[str]) -> str:
        """生成优化建议"""
        # 确定主要领域
        main_domains = self.predict(text, top_k=1)['domains']
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
        
        # 填充模板
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
        
        # 根据文本长度添加额外建议
        if len(text) < 50:
            suggestion += " 可以进一步展开论述，提供更多细节。"
        elif len(text) < 100:
            suggestion += " 可以补充具体案例，增强说服力。"
        
        return suggestion
    
    def analyze_logic_structure(self, text: str) -> float:
        """分析逻辑结构评分"""
        # 基于文本特征计算逻辑评分
        score = 0.5  # 基础分
        
        # 长度评分
        if len(text) > 200:
            score += 0.2
        elif len(text) > 100:
            score += 0.1
        elif len(text) < 30:
            score -= 0.1
        
        # 连接词分析
        connectives = ['首先', '其次', '再次', '最后', '因为', '所以', '然而', '但是', 
                      '而且', '此外', '例如', '比如', '总之', '综上所述']
        found_connectives = sum(1 for conn in connectives if conn in text)
        
        if found_connectives >= 3:
            score += 0.15
        elif found_connectives >= 1:
            score += 0.05
        
        # 标点符号分析
        punctuation_count = text.count('，') + text.count('。') + text.count('；')
        if len(text) > 0:
            punctuation_density = punctuation_count / len(text)
            if 0.05 <= punctuation_density <= 0.15:
                score += 0.1
            elif punctuation_density < 0.02:
                score -= 0.1
        
        # 确保评分在合理范围内
        score = max(0.3, min(0.95, score))
        
        return round(score, 2)
    
    def get_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取文本关键词"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return []
        
        words = processed_text.split()
        
        # 统计词频
        word_counts = Counter(words)
        
        # 获取最常见的词
        common_words = word_counts.most_common(top_n)
        
        keywords = []
        for word, count in common_words:
            if len(word) > 1:  # 只保留长度大于1的词
                keywords.append(word)
        
        return keywords
    
    def predict(self, text, top_k=3, threshold=0.1):
        """预测文本所属领域"""
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'text': text,
                'domains': [],
                'probabilities': {},
                'top_domain': None,
                'error': '文本为空或预处理后无有效内容'
            }
        
        try:
            # 提取特征
            X = self.vectorizer.transform([processed_text])
            
            # 预测概率
            probabilities = self.classifier.predict_proba(X)[0]
            
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
            
            return {
                'text': text,
                'domains': top_domains,
                'probabilities': top_probabilities,
                'top_domain': top_domains[0] if top_domains else None
            }
            
        except Exception as e:
            return {
                'text': text,
                'domains': [],
                'probabilities': {},
                'top_domain': None,
                'error': f'预测失败: {str(e)}'
            }
    
    def train(self, texts=None, labels=None, test_size=0.2, random_state=42):
        """训练分类器"""
        if texts is None or labels is None:
            texts, labels = self.create_training_data(200)
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 转换标签
        y = self.mlb.fit_transform(labels)
        
        # 分割训练集和测试集
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, y_train = processed_texts, y
            X_test, y_test = [], []
        
        # 提取特征
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        if len(X_test) > 0:
            X_test_vec = self.vectorizer.transform(X_test)
        
        # 训练分类器
        self.classifier.fit(X_train_vec, y_train)
        
        # 评估模型
        if len(X_test) > 0:
            y_pred = self.classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"训练完成！")
            print(f"训练样本数: {len(X_train)}")
            print(f"测试样本数: {len(X_test)}")
            print(f"准确率: {accuracy:.2%}")
            print(f"特征维度: {X_train_vec.shape[1]}")
            print(f"支持的领域: {self.domains}")
            
            return accuracy
        else:
            print(f"训练完成！")
            print(f"训练样本数: {len(X_train)}")
            print(f"特征维度: {X_train_vec.shape[1]}")
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
            ("深度学习在图像识别和自然语言处理方面取得了重大突破。", ['科技']),
            ("区块链技术为数字货币和供应链管理提供了新的解决方案。", ['科技']),
        ]
        
        # 生活类
        life_texts = [
            ("如何合理安排一天的时间，提高生活效率和工作质量？", ['生活']),
            ("家庭装修时需要注意哪些细节，才能创造舒适的居住环境？", ['生活']),
            ("日常生活中的健康小习惯，长期坚持会让你受益匪浅。", ['生活']),
            ("学会情绪管理，让生活更加从容和幸福。", ['生活']),
        ]
        
        # 教育类
        edu_texts = [
            ("在线教育的兴起改变了传统教学模式，为学生提供了更多学习选择。", ['教育']),
            ("如何培养孩子的自主学习能力和创新思维？", ['教育']),
            ("大学专业选择对未来职业发展有重要影响，需要慎重考虑。", ['教育']),
            ("终身学习理念在快速变化的时代变得越来越重要。", ['教育']),
        ]
        
        # 职场类
        work_texts = [
            ("职场新人如何快速适应工作环境，建立良好的人际关系？", ['职场']),
            ("职业规划对个人发展至关重要，需要定期审视和调整。", ['职场']),
            ("有效的时间管理和任务优先级安排能显著提高工作效率。", ['职场']),
            ("职场沟通技巧包括倾听、表达和反馈等多个方面。", ['职场']),
        ]
        
        # 旅游类
        travel_texts = [
            ("日本京都的樱花季是最佳的旅游时间，可以体验传统和服文化。", ['旅游']),
            ("自驾游西藏需要做好充分准备，包括车辆检查和高原反应预防。", ['旅游']),
            ("巴厘岛的海滩度假和SPA体验是放松身心的绝佳选择。", ['旅游']),
            ("城市深度游可以探索当地的历史文化和美食特色。", ['旅游']),
        ]
        
        # 多标签示例
        multi_label_texts = [
            ("在线教育平台使用人工智能技术为学生提供个性化学习方案。", ['科技', '教育']),
            ("职场人士的旅游攻略：如何平衡工作和休闲时间。", ['职场', '旅游']),
            ("健康饮食和规律运动对职场人士的工作效率有积极影响。", ['健康', '职场']),
            ("财经新闻APP利用大数据分析提供个性化投资建议。", ['科技', '财经']),
        ]
        
        # 合并所有数据
        all_data = (tech_texts + life_texts + edu_texts + work_texts + 
                   travel_texts + multi_label_texts)
        
        # 复制数据以增加样本量
        for _ in range(num_samples // len(all_data) + 1):
            for text, label in all_data:
                texts.append(text)
                labels.append(label)
        
        return texts[:num_samples], labels[:num_samples]
    
    def save_model(self, filepath):
        """保存模型"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型数据
        model_data = {
            'vectorizer': self.vectorizer,
            'mlb': self.mlb,
            'classifier': self.classifier,
            'domains': self.domains
        }
        
        joblib.dump(model_data, filepath)
        
        # 保存模型信息
        info_file = filepath.replace('.pkl', '_info.json')
        model_info = {
            'name': 'TextDomainClassifier',
            'version': '2.0.0',
            'features': ['domain_classification', 'viewpoint_analysis', 'keyword_extraction'],
            'domains': self.domains,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'saved_at': pd.Timestamp.now().isoformat()
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {filepath}")
        print(f"模型信息已保存到: {info_file}")
        
        return True
    
    def load_model(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            # 加载模型数据
            model_data = joblib.load(filepath)
            
            self.vectorizer = model_data['vectorizer']
            self.mlb = model_data['mlb']
            self.classifier = model_data['classifier']
            self.domains = model_data['domains']
            
            # 加载模型信息
            info_file = filepath.replace('.pkl', '_info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                print(f"加载模型信息: {model_info}")
            
            print(f"模型已从 {filepath} 加载")
            print(f"版本: 2.0.0 (包含观点分析功能)")
            print(f"支持的领域: {self.domains}")
            print(f"特征数量: {len(self.vectorizer.get_feature_names_out())}")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

# 测试函数
def test_analysis_functionality():
    """测试观点分析功能"""
    print("测试观点分析功能...")
    
    # 创建分类器
    classifier = TextDomainClassifier()
    
    # 测试数据
    test_cases = [
        {
            'original': "人工智能将改变未来工作方式",
            'response': "我认为人工智能确实会改变工作方式，特别是重复性工作将被自动化，但创造性工作和人际交往类工作仍需要人类完成。这需要人们不断提升自己的技能。",
            'perspective': 'neutral'
        },
        {
            'original': "线上教育比传统教育更有效",
            'response': "我不完全同意这个观点。线上教育虽然方便灵活，但缺乏面对面的互动和即时反馈，对于需要实践操作的课程效果可能不如传统教育。",
            'perspective': 'opposite'
        },
        {
            'original': "环保应该优先于经济发展",
            'response': "我认为应该在环保和经济发展之间找到平衡。可以发展绿色经济，既保护环境又促进经济发展，实现可持续发展。",
            'perspective': 'supplement'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}")
        print(f"{'='*60}")
        print(f"原始观点: {test_case['original']}")
        print(f"用户回应: {test_case['response']}")
        print(f"视角类型: {test_case['perspective']}")
        
        # 进行观点分析
        analysis = classifier.analyze_viewpoint(
            test_case['original'],
            test_case['response'],
            test_case['perspective']
        )
        
        print(f"\n分析结果:")
        print(f"👍 肯定部分: {analysis['positive']}")
        print(f"💡 优化建议: {analysis['suggestion']}")
        print(f"🔑 关键词: {', '.join(analysis['keywords'])}")
        print(f"📊 逻辑评分: {analysis['logic_score']}")
    
    return classifier

if __name__ == "__main__":
    # 运行测试
    classifier = test_analysis_functionality()
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    classifier.save_model("models/text_domain_classifier_v2.pkl")
    
    print("\n模型测试完成，已保存到 models/text_domain_classifier_v2.pkl")

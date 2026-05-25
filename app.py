"""
VOC批量异常风险识别与预警系统 v3.2
VOC Batch Anomaly Detection & Risk Alert System

独立作品 — 多模型 AI 引擎架构
支持: Ollama(本地免费) | DeepSeek V4 | Gemini 2.0 Flash(免费) | Groq(免费)
技术栈: Python + Streamlit + scikit-learn + Plotly + Multi-LLM
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import re

# 尝试导入可选依赖
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VOC异常风险识别系统 - AI Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════

MODELS = {
    "stat-only": {
        "name": "统计引擎（默认）",
        "provider": "内置",
        "icon": "📊",
        "model_id": None,
        "key_required": False,
        "key_name": None,
        "base_url": None,
        "sdk_type": "stat",
        "description": "TF-IDF + KMeans 聚类 + 时间序列检测，无需网络/Key",
        "speed": "⚡ 即时",
        "cost": "免费",
    },
    "ollama-qwen": {
        "name": "Qwen2.5 3B (本地)",
        "provider": "Ollama",
        "icon": "💻",
        "model_id": "qwen2.5:3b",
        "key_required": False,
        "key_name": None,
        "base_url": "http://localhost:11434/v1",
        "sdk_type": "openai",
        "description": "本地运行，完全免费，无需网络",
        "speed": "🚀 快（本地）",
        "cost": "免费",
    },
    "deepseek": {
        "name": "DeepSeek V4",
        "provider": "DeepSeek",
        "icon": "🐋",
        "model_id": "deepseek-chat",
        "key_required": True,
        "key_name": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "sdk_type": "openai",
        "description": "中文能力最强，需注册获取 API Key",
        "speed": "⚡ 较快",
        "cost": "¥1/百万tokens",
    },
    "gemini": {
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "icon": "🌐",
        "model_id": "gemini-2.0-flash",
        "key_required": True,
        "key_name": "GEMINI_API_KEY",
        "base_url": None,
        "sdk_type": "gemini",
        "description": "Google 免费额度：1500次/天",
        "speed": "⚡ 较快",
        "cost": "免费（15次/分钟）",
    },
    "groq": {
        "name": "Llama 3.3 70B (Groq)",
        "provider": "Groq",
        "icon": "⚡",
        "model_id": "llama-3.3-70b-versatile",
        "key_required": True,
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "sdk_type": "openai",
        "description": "Groq 免费额度：30次/分钟",
        "speed": "🔥 极快",
        "cost": "免费（30次/分钟）",
    },
}

# ═══════════════════════════════════════════════════════════
# 多模型客户端
# ═══════════════════════════════════════════════════════════

def get_client(model_key):
    config = MODELS[model_key]
    if config["sdk_type"] == "stat":
        return None
    if config["sdk_type"] == "openai":
        from openai import OpenAI
        if not config["key_required"]:
            if not check_ollama_available() or not check_ollama_model(config["model_id"]):
                return None
            return OpenAI(
                base_url=config["base_url"],
                api_key="ollama",
                timeout=20.0,
                max_retries=0,
            )
        key = os.getenv(config["key_name"], "") or st.session_state.get(f"key_{model_key}", "")
        if not key:
            return None
        return OpenAI(base_url=config["base_url"], api_key=key, timeout=30.0, max_retries=0)
    elif config["sdk_type"] == "gemini":
        import google.generativeai as genai
        key = os.getenv("GEMINI_API_KEY", "") or st.session_state.get("key_gemini", "")
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel(config["model_id"])
    return None


@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_available():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", 11434))
        s.close()
        return result == 0
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_model(model_id="qwen2.5:3b"):
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=2.0,
            max_retries=0,
        )
        models = client.models.list()
        return model_id in [m.id for m in models]
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# AI 语义聚类 + 预警建议
# ═══════════════════════════════════════════════════════════

def llm_semantic_cluster(texts, model_key, client):
    """使用 LLM 对 VOC 文本进行语义级聚类"""
    if not client or len(texts) < 3:
        return []

    config = MODELS[model_key]
    sample = texts[:150]
    texts_block = "\n".join([f"{i+1}. {t[:150]}" for i, t in enumerate(sample)])

    prompt = f"""你是电商平台VOC风险监控专家。以下是 {len(sample)} 条客户之声(VOC)文本。请进行语义级聚类，识别其中的批量异常事件。

VOC列表：
{texts_block}

返回 JSON 数组（不要 markdown 标记）：
[{{
    "topic": "异常事件名称（简洁概括）",
    "description": "1-2句话描述该异常事件",
    "size": 12,
    "sample_indices": [1, 3, 5],
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "severity": "🔴 红色预警 或 🟠 橙色预警 或 🟡 黄色预警",
    "trend": "上升中 / 高峰 / 回落 / 持续",
    "root_cause": "可能的根因（1句话）",
    "affected_scope": "预估影响范围描述（如：涉及X个SKU/覆盖Y地区/影响Z类用户）",
    "sentiment_distribution": "愤怒X% 焦虑Y% 平静Z%（根据VOC语气推断）",
    "escalation_risk": "高/中/低 — 升级为PR危机/社交媒体事件的可能性",
    "financial_risk": "预估财务风险等级（高/中/低）及简要说明",
    "early_warning_signals": "最早出现的信号描述（哪些VOC最先出现、关键词是什么）",
    "detection_confidence": 0.9,
    "recommended_action": "针对性响应建议（3步以内，含具体话术方向）"
}}]

要求：
- 只返回 >=3条聚集的异常主题
- severity: >=15条→红色, 8-14条→橙色, 3-7条→黄色
- escalation_risk: 涉及媒体/法律/群体维权→高, 情绪激烈→中, 普通批量→低
- detection_confidence: 0-1之间，表示对该聚类是否真实异常的置信度
- 无异常返回空数组 []
- 严格 JSON 数组格式"""

    try:
        if config["sdk_type"] == "openai":
            r = client.chat.completions.create(
                model=config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=2000,
            )
            raw = r.choices[0].message.content.strip()
        elif config["sdk_type"] == "gemini":
            r = client.generate_content(prompt)
            raw = r.text.strip()
        else:
            return []

        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.startswith("```")]
            raw = "\n".join(lines)
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
# 统计引擎（Baseline）
# ═══════════════════════════════════════════════════════════

STOP_WORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "这个", "那个", "可以", "然后", "因为", "所以", "但是", "还是", "已经",
    "我们", "你们", "他们", "怎么", "什么", "为什么", "哪里",
    "一下", "一点", "已经", "正在", "比较", "非常", "应该", "可能",
    "吗", "呢", "吧", "啊", "哦", "嗯", "的", "地", "得",
    "收到", "好的", "好滴", "好哒", "好嘞", "行吧", "好吧", "可以的",
    "谢谢", "感谢", "不客气", "没事", "没关系", "没问题",
    "您好", "你好", "在的", "在呢", "亲", "亲亲",
    "知道了", "收到啦", "明白了", "清楚了", "了解了", "晓得了",
    "嗯嗯", "哦哦", "啊啊", "哈哈", "呵呵", "嘿嘿",
    "哈", "呀", "嘛", "哎", "唉", "喂", "~",
])

DOMAIN_KEYWORDS = set([
    "退款", "退货", "赔付", "赔偿", "物流", "快递", "发货", "配送", "集运", "包裹",
    "质量", "破损", "假货", "瑕疵", "材质", "掉色", "氧化", "纯银", "镀银", "铜",
    "态度", "骂人", "敷衍", "推诿", "不理", "挂断", "投诉", "曝光", "12315", "315",
    "延迟", "积压", "中转", "签收", "丢件", "商家", "客服", "平台", "差价",
    "承诺", "回复", "联系", "申请", "订单", "处理", "解决",
])

SENSITIVE_PATTERNS = [
    "曝光", "315", "12315", "工商", "媒体", "微博", "小红书", "抖音",
    "集体", "维权", "举报", "起诉", "法院", "律师", "死亡", "炸",
]

# ═══════════════════════════════════════════════════════════
# 事件类型层级分类
# ═══════════════════════════════════════════════════════════

EVENT_TAXONOMY = {
    "商品问题": {
        "icon": "⚠️",
        "children": {
            "质量缺陷": ["质量", "坏了", "破损", "开胶", "断裂", "起球", "褪色", "缩水", "变形"],
            "材质不符": ["材质", "纯棉", "纯银", "含银量", "纯度", "镀", "成分", "面料", "不是纯"],
            "假冒伪劣": ["假货", "假冒", "仿冒", "伪劣", "山寨", "假的", "冒充"],
            "过期变质": ["过期", "变质", "发霉", "霉变", "哈喇", "馊", "臭了", "坏了不能"],
            "描述不符": ["描述不符", "和图片不一样", "色差", "实物不符", "不一样", "虚假宣传", "和详情页"],
        },
    },
    "物流问题": {
        "icon": "📦",
        "children": {
            "延迟未达": ["迟迟", "还没到", "没收到", "等待", "等了一个", "等了好", "未收到", "延迟"],
            "包裹丢失": ["丢件", "弄丢", "丢了", "不见了", "没收到货", "找不到"],
            "包装破损": ["包装破", "盒子扁", "外包装", "包装损", "压坏", "碎了"],
            "物流信息异常": ["物流信息", "不更新", "没更新", "查不到", "物流显示", "物流异常"],
            "集运积压": ["集运", "积压", "中转", "卡在", "滞留", "海关", "台湾流"],
        },
    },
    "服务问题": {
        "icon": "😠",
        "children": {
            "态度恶劣": ["态度", "骂人", "挂断", "不耐烦", "冷漠", "恶劣", "凶"],
            "推诿不处理": ["推诿", "踢皮球", "推卸", "不处理", "没人管", "不管", "要我找"],
            "回复慢/不回复": ["回复慢", "不理", "不回", "没人理", "才回", "不回复", "联系不上"],
            "承诺未兑现": ["承诺", "说好的", "答应", "说给", "允诺", "骗", "忽悠"],
            "退款到账慢": ["退款", "到账", "退到", "退了", "没退", "退钱", "退差价"],
        },
    },
    "合规风险": {
        "icon": "🚨",
        "children": {
            "虚假宣传": ["虚假宣传", "夸大", "误导", "虚假", "不是真的"],
            "食品安全": ["食品", "吃了", "喝了", "拉肚子", "腹泻", "过敏", "中毒", "异物"],
            "价格/促销争议": ["降价", "差价", "优惠券", "满减", "活动", "涨价"],
            "隐私/安全": ["信息泄露", "骚扰", "短信", "隐私", "电话轰炸"],
        },
    },
    "咨询建议": {
        "icon": "💬",
        "children": {
            "使用咨询": ["怎么", "如何", "能不能", "可以吗", "请教", "问一下"],
            "退换货咨询": ["退货", "换货", "怎么退", "退不了", "退货入口"],
            "账户/订单查询": ["订单", "查", "在哪", "修改地址", "发票", "记录"],
        },
    },
}


def keyword_event_classify(text):
    """基于关键词将VOC分类到一级+二级事件类型"""
    if not isinstance(text, str) or not text.strip():
        return "其他", "未分类"

    text_lower = text.lower()
    best_level1 = "其他"
    best_level2 = "未分类"
    best_score = 0

    for level1, cfg in EVENT_TAXONOMY.items():
        for level2, keywords in cfg["children"].items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_level1 = level1
                best_level2 = level2

    return best_level1, best_level2


# 可编辑副本（供规则配置面板使用）
_event_taxonomy_editable = {
    k: {"icon": v["icon"], "children": dict(v["children"])}
    for k, v in EVENT_TAXONOMY.items()
}


def smart_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    if HAS_JIEBA:
        words = jieba.lcut(text)
    else:
        chinese_chunks = re.findall(r'[一-鿿]+', text)
        words = []
        for chunk in chinese_chunks:
            for i in range(len(chunk)):
                for j in [2, 3, 4]:
                    if i + j <= len(chunk):
                        words.append(chunk[i:i+j])
    filtered = []
    for w in words:
        w = w.strip()
        if len(w) < 2 or w.isdigit():
            continue
        if w in DOMAIN_KEYWORDS:
            filtered.append(w)
            continue
        if w in STOP_WORDS:
            continue
        filtered.append(w)
    return filtered


def extract_keywords_from_texts(texts, top_n=15):
    all_words = []
    for text in texts:
        all_words.extend(smart_tokenize(text))
    return Counter(all_words).most_common(top_n)


def stat_cluster_texts(texts, n_clusters=None):
    """TF-IDF + KMeans 统计聚类"""
    if not HAS_SKLEARN or len(texts) < 3:
        return fallback_grouping(texts)
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=smart_tokenize,
            token_pattern=None,
            max_features=100,
            min_df=1,
            max_df=0.9,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception:
        return fallback_grouping(texts)
    if n_clusters is None:
        n_clusters = max(2, min(len(texts) // 3, 8))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
    except Exception:
        return fallback_grouping(texts)
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)
    results = []
    for label, indices in clusters.items():
        subset = [texts[i] for i in indices]
        keywords = extract_keywords_from_texts(subset, top_n=5)
        topic = "、".join([kw for kw, _ in keywords[:3]]) if keywords else f"主题{label+1}"
        results.append({"cluster_id": label, "topic_name": topic, "size": len(indices),
                        "keywords": keywords, "indices": indices, "sample_text": subset[0][:80] if subset else ""})
    results.sort(key=lambda x: x["size"], reverse=True)
    return results


def fallback_grouping(texts):
    groups = defaultdict(list)
    assigned = set()
    for i, text in enumerate(texts):
        if i in assigned:
            continue
        words_i = set(smart_tokenize(text))
        group = [i]
        assigned.add(i)
        for j, other in enumerate(texts):
            if j in assigned:
                continue
            words_j = set(smart_tokenize(other))
            if len(words_i & words_j) >= 2:
                group.append(j)
                assigned.add(j)
                words_i |= words_j
        groups[f"group_{i}"] = group
    results = []
    for gid, indices in groups.items():
        if len(indices) < 2:
            continue
        subset = [texts[i] for i in indices]
        keywords = extract_keywords_from_texts(subset, top_n=5)
        topic = "、".join([kw for kw, _ in keywords[:3]]) if keywords else gid
        results.append({"cluster_id": gid, "topic_name": topic, "size": len(indices),
                        "keywords": keywords, "indices": indices, "sample_text": subset[0][:80] if subset else ""})
    results.sort(key=lambda x: x["size"], reverse=True)
    return results


def detect_time_anomaly(df, date_col="create_time", window=7, threshold_multiplier=2.0):
    if date_col not in df.columns:
        return []
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col])
    df_copy["date"] = df_copy[date_col].dt.date
    daily_counts = df_copy.groupby("date").size().sort_index()
    anomalies = []
    if len(daily_counts) >= window:
        for i in range(window, len(daily_counts)):
            baseline_mean = daily_counts.iloc[i-window:i].mean()
            current = daily_counts.iloc[i]
            if baseline_mean > 0 and current > baseline_mean * threshold_multiplier:
                anomalies.append({"date": daily_counts.index[i], "current_count": int(current),
                                  "baseline_avg": round(baseline_mean, 1), "ratio": round(current / baseline_mean, 1)})
    return anomalies


def assess_alert_level(cluster_size, time_ratio=None, has_sensitive=False):
    score = 0
    if cluster_size >= 15:
        score += 3
    elif cluster_size >= 8:
        score += 2
    elif cluster_size >= 3:
        score += 1
    if time_ratio is not None:
        if time_ratio >= 3.0:
            score += 3
        elif time_ratio >= 2.0:
            score += 2
        elif time_ratio >= 1.5:
            score += 1
    if has_sensitive:
        score += 2
    if score >= 6:
        return "🔴 红色预警", "red"
    elif score >= 3:
        return "🟠 橙色预警", "orange"
    elif score >= 1:
        return "🟡 黄色预警", "yellow"
    return "⚪ 无预警", "grey"


def check_sensitive(text):
    if not isinstance(text, str):
        return False
    return any(kw in text for kw in SENSITIVE_PATTERNS)


# ═══════════════════════════════════════════════════════════
# 版本历史
# ═══════════════════════════════════════════════════════════

VERSION_HISTORY = [
    {
        "version": "v3.2", "date": "2026-05-20",
        "title": "预警可视化增强 — 风险日历 + 趋势预测 + 一键报告",
        "changes": [
            "新增风险日历热力图：日期×事件类型矩阵，颜色深浅=风险密度",
            "新增趋势预测：时间序列向前外推3天预测线+置信区间",
            "新增一键风险报告：汇总所有异常事件+趋势+建议，可下载Markdown",
        ],
        "advantage": "不仅发现异常，更做可视化预测和闭环报告输出",
        "icon": "🗓️",
    },
    {
        "version": "v3.1", "date": "2026-05-20",
        "title": "多模型 AI 引擎架构",
        "changes": [
            "新增 Ollama Qwen2.5 本地模型支持（免费，无需 Key）",
            "新增 Gemini 2.0 Flash / Groq Llama 3.3 免费 API 选项",
            "新增 DeepSeek V4 选项",
            "新增 AI 语义聚类引擎，替代纯统计聚类",
            "新增 AI 预警建议生成（根因分析+响应方案）",
            "侧边栏模型选择器，一键切换引擎",
            "统计引擎作为默认 baseline，无 AI 时正常运行",
        ],
        "advantage": "统计引擎秒级出结果，AI 引擎提供语义级聚类和智能建议",
        "icon": "🔄",
    },
    {
        "version": "v2.0", "date": "2026-05-17",
        "title": "统计引擎版",
        "changes": [
            "TF-IDF + KMeans 文本聚类",
            "滑动窗口时间序列异常检测",
            "三级预警体系（红/橙/黄）+ 敏感词加权",
            "Jieba 分词 + 停用词过滤 + 领域词白名单",
            "Plotly 时间趋势图 + 聚类分布图",
            "预警处置状态追踪（待处理/处理中/已闭环）",
        ],
        "advantage": "不依赖外部服务，纯统计模型快速识别批量异常",
        "icon": "📊",
    },
    {
        "version": "v1.0", "date": "2026-05-17",
        "title": "原型版",
        "changes": [
            "基础 VOC 数据处理和聚类框架",
            "模拟数据生成器",
            "Streamlit UI 搭建",
        ],
        "advantage": "验证了批量异常检测的产品方向",
        "icon": "🚀",
    },
]

# ═══════════════════════════════════════════════════════════
# 问题反馈
# ═══════════════════════════════════════════════════════════

ISSUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "issue_reports.jsonl")


def save_issue(issue_data):
    issue_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ISSUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(issue_data, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════
# 模拟数据
# ═══════════════════════════════════════════════════════════

def generate_sample_data():
    """生成200条VOC模拟数据，含3组埋点批量异常"""
    np.random.seed(42)
    records = []
    normal_templates = [
        "买的东西物流好慢啊，等了快一周了还没收到",
        "客服回复太慢了，等了好久才回我一句",
        "收到的商品和图片颜色不太一样，有点失望",
        "衣服质量还行，就是码数偏小",
        "怎么退货啊，找不到退货入口",
        "物流显示签收了我没收到货，快递员电话打不通",
        "优惠券用不了，显示不符合条件",
        "客服态度还行，帮我解决了问题",
        "收到的包装破了，不过东西没坏",
        "买了三件只发了两件，少发了一件",
        "鞋子穿着不太舒服想换一双",
        "能不能改收货地址，已经下单了",
        "为什么我的退款还没到账，已经五天了",
        "商品描述说是纯棉的结果不是的",
        "客服帮我查了物流，态度挺好的",
        "发货速度挺快的，第二天就到了",
        "这个商品降价了能不能退差价",
        "买的食品快过期了，不敢吃",
        "同城配送为什么也要两天",
        "发票怎么申请电子发票",
        "收到的电子产品没有中文说明书",
        "包装太简陋了盒子都扁了",
        "物流信息三天没更新了不知道货到哪了",
        "买之前咨询客服态度很好售后就变了",
        "收到的颜色和图片差太多了",
    ]
    for i in range(164):
        day = np.random.randint(1, 31)
        date = f"2026-04-{day:02d}"
        template = np.random.choice(normal_templates)
        amount = np.random.choice([39, 68, 99, 129, 188, 259, 329, 399, 459, 599])
        records.append([f"N{i:03d}", template, amount, date])

    # 批量异常1：银饰品材质不符
    silver = [
        "买的银手镯说是999纯银拿回来一测含银量只有60%，这是欺诈吧",
        "银项链掉色太严重了戴了两天脖子都绿了，根本不是纯银",
        "S925银戒指戴了一周就发黑，以前买的银饰戴一年都不会这样",
        "银耳钉收到就有铜锈味，检测出来是铜镀银",
        "银饰套盒里面好几件都有氧化斑点，商家还说是正常现象",
        "银手镯材质不对，多个买家都有一样的问题",
        "又是银饰又是材质不符，平台能不能管管这类商家",
        "S925套链收到根本不是银的，戴了一次就过敏起疹子",
        "银饰手镯纯银承诺完全是虚假宣传，材质检测不过关",
        "这个银饰商家太黑了，银手镯完全不是纯银的",
        "买的银戒指掉色露出红色底色，根本就是铜的假冒伪劣",
        "银项链的材质和详情页完全不符，证书也是假的",
    ]
    for i, text in enumerate(silver):
        day = 22 + (i % 4)
        records.append([f"B1-{i:02d}", text, 199 + (i % 5) * 100, f"2026-04-{day:02d}"])

    # 批量异常2：集运物流积压
    logistics = [
        "台湾集运包裹已经等了一个月了还没到，物流显示一直在中转",
        "集运包裹卡在中转站不动了客服也联系不上，我的货到底在哪",
        "台湾流向的物流是不是出问题了，集运订单20多天没更新物流了",
        "三个集运包裹全部积压，物流公司说是运力不够",
        "集运台湾的订单物流超过30天，打了多次电话都说在处理",
        "集运包裹显示异常，客服说在协调但等了一周没进展",
        "台湾集运商到底什么时候能恢复，我的订单都等着用呢",
        "物流积压这么严重平台应该主动通知消费者",
        "我的集运包裹在中转站卡了两周了完全没有移动",
        "台湾方向的物流为什么全部停掉了，四个包裹全部积压",
        "集运台湾的包裹一直没有物流更新，要求退款",
        "因为物流积压我的货全部滞留了，这损失谁来承担",
    ]
    for i, text in enumerate(logistics):
        day = 26 + (i % 4)
        records.append([f"B2-{i:02d}", text, 299 + (i % 6) * 100, f"2026-04-{day:02d}"])

    # 批量异常3：婴幼儿食品安全
    food = [
        "买的婴幼儿米粉打开有股哈喇味，已经过期了吧不敢给孩子吃",
        "儿童辅食罐头里面有异物，黑色的不知道是什么东西太恶心了",
        "买的宝宝饼干包装破损而且受潮了，软的完全不脆",
        "婴幼儿奶粉冲不开有结块，宝宝喝了拉肚子",
        "儿童零食配料表写的和实物明显不符，添加剂比写的多",
        "婴儿辅食泥有变质味道，而且瓶盖凸起说明已经坏了",
        "买的幼儿奶片表面发霉了，生产日期还是上个月的",
        "儿童营养面过期了还在卖，煮出来一股陈味",
    ]
    for i, text in enumerate(food):
        day = 14 + (i % 3)
        records.append([f"B3-{i:02d}", text, 88 + (i % 4) * 50, f"2026-04-{day:02d}"])

    return pd.DataFrame(records, columns=["voc_id", "voc_text", "order_amount", "create_time"])


# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "voc_working_df": None,
        "selected_model": "stat-only",
        "key_deepseek": "", "key_gemini": "", "key_groq": "",
        "alert_threshold_multiplier": 2.0,
        "alert_min_cluster_size": 3,
        "alert_window_days": 7,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def show_model_selector():
    st.subheader("🤖 分析引擎选择")
    ollama_running = check_ollama_available()
    ollama_has_model = check_ollama_model() if ollama_running else False

    model_options = {}
    model_status = {}
    model_options["stat-only"] = "📊 统计引擎（默认·即时可用）"
    model_status["stat-only"] = "✅ 已就绪"

    for key, cfg in MODELS.items():
        if key == "stat-only":
            continue
        elif key == "ollama-qwen":
            if ollama_running and ollama_has_model:
                model_options[key] = f"{cfg['icon']} {cfg['name']}"
                model_status[key] = "✅ 已就绪"
            elif ollama_running:
                model_options[key] = f"{cfg['icon']} {cfg['name']} (需拉取模型)"
                model_status[key] = "⚠️ 模型未拉取"
            else:
                model_options[key] = f"{cfg['icon']} {cfg['name']} (Ollama 未启动)"
                model_status[key] = "❌ 未启动"
        else:
            has_key = bool(st.session_state.get(f"key_{key}"))
            model_options[key] = f"{cfg['icon']} {cfg['name']} ({'需' if not has_key else '已配'} Key)"
            model_status[key] = "🔑 需 Key" if not has_key else "✅ 已配置"

    current = st.session_state.get("selected_model", "stat-only")
    if current not in model_options:
        current = "stat-only"
    selected = st.selectbox(
        "选择分析引擎",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=list(model_options.keys()).index(current),
    )
    st.session_state["selected_model"] = selected
    cfg = MODELS[selected]

    with st.container(border=True):
        st.caption(f"**{cfg['icon']} {cfg['name']}** | {cfg['provider']} | {cfg['speed']} | {cfg['cost']}")
        st.caption(cfg["description"])
        st.caption(f"状态: {model_status.get(selected, '')}")

    if cfg["key_required"]:
        kv = st.text_input(f"{cfg['provider']} API Key", type="password",
                           value=st.session_state.get(f"key_{selected}", ""),
                           placeholder=f"输入 {cfg['provider']} API Key...",
                           key=f"api_key_{selected}")
        if kv:
            st.session_state[f"key_{selected}"] = kv

    if selected == "ollama-qwen":
        if not ollama_running:
            st.warning("⚠️ Ollama 未运行。安装: https://ollama.com/download/windows")
        elif not ollama_has_model:
            st.warning("⚠️ 运行 `ollama pull qwen2.5:3b` 拉取模型")

    return selected


def show_rules_config():
    with st.expander("📋 规则配置", expanded=False):
        st.caption("调整统计引擎的停用词、领域词和预警参数")

        st.markdown("**预警参数**")
        st.session_state["alert_threshold_multiplier"] = st.slider(
            "异常阈值倍数", 1.2, 5.0, st.session_state["alert_threshold_multiplier"], 0.1)
        st.session_state["alert_min_cluster_size"] = st.slider(
            "最小聚类规模", 2, 20, st.session_state["alert_min_cluster_size"])
        st.session_state["alert_window_days"] = st.slider(
            "基线窗口（天）", 3, 14, st.session_state["alert_window_days"])

        st.divider()
        st.markdown("**领域关键词**（白名单，始终保留）")
        key = "cfg_domain_kw"
        if key not in st.session_state:
            st.session_state[key] = "、".join(sorted(DOMAIN_KEYWORDS))
        new_domain = st.text_area("领域词", value=st.session_state[key], key="input_domain", height=68, label_visibility="collapsed")
        st.session_state[key] = new_domain
        DOMAIN_KEYWORDS.clear()
        DOMAIN_KEYWORDS.update(kw.strip() for kw in new_domain.replace(",", "、").split("、") if kw.strip())

        st.divider()
        st.markdown("**敏感词**（触发预警加权）")
        sk = "cfg_sensitive_kw"
        if sk not in st.session_state:
            st.session_state[sk] = "、".join(SENSITIVE_PATTERNS)
        new_sens = st.text_area("敏感词", value=st.session_state[sk], key="input_sensitive", height=50, label_visibility="collapsed")
        st.session_state[sk] = new_sens
        SENSITIVE_PATTERNS.clear()
        SENSITIVE_PATTERNS.extend(kw.strip() for kw in new_sens.replace(",", "、").split("、") if kw.strip())

        # 事件类型配置
        st.divider()
        st.markdown("**事件类型分类规则**（一级→二级→关键词）")
        for l1, l1_cfg in EVENT_TAXONOMY.items():
            with st.expander(f"{l1_cfg['icon']} {l1}", expanded=False):
                for l2, keywords in l1_cfg["children"].items():
                    key = f"event_{l1}_{l2}"
                    if key not in st.session_state:
                        st.session_state[key] = "、".join(keywords)
                    new_val = st.text_area(
                        f"{l2}",
                        value=st.session_state[key],
                        key=f"input_{key}",
                        height=40,
                        label_visibility="visible",
                    )
                    st.session_state[key] = new_val
                    kw_list = [kw.strip() for kw in new_val.replace(",", "、").split("、") if kw.strip()]
                    EVENT_TAXONOMY[l1]["children"][l2] = kw_list

        if st.button("🔄 恢复默认规则", width='stretch'):
            for k in ["cfg_domain_kw", "cfg_sensitive_kw"]:
                st.session_state.pop(k, None)
            # 恢复事件类型
            for l1, l1_cfg in EVENT_TAXONOMY.items():
                for l2 in l1_cfg["children"]:
                    key = f"event_{l1}_{l2}"
                    st.session_state.pop(key, None)
                    keywords = _event_taxonomy_editable[l1]["children"][l2]
                    EVENT_TAXONOMY[l1]["children"][l2] = list(keywords)
            st.rerun()


def show_issue_report():
    with st.expander("🐛 问题反馈", expanded=False):
        st.caption("发现 bug、聚类不准、体验问题？请提交")
        title = st.text_input("问题标题", key="voc_issue_title", placeholder="一句话描述问题")
        desc = st.text_area("详细描述", key="voc_issue_desc", height=80)
        c1, c2 = st.columns(2)
        with c1:
            itype = st.selectbox("类型", ["聚类不准确", "系统Bug", "功能建议", "数据问题", "界面体验", "其他"], key="voc_issue_type")
        with c2:
            iurg = st.selectbox("紧急度", ["一般", "重要", "紧急"], key="voc_issue_urg")
        if st.button("提交问题", key="voc_issue_submit", type="primary", width='stretch'):
            if not title.strip():
                st.error("请填写标题")
            else:
                save_issue({"title": title, "description": desc, "type": itype, "urgency": iurg})
                st.success("已提交！")
                st.rerun()


def show_risk_calendar(df):
    """风险日历热力图"""
    st.subheader("风险事件日历热力图")
    if "create_time" not in df.columns or "一级事件" not in df.columns:
        st.info("需要 create_time 和事件类型字段")
        return
    dft = df.copy()
    dft["create_time"] = pd.to_datetime(dft["create_time"], errors="coerce")
    dft = dft.dropna(subset=["create_time"])
    dft["date"] = dft["create_time"].dt.date
    heat = dft.groupby(["date", "一级事件"]).size().reset_index(name="count")
    if heat.empty:
        st.info("数据不足")
        return
    pivot = heat.pivot(index="date", columns="一级事件", values="count").fillna(0)
    daily_total = dft.groupby("date").size()
    threshold = daily_total.mean() * 2
    anomaly_dates = set(daily_total[daily_total > threshold].index)
    fig = px.imshow(pivot.values.T, x=[str(d) for d in pivot.index], y=list(pivot.columns),
                    title="VOC 风险日历（横轴=日期，纵轴=事件类型，颜色深=量大，异常日总量>日均2x）",
                    color_continuous_scale="YlOrRd", aspect="auto")
    fig.update_xaxes(side="top", tickangle=45)
    st.plotly_chart(fig, width='stretch')
    st.caption("颜色越深=当日该类事件越多；可在时间序列Tab查看异常日详情")


def generate_risk_report(df, stat_clusters, time_anomalies, ai_clusters):
    lines = []
    lines.append("# VOC 批量异常风险报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"数据范围: {len(df)} 条 VOC")
    lines.append("")
    lines.append("## 统计聚类异常")
    for c in stat_clusters[:5]:
        if c["size"] >= st.session_state.get("alert_min_cluster_size", 3):
            lines.append(f"- **{c['topic_name']}**: {c['size']}条 | 关键词: {'、'.join([kw for kw,_ in c['keywords'][:5]])}")
    lines.append("")
    lines.append("## 时间异常点")
    for a in time_anomalies[:5]:
        lines.append(f"- {a['date']}: 当日{a['current_count']}条 (基线{a['baseline_avg']}条, 增幅{a['ratio']}x)")
    if ai_clusters:
        lines.append("")
        lines.append("## AI 语义识别")
        for a in ai_clusters:
            lines.append(f"- {a.get('topic','')}: {a.get('description','')} | 趋势:{a.get('trend','')} | 升级风险:{a.get('escalation_risk','')}")
    lines.append("")
    lines.append("## 建议动作")
    lines.append("1. 定位涉事商家/商品/物流商，核实影响面")
    lines.append("2. 评估是否升级至值班主管")
    lines.append("3. 制定批量处理策略，输出标准话术通知一线")
    return "\n".join(lines)


def show_sidebar():
    with st.sidebar:
        st.header("⚙️ 控制面板")
        selected_model = show_model_selector()
        cfg = MODELS[selected_model]
        st.divider()
        st.subheader("📤 数据加载")
        uploaded = st.file_uploader("上传VOC数据CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                tc = None
                for c in ["voc_text", "complaint_text", "客诉文本", "投诉内容", "text", "content"]:
                    if c in df.columns:
                        tc = c; break
                if tc:
                    df["voc_text"] = df[tc].astype(str)
                    st.session_state["voc_working_df"] = df
                    st.success(f"已加载 {len(df)} 条")
                else:
                    st.error("未找到文本列")
                    st.info("请确认 CSV 至少包含以下任一列名：voc_text、complaint_text、客诉文本、投诉内容、text、content。")
            except Exception as e:
                st.error(f"读取失败: {e}")
                st.info("建议先使用「加载200条模拟VOC数据」完成演示；上传文件请使用 UTF-8 CSV，并保留文本列。")
        st.divider()
        st.subheader("📥 快速体验")
        if st.button("🎲 加载200条模拟VOC数据", type="primary", width='stretch'):
            st.session_state["voc_working_df"] = generate_sample_data()
            st.rerun()
        if st.session_state.get("voc_working_df") is not None:
            if st.button("🗑️ 清除数据", width='stretch'):
                st.session_state["voc_working_df"] = None
                st.rerun()
        st.divider()
        st.subheader("📋 当前引擎")
        with st.container(border=True):
            st.markdown(f"**{cfg['icon']} {cfg['name']}**")
            st.caption(f"提供商: {cfg['provider']} | {cfg['speed']} | {cfg['cost']}")
        st.divider()
        show_rules_config()
        show_issue_report()
        with st.expander("📜 版本演进", expanded=False):
            for e in VERSION_HISTORY:
                st.markdown(f"{e['icon']} **{e['version']}** — {e['title']}{' `当前`' if e == VERSION_HISTORY[0] else ''}")
                st.caption(f"{e['date']} | {len(e['changes'])}项改动 | {e['advantage'][:40]}...")
                if e != VERSION_HISTORY[-1]:
                    st.markdown("│")
        # 一键报告
        if st.session_state.get("voc_working_df") is not None:
            st.divider()
            st.subheader("📄 风险报告")
            if st.button("🎯 一键生成风险报告", type="primary", width='stretch'):
                df_tmp = st.session_state["voc_working_df"]
                # 重新运行统计聚类以获取最新结果
                stat_c = stat_cluster_texts(df_tmp["voc_text"].tolist())
                time_a = detect_time_anomaly(df_tmp, threshold_multiplier=st.session_state.get("alert_threshold_multiplier", 2.0),
                                             window=st.session_state.get("alert_window_days", 7))
                report = generate_risk_report(df_tmp, stat_c, time_a, [])
                st.session_state["risk_report"] = report
                st.download_button("⬇️ 下载报告 (Markdown)", data=report,
                                   file_name=f"VOC风险报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                   mime="text/markdown", width='stretch')
            if "risk_report" in st.session_state:
                with st.expander("预览报告", expanded=False):
                    st.markdown(st.session_state["risk_report"])

        return selected_model


def show_welcome():
    st.markdown("""
    ### 👋 批量异常识别与服务风险预警 v3.2

    这个 Demo 模拟服务 AI 工作流的第二步：从大量 VOC 中发现**聚集事件、敏感风险、时间异常和可响应预警**。

    **默认无需 API Key**：统计引擎（TF-IDF + KMeans + 时间检测）可直接演示；接入 AI 引擎后可增强语义聚类和风险总结。

    #### 🎯 双引擎能力

    | 模块 | 统计引擎 | AI 引擎 |
    |------|---------|---------|
    | 聚类方式 | TF-IDF + KMeans 数学聚类 | 语义理解，发现「措辞不同事件相同」的隐藏聚集 |
    | 异常检测 | 滑动窗口时间序列 | 语义级事件聚类 |
    | 预警建议 | 固定 SOP 模板 | 针对性根因分析+响应方案 |

    #### 🚀 面试演示路径
    1. 直接点击侧边栏 **加载 200 条模拟 VOC 数据**。
    2. 查看事件类型、统计聚类、时间序列和预警面板。
    3. 点击 **一键生成风险报告**，展示从识别到响应建议的闭环。
    """)
    st.info("👈 在侧边栏选择引擎 → 加载数据 → 查看聚类和预警结果")


def show_voc_results(df, stat_clusters, time_anomalies, ai_clusters, model_key, client):
    st.subheader("📊 概览")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("VOC总数", len(df))
    with c2:
        st.metric("统计聚类", len(stat_clusters))
    with c3:
        has_ai = len(ai_clusters) > 0
        st.metric("AI 异常事件", len(ai_clusters), delta="语义聚类" if has_ai else "未启用")
    with c4:
        st.metric("时间异常点", len(time_anomalies))
    with c5:
        sc = sum(1 for t in df["voc_text"] if check_sensitive(t))
        st.metric("含敏感词", sc)
    st.divider()

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 事件类型分布", "🔍 聚类分析", "📈 时间序列", "🗓️ 风险日历", "🚨 预警面板", "📝 数据明细"]
    )

    with tab0:
        # 对全量数据做事件类型分类
        event_types = [keyword_event_classify(t) for t in df["voc_text"]]
        df["一级事件"] = [et[0] for et in event_types]
        df["二级事件"] = [et[1] for et in event_types]

        st.subheader("事件类型层级分布")

        col_e1, col_e2 = st.columns([3, 2])
        with col_e1:
            # 旭日图：一级→二级层级
            sunburst_data = []
            for l1, l1_cfg in EVENT_TAXONOMY.items():
                l1_count = len(df[df["一级事件"] == l1])
                if l1_count > 0:
                    sunburst_data.append({
                        "一级": l1, "二级": "", "数量": l1_count, "icon": l1_cfg["icon"],
                    })
                    for l2 in l1_cfg["children"]:
                        l2_count = len(df[(df["一级事件"] == l1) & (df["二级事件"] == l2)])
                        if l2_count > 0:
                            sunburst_data.append({
                                "一级": l1, "二级": l2, "数量": l2_count, "icon": "",
                            })

            if sunburst_data:
                sun_df = pd.DataFrame(sunburst_data)
                fig_sun = px.sunburst(
                    sun_df, path=["一级", "二级"], values="数量",
                    title="事件类型一二级分布（点击下钻）",
                    color="一级",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_sun.update_traces(textinfo="label+percent entry")
                st.plotly_chart(fig_sun, width='stretch')

        with col_e2:
            # 一级事件统计卡片
            st.markdown("**一级事件统计**")
            l1_counts = df["一级事件"].value_counts()
            for l1, cnt in l1_counts.items():
                icon = EVENT_TAXONOMY.get(l1, {}).get("icon", "📌")
                pct = cnt / len(df) * 100
                st.metric(f"{icon} {l1}", f"{cnt} 条", delta=f"{pct:.0f}%")

        # 二级事件明细
        st.divider()
        st.markdown("**二级事件下钻**")
        for l1, l1_cfg in EVENT_TAXONOMY.items():
            l1_df = df[df["一级事件"] == l1]
            if l1_df.empty:
                continue
            with st.expander(f"{l1_cfg['icon']} {l1}（{len(l1_df)} 条）", expanded=(len(l1_df) > len(df) * 0.1)):
                for l2 in l1_cfg["children"]:
                    l2_count = len(l1_df[l1_df["二级事件"] == l2])
                    if l2_count > 0:
                        pct = l2_count / len(df) * 100
                        st.markdown(f"• **{l2}**: {l2_count} 条 ({pct:.1f}%)")
                        # 取该二级事件的样本
                        samples = l1_df[l1_df["二级事件"] == l2]["voc_text"].head(2).tolist()
                        for s in samples:
                            st.caption(f"  「{s[:60]}...」")

    with tab1:
        st.subheader("VOC 聚类分析")
        if not stat_clusters and not ai_clusters:
            st.info("未检测到明显聚类")
        else:
            # 统计聚类图
            if stat_clusters:
                st.markdown("#### 📊 统计引擎聚类（TF-IDF + KMeans）")
                cd = pd.DataFrame([{"主题": c["topic_name"][:20], "数量": c["size"]} for c in stat_clusters])
                cd = cd.sort_values("数量", ascending=True)
                fig = px.bar(cd, x="数量", y="主题", orientation="h", title="统计聚类分布",
                             color_discrete_sequence=["#2196F3"])
                st.plotly_chart(fig, width='stretch')

            # AI 聚类结果
            if ai_clusters:
                st.divider()
                st.markdown(f"#### 🤖 {MODELS[model_key]['name']} 语义聚类")
                for a in ai_clusters:
                    with st.expander(f"{a.get('severity', '🟡')} {a.get('topic', '未命名')} — 约{a.get('size', '?')}条", expanded=True):
                        x1, x2 = st.columns(2)
                        with x1:
                            st.markdown(f"**描述**: {a.get('description', '')}")
                            st.markdown(f"**关键词**: {'、'.join(a.get('keywords', []))}")
                            st.markdown(f"**趋势**: {a.get('trend', '-')} | **根因**: {a.get('root_cause', '-')}")
                            st.markdown(f"**影响范围**: {a.get('affected_scope', '-')}")
                            st.markdown(f"**情绪分布**: {a.get('sentiment_distribution', '-')}")
                            st.markdown(f"**早期信号**: {a.get('early_warning_signals', '-')}")
                        with x2:
                            st.markdown(f"**升级风险**: {a.get('escalation_risk', '-')}")
                            st.markdown(f"**财务风险**: {a.get('financial_risk', '-')}")
                            st.markdown(f"**置信度**: {a.get('detection_confidence', 0):.0%}" if isinstance(a.get('detection_confidence'), (int, float)) else f"**置信度**: {a.get('detection_confidence', '-')}")
                            st.markdown("**响应建议**")
                            st.info(a.get("recommended_action", ""))

        # 全局关键词
        if stat_clusters:
            st.subheader("全局高频关键词")
            kw = extract_keywords_from_texts(df["voc_text"].tolist(), top_n=30)
            kwd = pd.DataFrame(kw, columns=["关键词", "频次"])
            fig = px.bar(kwd.head(20), x="关键词", y="频次", title="Top 20 关键词")
            st.plotly_chart(fig, width='stretch')

    with tab2:
        st.subheader("时间序列异常检测")
        if "create_time" in df.columns:
            dft = df.copy()
            dft["create_time"] = pd.to_datetime(dft["create_time"], errors="coerce")
            dft = dft.dropna(subset=["create_time"])
            dft["date"] = dft["create_time"].dt.date
            dc = dft.groupby("date").size().reset_index(name="count")
            w = st.session_state.get("alert_window_days", 7)
            t = st.session_state.get("alert_threshold_multiplier", 2.0)
            dc["MA"] = dc["count"].rolling(window=w, min_periods=1).mean().round(1)
            dc["Threshold"] = dc["MA"] * t

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dc["date"], y=dc["count"], mode="lines+markers", name="日VOC量",
                                     line=dict(color="#2196F3", width=2)))
            fig.add_trace(go.Scatter(x=dc["date"], y=dc["MA"], mode="lines", name=f"{w}日均值",
                                     line=dict(color="#9E9E9E", width=1.5, dash="dash")))
            fig.add_trace(go.Scatter(x=dc["date"], y=dc["Threshold"], mode="lines", name=f"阈值({t}x)",
                                     line=dict(color="#FF5722", width=1, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(255,87,34,0.08)"))

            ad = set(a["date"] for a in time_anomalies)
            ap = dc[dc["date"].isin(ad)]
            if not ap.empty:
                fig.add_trace(go.Scatter(x=ap["date"], y=ap["count"], mode="markers", name="异常点",
                                         marker=dict(color="#FF0000", size=12, symbol="x")))
            # 趋势预测
            if len(dc) >= 3:
                last_vals = dc["count"].tail(3).tolist()
                last_ma = dc["MA"].iloc[-1]
                date_span = pd.to_datetime(dc["date"].iloc[-1]) - pd.to_datetime(dc["date"].iloc[-3])
                slope = (last_vals[-1] - last_vals[0]) / max(1, date_span.days)
                from datetime import timedelta
                pred_dates = [dc["date"].iloc[-1] + timedelta(days=i) for i in range(1, 4)]
                pred_vals = [last_ma + slope * i for i in range(1, 4)]
                pred_upper = [v * 1.3 for v in pred_vals]
                pred_lower = [v * 0.7 for v in pred_vals]
                fig.add_trace(go.Scatter(x=pred_dates, y=pred_vals, mode="lines+markers", name="趋势预测",
                                         line=dict(color="#FF9800", width=2, dash="dot"),
                                         marker=dict(size=8, symbol="diamond")))
                fig.add_trace(go.Scatter(x=pred_dates + pred_dates[::-1], y=pred_upper + pred_lower[::-1],
                                         fill="toself", fillcolor="rgba(255,152,0,0.12)", line=dict(width=0),
                                         name="预测区间"))
                pred_note = f"若趋势持续，预计 {pred_dates[-1]} 日 VOC 量约 {pred_vals[-1]:.0f} 条"
                st.info(f"📈 {pred_note}")

            fig.update_layout(height=400, title="VOC日趋势与异常检测", hovermode="x unified")
            st.plotly_chart(fig, width='stretch')
            if time_anomalies:
                st.warning(f"检测到 {len(time_anomalies)} 个时间异常点")
        else:
            st.info("数据未包含 create_time 字段")

    with tab3:
        show_risk_calendar(df)

    with tab4:
        st.subheader("🚨 预警面板")
        has_kw_alerts = False
        has_ai_alerts = len(ai_clusters) > 0

        # ── 关键词规则预警 ──
        st.markdown("### 🔧 关键词规则预警")
        st.caption("基于 TF-IDF 聚类 + 敏感词匹配 + 时间增幅的规则引擎预警")
        kw_alerts = []
        min_sz = st.session_state.get("alert_min_cluster_size", 3)
        for c in stat_clusters:
            if c["size"] < min_sz:
                continue
            hs = any(check_sensitive(df["voc_text"].iloc[i]) for i in c["indices"])
            tr = next((a["ratio"] for a in time_anomalies if str(a.get("label", "")) == str(c["cluster_id"])), None)
            lv, col = assess_alert_level(c["size"], tr, hs)
            if lv == "⚪ 无预警":
                continue
            kw_alerts.append({"预警等级": lv, "主题": c["topic_name"], "规模": c["size"],
                             "时间增幅": f"{tr}x" if tr else "-", "含敏感词": "是" if hs else "否",
                             "color": col, "cluster": c})
        if kw_alerts:
            has_kw_alerts = True
            kw_alerts.sort(key=lambda x: {"🔴": 0, "🟠": 1, "🟡": 2}.get(x["预警等级"][:2], 3))
            for i, a in enumerate(kw_alerts):
                with st.expander(f"{a['预警等级']} {a['主题']} — 规模{a['规模']}条 | 增幅{a['时间增幅']}", expanded=(i < 2)):
                    x1, x2 = st.columns(2)
                    with x1:
                        st.markdown(f"**规模**: {a['规模']}条 | **增幅**: {a['时间增幅']} | **敏感词**: {a['含敏感词']}")
                        c = a["cluster"]
                        st.markdown(f"**Top关键词**: {' | '.join([f'{kw}({cnt})' for kw, cnt in c['keywords'][:8]])}")
                        st.caption("检测方法: TF-IDF + KMeans 聚类 + 滑动窗口时间序列")
                    with x2:
                        st.markdown("**标准响应流程**")
                        st.info("1. 定位涉事商家/商品/物流商\n2. 评估升级必要性\n3. 制定批量处理策略\n4. 输出话术通知一线")
                    st.radio("处置状态", ["待处理", "处理中", "已闭环"], horizontal=True, key=f"kw_status_{i}")
        else:
            st.success("✅ 关键词规则未检测到预警")

        # ── AI 智能识别预警 ──
        st.divider()
        st.markdown("### 🤖 AI 智能识别预警")
        st.caption("基于 LLM 语义理解的批量异常识别 - 可发现「措辞不同但事件相同」的隐藏聚集")

        if ai_clusters:
            ai_alerts_sorted = sorted(ai_clusters, key=lambda x: x.get("size", 0), reverse=True)
            for i, a in enumerate(ai_alerts_sorted):
                sev = a.get("severity", "🟡 黄色预警")
                with st.expander(f"{sev} {a.get('topic', '未命名')} — 约{a.get('size', '?')}条 | 置信度{a.get('detection_confidence', 0):.0%}", expanded=(i < 3)):
                    x1, x2, x3 = st.columns([1, 1, 1])
                    with x1:
                        st.markdown(f"**描述**: {a.get('description', '-')}")
                        st.markdown(f"**趋势**: {a.get('trend', '-')}")
                        st.markdown(f"**根因**: {a.get('root_cause', '-')}")
                        st.markdown(f"**影响范围**: {a.get('affected_scope', '-')}")
                    with x2:
                        st.markdown(f"**情绪分布**: {a.get('sentiment_distribution', '-')}")
                        st.markdown(f"**升级风险**: {a.get('escalation_risk', '-')}")
                        st.markdown(f"**财务风险**: {a.get('financial_risk', '-')}")
                        st.markdown(f"**早期信号**: {a.get('early_warning_signals', '-')}")
                    with x3:
                        st.markdown(f"**关键词**: {'、'.join(a.get('keywords', []))}")
                        st.caption(f"检测方法: LLM 语义聚类")
                        st.markdown("**AI 响应建议**")
                        st.info(a.get("recommended_action", "-"))
                    st.radio("处置状态", ["待处理", "处理中", "已闭环"], horizontal=True, key=f"ai_status_{i}")
        else:
            if has_kw_alerts:
                st.info("💡 AI 引擎未启用或未发现额外异常。切换 AI 引擎可获取语义级智能识别。")
            else:
                st.success("✅ AI 智能识别未检测到预警")
            st.caption("AI 识别优势: 语义理解 / 根因分析 / 升级风险评估 / 财务影响预估 / 个性化响应建议")

    with tab5:
        st.subheader("VOC数据明细")
        dd = df.copy()
        dd["聚类标签"] = "未聚类"
        for c in stat_clusters:
            for idx in c["indices"]:
                if idx < len(dd):
                    dd.iloc[idx, dd.columns.get_loc("聚类标签")] = c["topic_name"]
        dd["含敏感词"] = dd["voc_text"].apply(lambda x: "是" if check_sensitive(x) else "否")
        if "一级事件" in df.columns:
            dd["一级事件"] = df["一级事件"]
            dd["二级事件"] = df["二级事件"]
        dc = [c for c in ["voc_id", "voc_text", "一级事件", "二级事件", "聚类标签", "order_amount", "create_time", "含敏感词"] if c in dd.columns]
        st.dataframe(dd[dc], width='stretch', height=400)
        st.download_button("⬇️ 导出CSV", data=dd.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                           file_name=f"VOC分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    init_session()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🚨 VOC批量异常风险识别与预警系统")
        st.caption("多引擎架构 | 统计聚类 + AI 语义聚类 + 时间序列检测")
    with c2:
        st.metric("版本", "v3.2", delta="预警可视化增强")
        st.metric("本地 AI", "就绪" if (check_ollama_available() and check_ollama_model()) else "待启动")

    st.divider()
    selected_model = show_sidebar()
    cfg = MODELS[selected_model]
    is_stat = (cfg["sdk_type"] == "stat")

    if st.session_state.get("voc_working_df") is None:
        show_welcome()
    else:
        df = st.session_state["voc_working_df"].copy()
        if "voc_text" not in df.columns:
            st.error("数据缺少 voc_text 列")
            return

        # 统计引擎始终运行
        with st.spinner("📊 统计引擎分析中..."):
            stat_clusters = stat_cluster_texts(df["voc_text"].tolist())
            time_anomalies = detect_time_anomaly(
                df, threshold_multiplier=st.session_state.get("alert_threshold_multiplier", 2.0),
                window=st.session_state.get("alert_window_days", 7),
            )

        # AI 引擎
        client = None if is_stat else get_client(selected_model)
        ai_clusters = []

        if is_stat:
            st.success(f"📊 统计引擎分析完成。共发现 {len(stat_clusters)} 个聚类、{len(time_anomalies)} 个时间异常点。切换到 AI 引擎可获取语义级聚类。")
        elif client:
            st.info(f"🤖 {cfg['name']} 语义聚类中...")
            ai_clusters = llm_semantic_cluster(df["voc_text"].tolist(), selected_model, client)
            if ai_clusters:
                st.success(f"✅ AI 发现 {len(ai_clusters)} 个语义异常事件")
            else:
                st.info(f"💡 AI 未发现额外异常事件，统计引擎结果如下")
        else:
            if cfg["key_required"]:
                st.info(f"💡 {cfg['name']} 需要 API Key。当前展示统计引擎结果。")
            elif selected_model == "ollama-qwen":
                if not check_ollama_available():
                    st.warning("⚠️ Ollama 未运行。当前展示统计引擎结果。")
                else:
                    st.warning("⚠️ 模型未拉取。运行 `ollama pull qwen2.5:3b`")

        show_voc_results(df, stat_clusters, time_anomalies, ai_clusters, selected_model, client)


if __name__ == "__main__":
    main()

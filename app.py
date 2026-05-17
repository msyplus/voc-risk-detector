"""
VOC批量异常风险识别与预警工具
VOC Batch Anomaly Detection & Risk Alert System

独立作品项目 — 面试演示用
技术栈: Python + Streamlit + jieba + scikit-learn + plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import json
import os

# 尝试导入 jieba，如果失败则降级使用简单分词
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VOC批量异常风险识别",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 文本处理工具
# ═══════════════════════════════════════════════════════════

# 停用词表（精简版 + 客服语气词）
STOP_WORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "这个", "那个", "可以", "然后", "因为", "所以", "但是", "还是", "已经",
    "我们", "你们", "他们", "怎么", "什么", "为什么", "怎么", "哪里",
    "一下", "一点", "已经", "正在", "比较", "非常", "应该", "可能",
    "吗", "呢", "吧", "啊", "哦", "嗯", "的", "地", "得",
    # 客服语气词/填充词（不承载业务含义）
    "收到", "好的", "好滴", "好哒", "好嘞", "行吧", "好吧", "可以的",
    "谢谢", "感谢", "不客气", "没事", "没关系", "没问题",
    "您好", "你好", "在的", "在呢", "亲", "亲亲",
    "知道了", "收到啦", "明白了", "清楚了", "了解了", "晓得了",
    "嗯嗯", "哦哦", "啊啊", "哈哈", "呵呵", "嘿嘿",
    "哈", "呀", "嘛", "哎", "唉", "喂", "~",
])

# 客诉领域关键词白名单（优先保留）
DOMAIN_KEYWORDS = set([
    "退款", "退货", "赔付", "赔偿", "物流", "快递", "发货", "配送", "集运", "包裹",
    "质量", "破损", "假货", "瑕疵", "材质", "掉色", "氧化", "纯银", "镀银", "铜",
    "态度", "骂人", "敷衍", "推诿", "不理", "挂断", "投诉", "曝光", "12315", "315",
    "延迟", "积压", "中转", "签收", "丢件", "商家", "客服", "平台", "差价",
    "承诺", "回复", "联系", "收到", "申请", "订单", "处理", "解决",
])


def smart_tokenize(text):
    """智能分词：优先使用jieba，否则回退到简单切分"""
    if not isinstance(text, str) or not text.strip():
        return []

    if HAS_JIEBA:
        words = jieba.lcut(text)
    else:
        # 简单回退：按常见分隔符切分 + 2-4字滑动窗口
        import re
        # 先提取连续中文字段
        chinese_chunks = re.findall(r'[一-鿿]+', text)
        words = []
        for chunk in chinese_chunks:
            for i in range(len(chunk)):
                for j in [2, 3, 4]:
                    if i + j <= len(chunk):
                        words.append(chunk[i:i+j])

    # 过滤停用词 + 保留领域关键词 + 过滤纯数字/单字
    filtered = []
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        if w in DOMAIN_KEYWORDS:
            filtered.append(w)
            continue
        if w in STOP_WORDS:
            continue
        if w.isdigit():
            continue
        filtered.append(w)

    return filtered


def extract_keywords_from_texts(texts, top_n=15):
    """从文本列表中提取高频关键词"""
    all_words = []
    for text in texts:
        all_words.extend(smart_tokenize(text))
    counter = Counter(all_words)
    return counter.most_common(top_n)


def cluster_texts(texts, n_clusters=None):
    """对文本进行TF-IDF聚类"""
    if not HAS_SKLEARN or len(texts) < 3:
        # 回退：基于关键词重合度分组
        return fallback_grouping(texts)

    # TF-IDF向量化
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=smart_tokenize,
            max_features=100,
            min_df=1,
            max_df=0.9,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception:
        return fallback_grouping(texts)

    # 自动确定聚类数
    if n_clusters is None:
        n_samples = len(texts)
        n_clusters = max(2, min(n_samples // 3, 8))

    if n_clusters >= n_samples:
        n_clusters = max(2, n_samples // 2)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
    except Exception:
        return fallback_grouping(texts)

    # 为每个聚类提取关键词
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)

    results = []
    for label, indices in clusters.items():
        cluster_texts_subset = [texts[i] for i in indices]
        keywords = extract_keywords_from_texts(cluster_texts_subset, top_n=5)
        topic_name = "、".join([kw for kw, _ in keywords[:3]]) if keywords else f"主题{label+1}"

        results.append({
            "cluster_id": label,
            "topic_name": topic_name,
            "size": len(indices),
            "keywords": keywords,
            "indices": indices,
            "sample_text": cluster_texts_subset[0][:80] if cluster_texts_subset else "",
        })

    results.sort(key=lambda x: x["size"], reverse=True)
    return results


def fallback_grouping(texts):
    """关键词重合度分组（不依赖sklearn的回退方案）"""
    groups = defaultdict(list)
    assigned = set()

    for i, text in enumerate(texts):
        if i in assigned:
            continue
        words_i = set(smart_tokenize(text))
        group = [i]
        assigned.add(i)

        for j, other_text in enumerate(texts):
            if j in assigned:
                continue
            words_j = set(smart_tokenize(other_text))
            overlap = words_i & words_j
            if len(overlap) >= 2:  # 至少2个共同关键词
                group.append(j)
                assigned.add(j)
                words_i |= words_j

        groups[f"group_{i}"] = group

    results = []
    for gid, indices in groups.items():
        if len(indices) < 2:
            continue
        cluster_texts_subset = [texts[i] for i in indices]
        keywords = extract_keywords_from_texts(cluster_texts_subset, top_n=5)
        topic_name = "、".join([kw for kw, _ in keywords[:3]]) if keywords else gid

        results.append({
            "cluster_id": gid,
            "topic_name": topic_name,
            "size": len(indices),
            "keywords": keywords,
            "indices": indices,
            "sample_text": cluster_texts_subset[0][:80] if cluster_texts_subset else "",
        })

    results.sort(key=lambda x: x["size"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════
# 时间序列异常检测
# ═══════════════════════════════════════════════════════════

def detect_time_anomaly(df, date_col="create_time", cluster_label_col=None, window=7, threshold_multiplier=2.0):
    """检测时间序列上的异常增幅"""
    if date_col not in df.columns:
        return []

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col])
    df_copy["date"] = df_copy[date_col].dt.date

    anomalies = []

    if cluster_label_col and cluster_label_col in df_copy.columns:
        # 按聚类分组检测
        for label in df_copy[cluster_label_col].unique():
            subset = df_copy[df_copy[cluster_label_col] == label]
            daily_counts = subset.groupby("date").size().sort_index()

            if len(daily_counts) < window:
                continue

            for i in range(window, len(daily_counts)):
                baseline_mean = daily_counts.iloc[i-window:i].mean()
                current = daily_counts.iloc[i]
                if baseline_mean > 0 and current > baseline_mean * threshold_multiplier:
                    anomalies.append({
                        "date": daily_counts.index[i],
                        "label": label,
                        "current_count": int(current),
                        "baseline_avg": round(baseline_mean, 1),
                        "ratio": round(current / baseline_mean, 1),
                    })
    else:
        # 全局检测
        daily_counts = df_copy.groupby("date").size().sort_index()
        if len(daily_counts) >= window:
            for i in range(window, len(daily_counts)):
                baseline_mean = daily_counts.iloc[i-window:i].mean()
                current = daily_counts.iloc[i]
                if baseline_mean > 0 and current > baseline_mean * threshold_multiplier:
                    anomalies.append({
                        "date": daily_counts.index[i],
                        "label": "全局",
                        "current_count": int(current),
                        "baseline_avg": round(baseline_mean, 1),
                        "ratio": round(current / baseline_mean, 1),
                    })

    return anomalies


# ═══════════════════════════════════════════════════════════
# 预警等级评估
# ═══════════════════════════════════════════════════════════

def assess_alert_level(cluster_size, time_ratio=None, has_sensitive_words=False):
    """综合评估预警等级"""
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

    if has_sensitive_words:
        score += 2

    if score >= 6:
        return "🔴 红色预警", "red"
    elif score >= 3:
        return "🟠 橙色预警", "orange"
    elif score >= 1:
        return "🟡 黄色预警", "yellow"
    return "⚪ 无预警", "grey"


SENSITIVE_PATTERNS = [
    "曝光", "315", "12315", "工商", "媒体", "微博", "小红书", "抖音",
    "集体", "维权", "举报", "起诉", "法院", "律师", "死亡", "炸",
]


def check_sensitive(text):
    """检查文本是否含敏感词"""
    if not isinstance(text, str):
        return False
    return any(kw in text for kw in SENSITIVE_PATTERNS)


# ═══════════════════════════════════════════════════════════
# 示例数据生成
# ═══════════════════════════════════════════════════════════

def generate_sample_data():
    """生成200条VOC模拟数据，含2组埋点批量异常"""
    np.random.seed(42)
    records = []

    # 正常分散数据（170条，前30天）
    normal_templates = [
        "买的东西物流好慢啊，等了快一周了还没收到",
        "客服回复太慢了，等了好久才回我一句",
        "收到的商品和图片颜色不太一样，有点失望",
        "衣服质量还行，就是码数偏小",
        "怎么退货啊，找不到退货入口",
        "物流显示签收了我没收到货，快递员电话打不通",
        "优惠券用不了，显示不符合条件，可是明明是符合条件的",
        "客服态度还行，帮我解决了问题",
        "收到的包装破了，不过东西没坏",
        "买了三件只发了两件，少发了一件",
        "鞋子穿着不太舒服想换一双，怎么换",
        "能不能改收货地址，已经下单了",
        "为什么我的退款还没到账，已经五天了",
        "商品描述说是纯棉的结果不是，有点失望",
        "客服帮我查了物流，态度挺好的",
        "发货速度挺快的，第二天就到了",
        "这个商品降价了能不能退差价",
        "买的食品快过期了，不敢吃",
        "同城配送为什么也要两天",
        "发票怎么申请电子发票",
        "收到的电子产品没有中文说明书",
        "包装太简陋了收到的时候盒子都扁了",
        "物流信息三天没更新了不知道货到哪了",
        "买之前咨询客服态度很好，售后就变了",
        "收到的颜色和图片差太多了好失望",
    ]

    for i in range(170):
        day = np.random.randint(1, 31)
        date = f"2026-04-{day:02d}"
        template = np.random.choice(normal_templates)
        amount = np.random.choice([39, 68, 99, 129, 188, 259, 329, 399, 459, 599])
        records.append([f"N{i:03d}", template, amount, date])

    # 批量异常1：银饰品材质不符（15条，集中在第22-25天）
    silver_texts = [
        "买的银手镯说是999纯银拿回来一测含银量只有60%，这是欺诈吧",
        "银项链掉色太严重了戴了两天脖子都绿了，根本不是纯银",
        "S925银戒指戴了一周就发黑，以前买的银饰戴一年都不会这样",
        "银耳钉收到就有铜锈味，检测出来是铜镀银，太坑人了",
        "银饰套盒里面好几件都有氧化斑点，商家还说是正常现象",
        "银手镯材质不对，多个买家都有一样的问题",
        "又是银饰又是材质不符，平台能不能管管这类商家",
        "S925套链收到根本不是银的，戴了一次就过敏起疹子",
        "银饰手镯纯银承诺完全是虚假宣传，材质检测不过关",
        "这个银饰商家太黑了，银手镯完全不是纯银的",
        "买的银戒指掉色露出红色底色，根本就是铜的假冒伪劣",
        "银项链的材质和详情页完全不符，证书也是假的",
        "两对银耳环都掉色了，里面露出来的是红色的",
        "银饰商品材质检测不过关，要求退款并彻查商家",
        "第三次买到假银饰了，平台到底有没有质量管控",
    ]

    for i, text in enumerate(silver_texts):
        day = 22 + (i % 4)
        date = f"2026-04-{day:02d}"
        amount = 199 + (i % 5) * 100
        records.append([f"B1-{i:02d}", text, amount, date])

    # 批量异常2：集运物流积压（15条，集中在第26-29天）
    logistics_texts = [
        "台湾集运包裹已经等了一个月了还没到，物流显示一直在中转",
        "集运包裹卡在中转站不动了客服也联系不上，我的货到底在哪",
        "台湾流向的物流是不是出问题了，集运订单20多天没更新物流了",
        "三个集运包裹全部积压，物流公司说是运力不够",
        "集运台湾的订单物流超过30天，打了多次电话都说在处理",
        "集运包裹显示异常，客服说在协调但等了一周没进展",
        "台湾集运商到底什么时候能恢复，我的订单都等着用呢",
        "物流积压这么严重平台应该主动通知消费者而不是让我们自己发现",
        "我的集运包裹在中转站卡了两周了完全没有移动",
        "台湾方向的物流为什么全部停掉了，四个包裹全部积压",
        "集运台湾的包裹一直没有物流更新，要求退款",
        "因为物流积压我的货全部滞留了，这损失谁来承担",
        "台湾集运到底是物流问题还是海关问题，总得给个说法",
        "一个月了集运包裹还没到台湾，严重怀疑弄丢了",
        "集运台湾物流太差了，整个月的包裹都卡住了",
    ]

    for i, text in enumerate(logistics_texts):
        day = 26 + (i % 4)
        date = f"2026-04-{day:02d}"
        amount = 299 + (i % 6) * 100
        records.append([f"B2-{i:02d}", text, amount, date])

    df = pd.DataFrame(records, columns=["voc_id", "voc_text", "order_amount", "create_time"])
    return df


# ═══════════════════════════════════════════════════════════
# Streamlit 界面
# ═══════════════════════════════════════════════════════════

def main():
    # 初始化
    if "voc_working_df" not in st.session_state:
        st.session_state["voc_working_df"] = None
    if "alert_rules" not in st.session_state:
        st.session_state["alert_rules"] = {
            "threshold_multiplier": 2.0,
            "min_cluster_size": 3,
            "window_days": 7,
            "exclude_keywords": [],
        }

    # ---- 顶部 ----
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚨 VOC批量异常风险识别与预警")
        st.caption("上传VOC数据 → 文本聚类 → 时间序列检测 → 预警输出 → 闭环追踪")
    with col2:
        deps_ok = HAS_JIEBA and HAS_SKLEARN
        st.metric("引擎状态", "完整模式" if deps_ok else "基础模式",
                  delta="jieba+sklearn" if deps_ok else "关键词回退")

    st.divider()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("⚙️ 操作面板")

        uploaded_file = st.file_uploader("📤 上传VOC数据CSV", type=["csv"],
                                         help="需包含 voc_text 列（客诉文本），可选 create_time")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                text_col = None
                for col in ["voc_text", "complaint_text", "客诉文本", "投诉内容", "text", "content"]:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    for col in df.columns:
                        if df[col].dtype == "object" and df[col].str.len().mean() > 15:
                            text_col = col
                            break

                if text_col is not None:
                    df["voc_text"] = df[text_col].astype(str)
                    st.session_state["voc_working_df"] = df
                    st.success(f"✅ 已加载 {len(df)} 条VOC数据")
                else:
                    st.error("未找到文本列，请确保包含 voc_text 列")
            except Exception as e:
                st.error(f"读取失败: {e}")

        st.divider()

        # 预警规则配置
        st.subheader("🔧 预警规则配置")
        st.session_state["alert_rules"]["threshold_multiplier"] = st.slider(
            "异常阈值倍数", 1.2, 5.0, 2.0, 0.1,
            help="当日量超过前N日均值的倍数即触发预警")
        st.session_state["alert_rules"]["min_cluster_size"] = st.slider(
            "最小聚类规模", 2, 20, 3,
            help="聚类内VOC数量少于此值则忽略")
        st.session_state["alert_rules"]["window_days"] = st.slider(
            "基线窗口（天）", 3, 14, 7,
            help="计算基线均值的天数范围")

        st.divider()

        # 加载示例数据
        st.subheader("📥 示例数据")
        if st.button("加载200条模拟VOC数据", type="primary", use_container_width=True):
            st.session_state["voc_working_df"] = generate_sample_data()
            st.rerun()

        if st.session_state["voc_working_df"] is not None:
            if st.button("🗑️ 清除数据", use_container_width=True):
                st.session_state["voc_working_df"] = None
                st.rerun()

        st.divider()
        st.caption("💡 核心功能：文本聚类发现批量问题 + 时间序列检测异常增幅 + 预警闭环追踪")

    # ---- 主区域 ----
    if st.session_state["voc_working_df"] is None:
        show_welcome_voc()
    else:
        df = st.session_state["voc_working_df"]
        rules = st.session_state["alert_rules"]

        with st.spinner("正在分析VOC数据..."):
            cluster_results = cluster_texts(df["voc_text"].tolist())
            time_anomalies = detect_time_anomaly(
                df, date_col="create_time",
                threshold_multiplier=rules["threshold_multiplier"],
            )

        show_voc_results(df, cluster_results, time_anomalies, rules)


def show_welcome_voc():
    st.markdown("""
    ### 👋 欢迎使用VOC批量异常风险识别系统

    本工具将服务运营中的批量客诉治理经验产品化，支持以下能力：

    | 功能 | 说明 |
    |------|------|
    | 🔍 **文本聚类** | TF-IDF + KMeans自动发现VOC中的相似主题聚集 |
    | 📈 **时间序列检测** | 监控各类主题的日增幅，超过基线阈值自动预警 |
    | 🚨 **三级预警** | 综合聚类规模 + 时间增幅 + 敏感词，输出红/橙/黄预警 |
    | 📊 **可视化分析** | 聚类分布图、时间趋势线、关键词云、预警日历 |
    | 📝 **预警闭环** | 预警详情页 + 处置状态追踪（待处理/处理中/已闭环） |

    ---

    👈 请在左侧边栏上传CSV或点击加载示例数据开始体验。
    """)


def show_voc_results(df, cluster_results, time_anomalies, rules):
    """展示VOC分析结果"""
    # KPI 指标行
    st.subheader("📊 概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("VOC总数", len(df))
    with col2:
        st.metric("发现主题聚类", len(cluster_results))
    with col3:
        high_alerts = [c for c in cluster_results if c["size"] >= rules["min_cluster_size"] * 3]
        st.metric("高风险聚类", len(high_alerts), delta="需关注" if high_alerts else "正常")
    with col4:
        st.metric("时间异常点", len(time_anomalies))
    with col5:
        sensitive_count = sum(1 for t in df["voc_text"] if check_sensitive(t))
        st.metric("含敏感词VOC", sensitive_count)

    st.divider()

    # Tab页
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 聚类分析", "📈 时间序列", "🚨 预警面板", "📝 数据明细"])

    # ---- Tab1: 聚类分析 ----
    with tab1:
        st.subheader("VOC主题聚类分析")

        if not cluster_results:
            st.info("未检测到明显的主题聚类。VOC分布可能比较分散。")
        else:
            col_c1, col_c2 = st.columns([3, 2])

            with col_c1:
                # 聚类规模条形图
                cluster_df = pd.DataFrame([
                    {"主题": c["topic_name"][:20], "数量": c["size"],
                     "预警等级": assess_alert_level(c["size"],
                        next((a["ratio"] for a in time_anomalies if str(a.get("label", "")) == str(c["cluster_id"])), None),
                        any(check_sensitive(df["voc_text"].iloc[i]) for i in c["indices"]))[0]}
                    for c in cluster_results
                ])
                cluster_df = cluster_df.sort_values("数量", ascending=True)

                colors = {"🔴 红色预警": "#FF4444", "🟠 橙色预警": "#FFA726",
                          "🟡 黄色预警": "#FFEB3B", "⚪ 无预警": "#90A4AE"}

                fig_bar = px.bar(
                    cluster_df, x="数量", y="主题", color="预警等级",
                    color_discrete_map=colors, orientation="h",
                    title="VOC主题聚类分布",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                st.markdown("**聚类详情**")
                for c in cluster_results[:10]:
                    alert_level, _ = assess_alert_level(
                        c["size"],
                        next((a["ratio"] for a in time_anomalies if str(a.get("label", "")) == str(c["cluster_id"])), None),
                        any(check_sensitive(df["voc_text"].iloc[i]) for i in c["indices"])
                    )
                    with st.expander(f"{alert_level} {c['topic_name']}（{c['size']}条）"):
                        st.markdown(f"**Top关键词**: {' | '.join([f'{kw}({cnt})' for kw, cnt in c['keywords'][:8]])}")
                        st.markdown(f"**样例**: {c['sample_text']}...")

        # 关键词总览
        if cluster_results:
            st.subheader("全局高频关键词")
            all_texts = df["voc_text"].tolist()
            global_keywords = extract_keywords_from_texts(all_texts, top_n=30)

            kw_df = pd.DataFrame(global_keywords, columns=["关键词", "频次"])
            fig_kw = px.bar(kw_df.head(20), x="关键词", y="频次", title="Top 20 高频关键词")
            st.plotly_chart(fig_kw, use_container_width=True)

    # ---- Tab2: 时间序列 ----
    with tab2:
        st.subheader("时间序列异常检测")

        if "create_time" in df.columns:
            df_time = df.copy()
            df_time["create_time"] = pd.to_datetime(df_time["create_time"], errors="coerce")
            df_time = df_time.dropna(subset=["create_time"])
            df_time["date"] = df_time["create_time"].dt.date
            daily_counts = df_time.groupby("date").size().reset_index(name="count")

            # 添加移动平均线
            daily_counts["MA7"] = daily_counts["count"].rolling(window=7, min_periods=1).mean().round(1)
            daily_counts["Threshold"] = daily_counts["MA7"] * rules["threshold_multiplier"]

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=daily_counts["date"], y=daily_counts["count"],
                mode="lines+markers", name="日VOC量",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=6),
            ))
            fig_ts.add_trace(go.Scatter(
                x=daily_counts["date"], y=daily_counts["MA7"],
                mode="lines", name=f"{rules['window_days']}日均值",
                line=dict(color="#9E9E9E", width=1.5, dash="dash"),
            ))
            fig_ts.add_trace(go.Scatter(
                x=daily_counts["date"], y=daily_counts["Threshold"],
                mode="lines", name=f"预警阈值（{rules['threshold_multiplier']}x）",
                line=dict(color="#FF5722", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(255,87,34,0.08)",
            ))

            # 标记异常点
            anomaly_dates = set()
            for a in time_anomalies:
                anomaly_dates.add(a["date"])

            anomaly_points = daily_counts[daily_counts["date"].isin(anomaly_dates)]
            if not anomaly_points.empty:
                fig_ts.add_trace(go.Scatter(
                    x=anomaly_points["date"], y=anomaly_points["count"],
                    mode="markers", name="异常点",
                    marker=dict(color="#FF0000", size=12, symbol="x"),
                ))

            fig_ts.update_layout(
                height=400,
                title="VOC日趋势与异常检测",
                xaxis_title="日期",
                yaxis_title="VOC数量",
                hovermode="x unified",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            if time_anomalies:
                st.warning(f"⚠️ 检测到 **{len(time_anomalies)}** 个时间异常点")
                anomaly_df = pd.DataFrame(time_anomalies)
                anomaly_df = anomaly_df.rename(columns={
                    "date": "日期", "label": "关联聚类", "current_count": "当日量",
                    "baseline_avg": "基线均值", "ratio": "增幅倍数",
                })
                st.dataframe(anomaly_df, use_container_width=True)
            else:
                st.success("✅ 未检测到明显的时间异常")
        else:
            st.info("数据中未包含 create_time 字段，无法进行时间序列分析")

    # ---- Tab3: 预警面板 ----
    with tab3:
        st.subheader("🚨 综合预警面板")

        alerts = []
        for c in cluster_results:
            if c["size"] < rules["min_cluster_size"]:
                continue

            has_sensitive = any(check_sensitive(df["voc_text"].iloc[i]) for i in c["indices"])
            time_ratio = next((a["ratio"] for a in time_anomalies
                               if str(a.get("label", "")) == str(c["cluster_id"])), None)

            alert_level, color = assess_alert_level(c["size"], time_ratio, has_sensitive)
            if alert_level == "⚪ 无预警":
                continue

            alerts.append({
                "预警等级": alert_level,
                "主题": c["topic_name"],
                "聚类规模": c["size"],
                "时间增幅": f"{time_ratio}x" if time_ratio else "-",
                "含敏感词": "是" if has_sensitive else "否",
                "color": color,
                "cluster": c,
            })

        if alerts:
            alerts.sort(key=lambda x: {"🔴": 0, "🟠": 1, "🟡": 2}.get(x["预警等级"][:2], 3))

            for i, alert in enumerate(alerts):
                with st.expander(
                    f"{alert['预警等级']} {alert['主题']} — 规模{alert['聚类规模']}条 — 时间增幅{alert['时间增幅']}",
                    expanded=(i < 3),
                ):
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.markdown(f"**预警主题**: {alert['主题']}")
                        st.markdown(f"**聚类规模**: {alert['聚类规模']} 条VOC")
                        st.markdown(f"**时间增幅**: {alert['时间增幅']}")

                        c = alert["cluster"]
                        st.markdown(f"**关键词**: {' | '.join([f'{kw}({cnt})' for kw, cnt in c['keywords'][:8]])}")

                    with col_a2:
                        st.markdown("**📋 建议响应动作**")
                        st.info(f"""
                        1. 定位涉事商家/商品/物流商，核实影响面
                        2. 评估是否需要升级至值班主管
                        3. 制定批量处理策略：
                           - 自动拦截（低风险标准场景）
                           - 人工兜底（高风险/需个性化处理）
                        4. 输出标准话术模板，通知一线客服
                        5. 同步招商/治理/物流相关部门
                        """)

                        # 处置状态追踪
                        st.markdown("**处置状态**")
                        status = st.radio(
                            "标记状态",
                            ["待处理", "处理中", "已闭环"],
                            horizontal=True,
                            key=f"status_{i}",
                        )
                        if status == "已闭环":
                            st.success(f"✅ 已标记为闭环 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        else:
            st.success("✅ 当前未检测到需要预警的批量异常")

    # ---- Tab4: 数据明细 ----
    with tab4:
        st.subheader("VOC数据明细与导出")

        # 为数据添加聚类标签
        df_display = df.copy()
        df_display["聚类标签"] = "未聚类"
        for c in cluster_results:
            for idx in c["indices"]:
                if idx < len(df_display):
                    df_display.iloc[idx, df_display.columns.get_loc("聚类标签")] = c["topic_name"]

        df_display["含敏感词"] = df_display["voc_text"].apply(lambda x: "是" if check_sensitive(x) else "否")

        display_cols = [c for c in ["voc_id", "voc_text", "聚类标签", "order_amount", "create_time", "含敏感词"]
                        if c in df_display.columns]
        st.dataframe(df_display[display_cols], use_container_width=True, height=400)

        # 导出按钮
        st.download_button(
            label="⬇️ 下载完整分析结果CSV",
            data=df_display.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"VOC风险分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

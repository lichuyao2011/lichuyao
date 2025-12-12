# app_streamlit.py —— 三语共現 + BERT 语義重排（最终修复版 v3）
# (修复 w_sem=0.0 时因 tie-breaker 导致的排序不一致问题)

import os, re, json, sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 路径设定（确保与 .db、generateEmoji 同目录）
# =========================
ROOT = Path(__file__).parent.resolve()
db_path_cn = str(ROOT / "emoji_reco_pos_cn.db")
db_path_en = str(ROOT / "emoji_reco_freq.db")
db_path_ja = str(ROOT / "emoji_reco_pos_ja.db")

EMOJI_BASE_DIR = ROOT / "generateEmoji"          # 你的图片文件夹
EMOJI_JSON = EMOJI_BASE_DIR / "emojis.json"      # 图片映射文件（title -> src 或 [{title, src}, ...]）
THUMB_H = 32

# 隐藏侧边栏的展开/收起按钮
st.set_page_config(
    page_title="Emoji推薦",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed" # <-- 隐藏侧边栏
)

# =========================
# 工具与依赖（自动尝试安装）
# =========================
def _ensure(pkg: str):
    try:
        __import__(pkg); return True
    except Exception:
        try:
            os.system(f"{sys.executable} -m pip install {pkg} --quiet")
            __import__(pkg); return True
        except Exception:
            return False

# 尝试确保常用包（不是强制）
_HAVE_TORCH = _ensure("torch")
_HAVE_TRANS = _ensure("transformers")
_HAVE_EMOJI = _ensure("emoji")
_ = _ensure("openpyxl")
_ = _ensure("matplotlib")
_ = _ensure("janome")

if _HAVE_TORCH and _HAVE_TRANS:
    import torch
    from transformers import AutoTokenizer, AutoModel
else:
    torch = None
    AutoTokenizer = AutoModel = None

# janome（用于日语分词）
try:
    from janome.tokenizer import Tokenizer
    _ja_tokenizer = Tokenizer()
except Exception:
    os.system(f"{sys.executable} -m pip install janome --quiet")
    from janome.tokenizer import Tokenizer
    _ja_tokenizer = Tokenizer()

try:
    if _HAVE_EMOJI:
        import emoji as _emoji
    else:
        _emoji = None
except Exception:
    _emoji = None

# 关键：确保并导入 streamlit-keyup (这是正确的包名)
try:
    from st_keyup import st_keyup
    _HAVE_DEBOUNCE = True
except Exception:
    try:
        os.system(f"{sys.executable} -m pip install streamlit-keyup --quiet")
        from st_keyup import st_keyup
        _HAVE_DEBOUNCE = True
    except Exception as e:
        _HAVE_DEBOUNCE = False
        print(f"⚠️ streamlit-keyup not loaded: {e}")

# =========================
# 读取 emoji 映射（兼容 dict/list 两种结构）
# =========================
@st.cache_resource(show_spinner="正在加载 Emoji 图片映射...")
def _load_title2abs(json_path: Path, base_dir: Path):
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text("utf-8"))
    except Exception:
        try:
            data = json.loads(json_path.read_text("utf-8", errors="ignore"))
        except Exception:
            return {}
    def to_abs(src: str) -> str:
        s = str(src).replace("\\", "/")
        if s.startswith("http://") or s.startswith("https://"):
            return s
        return str((base_dir / s).resolve())
    title2abs = {}
    if isinstance(data, dict):
        for k, v in data.items():
            k = str(k).strip()
            if k: title2abs[k] = to_abs(v)
    elif isinstance(data, list):
        for item in data:
            t = str(item.get("title", "")).strip()
            s = item.get("src", "")
            if t and s: title2abs[t] = to_abs(s)
    return title2abs

TITLE2ABS = _load_title2abs(Path(EMOJI_JSON), EMOJI_BASE_DIR)

def _title_variants(t: str):
    t = (t or "").strip()
    if not t: return []
    core = t.strip("_")
    return [t, f"_{t}_", f"_{t}", f"{t}_", core, f"_{core}_", f"_{core}", f"{core}_"]

# =========================
# BERT 语义向量（多语）
# =========================
BERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner="正在加载 BERT 模型...")
def _bert_init():
    if not (_HAVE_TORCH and _HAVE_TRANS):
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        mdl = AutoModel.from_pretrained(BERT_MODEL_NAME)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        st.error(f"BERT 模型加载失败: {e}")
        return None, None

def _bert_ready(tok, mdl):
    return tok is not None and mdl is not None and torch is not None

def _embed_texts(texts, tok, mdl):
    if not _bert_ready(tok, mdl): return None
    if isinstance(texts, str): texts = [texts]
    try:
        batch = tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**batch)
            last = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            summed = (last * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            vec = (summed / count).cpu().numpy()
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        return vec
    except Exception:
        return None

_CN_EMOJI_EMB = {}

@st.cache_resource(show_spinner="正在预计算中文向量...")
def _build_cn_title_embeddings_cached(titles):
    tok, mdl = _bert_init()
    if not _bert_ready(tok, mdl): return {}
    vecs = _embed_texts(titles, tok, mdl)
    if vecs is None: return {}
    return {t: v for t, v in zip(titles, vecs)}

if TITLE2ABS:
    _CN_EMOJI_EMB.update(_build_cn_title_embeddings_cached(list(TITLE2ABS.keys())))

def _cos(a, b):
    return float(np.dot(a, b))

# 反查 emoji 英文短名（供英/日描述）
_EMO_NAME_CACHE = {}
def _name_from_emoji_char(ch: str) -> str:
    if not ch: return ""
    if ch in _EMO_NAME_CACHE: return _EMO_NAME_CACHE[ch]
    nm = ch
    try:
        if _emoji:
            nm = _emoji.demojize(ch, language='en')
            nm = nm.strip(":").replace("_", " ").strip() or ch
    except Exception:
        nm = ch
    _EMO_NAME_CACHE[ch] = nm
    return nm

def _desc_for_token(lang: str, token: str) -> str:
    # 中文用标题, 英/日用 demogize
    return token if lang == "中国語" else _name_from_emoji_char(token)

# =========================
# 分词与评分
# =========================
def _log1p(v: int) -> float:
    return float(np.log1p(max(0, int(v))))

_word_re = re.compile(r"[A-Za-z]+")
def _words_en(text: str):
    return list({t.lower() for t in _word_re.findall(text or "") if len(t) > 1})

def _tokenize_ja(text: str):
    if not text: return []
    keep = ("名詞", "動詞", "形容詞")
    toks = []
    for t in _ja_tokenizer.tokenize(text):
        head = t.part_of_speech.split(",")[0]
        if head not in keep: continue
        base = t.base_form if t.base_form != "*" else t.surface
        base = base.strip()
        if base: toks.append(base)
    return list(dict.fromkeys(toks))

# =========================
# 推荐（中文 / 英文 / 日文）
# =========================
DEFAULT_TOP_K = 5
DEFAULT_W_SEM = 0.3 # 默认语义权重

# --- recommend_cn (带 BERT) ---
def recommend_cn(text: str, top_k=DEFAULT_TOP_K, use_semantic=False, w_sem=DEFAULT_W_SEM):
    s = (text or "").strip()
    if not s: return [], []
    try:
        with sqlite3.connect(db_path_cn) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT word_id, word FROM word
                WHERE instr(LOWER(?), LOWER(word)) > 0
            """, (s,))
            hits = cur.fetchall()
            if not hits: return [], []
            scores = {}
            for wid, _ in hits:
                cur.execute("""
                    SELECT e.emoji, we.co_count
                    FROM word_emoji we
                    JOIN emoji e ON e.emoji_id = we.emoji_id
                    WHERE we.word_id = ?
                """, (wid,))
                for key, co in cur.fetchall():
                    scores[key] = scores.get(key, 0) + int(co)
    except Exception as e:
        st.error(f"中文数据库查询失败: {e}")
        return [], []
    
    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(s, tok, mdl)
            if qv is not None:
                qv = qv[0]
                rescored = []
                for title, co in scores.items():
                    sem = 0.0
                    if title in _CN_EMOJI_EMB:
                        sem = _cos(qv, _CN_EMOJI_EMB[title])
                    else:
                        for cand in _title_variants(title):
                            if cand in _CN_EMOJI_EMB:
                                sem = _cos(qv, _CN_EMOJI_EMB[cand]); break
                    if sem == 0.0:
                        dv = _embed_texts(title, tok, mdl) 
                        if dv is not None: sem = _cos(qv, dv[0])
                    
                    final = (1.0 - w_sem) * _log1p(co) + w_sem * sem
                    rescored.append((title, final))
                
                # --- 最终修复：使用与 fallback 一致的 tie-breaker ---
                rescored.sort(key=lambda x: (-x[1], x[0]))
                return [w for _, w in hits], [k for k, _ in rescored[:top_k]]

    # "共起"行 (use_semantic=False)
    top = sorted(scores.items(), key=lambda x: (-_log1p(x[1]), x[0]))[:top_k]
    return [w for _, w in hits], [k for k, _ in top]

# --- recommend_en (带 BERT) ---
def recommend_en(text: str, top_k=DEFAULT_TOP_K, use_semantic=False, w_sem=DEFAULT_W_SEM):
    toks = _words_en(text)
    if not toks: return [], []
    try:
        with sqlite3.connect(db_path_en) as conn:
            cur = conn.cursor()
            q = ",".join("?" for _ in toks)
            cur.execute(f"SELECT word_id, word FROM word WHERE word IN ({q})", toks)
            hits = cur.fetchall()
            if not hits: return [], []
            scores = {}
            for wid, _ in hits:
                cur.execute("""
                    SELECT e.emoji, we.co_count
                    FROM word_emoji we
                    JOIN emoji e ON e.emoji_id = we.emoji_id
                    WHERE we.word_id = ?
                """, (wid,))
                for key, co in cur.fetchall():
                    scores[key] = scores.get(key, 0) + int(co)
    except Exception as e:
        st.error(f"英文数据库查询失败: {e}")
        return [], []

    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(text, tok, mdl)
            if qv is not None:
                qv = qv[0]
                items = list(scores.items())
                names = [_desc_for_token("英語", emo_char) for emo_char, _ in items]
                vecs = _embed_texts(names, tok, mdl)
                if vecs is not None:
                    rescored = []
                    for (emo_char, co), v in zip(items, vecs):
                        sem = _cos(qv, v)
                        final = (1.0 - w_sem) * _log1p(co) + w_sem * sem
                        rescored.append((emo_char, final))
                    
                    # --- 最终修复：使用与 fallback 一致的 tie-breaker ---
                    rescored.sort(key=lambda x: (-x[1], x[0]))
                    return [w for _, w in hits], [k for k, _ in rescored[:top_k]]
    
    # "共起"行 (use_semantic=False)
    top = sorted(scores.items(), key=lambda x: (-_log1p(x[1]), x[0]))[:top_k]
    return [w for _, w in hits], [k for k, _ in top]

# --- recommend_ja (带 BERT) ---
def recommend_ja(text: str, top_k=DEFAULT_TOP_K, use_semantic=False, w_sem=DEFAULT_W_SEM):
    toks = _tokenize_ja(text)
    if not toks: return [], []
    try:
        with sqlite3.connect(db_path_ja) as conn:
            cur = conn.cursor()
            q = ",".join("?" for _ in toks)
            cur.execute(f"SELECT word_id, word FROM word WHERE word IN ({q})", toks)
            hits = cur.fetchall()
            if not hits: return [], []
            scores = {}
            for wid, _ in hits:
                cur.execute("""
                    SELECT e.emoji, we.co_count
                    FROM word_emoji we
                    JOIN emoji e ON e.emoji_id = we.emoji_id
                    WHERE we.word_id = ?
                """, (wid,))
                for key, co in cur.fetchall():
                    scores[key] = scores.get(key, 0) + int(co)
    except Exception as e:
        st.error(f"日文数据库查询失败: {e}")
        return [], []

    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(text, tok, mdl)
            if qv is not None:
                qv = qv[0]
                items = list(scores.items())
                names = [_desc_for_token("日本語", emo_char) for emo_char, _ in items]
                vecs = _embed_texts(names, tok, mdl)
                if vecs is not None:
                    rescored = []
                    for (emo_char, co), v in zip(items, vecs):
                        sem = _cos(qv, v)
                        final = (1.0 - w_sem) * _log1p(co) + w_sem * sem
                        rescored.append((emo_char, final))

                    # --- 最终修复：使用与 fallback 一致的 tie-breaker ---
                    rescored.sort(key=lambda x: (-x[1], x[0]))
                    return [w for _, w in hits], [k for k, _ in rescored[:top_k]]
    
    # "共起"行 (use_semantic=False)
    top = sorted(scores.items(), key=lambda x: (-_log1p(x[1]), x[0]))[:top_k]
    return [w for _, w in hits], [k for k, _ in top]

def build_clickables(lang: str, rec_list):
    files, labels = [], []
    if lang == "中国語":
        for t in rec_list:
            src = None
            for cand in _title_variants(t):
                if cand in TITLE2ABS: src = TITLE2ABS[cand]; break
            if src and (src.startswith("http") or os.path.exists(src)):
                files.append(src); labels.append(t)
    else:
        labels = rec_list
    return files[:5], labels[:5]

# =========================
# 雷达图（问卷）读取
# =========================
RADAR_XLSX_PATH = Path("日本人数据.xlsx") 
RADAR_DIMS = ["怒り", "嫌悪", "恐怖", "喜び", "悲しみ", "驚き"]

NORMALIZE_MODE = "per_emoji"
RADAR_SCALE = "log1p"
RADAR_MIN_VISIBLE = 0.06
RADAR_BASELINE = 0.18

@st.cache_data(show_spinner="正在读取 Excel...")
def _read_excel_bytesio(fp):
    try:
        return pd.read_excel(fp, engine="openpyxl", header=0)
    except Exception:
        try:
            return pd.read_excel(fp, header=0)
        except Exception:
            raise

def load_radar_df_with_fallback(path: Path):
    if path.exists():
        try:
            df = pd.read_excel(path, engine="openpyxl", header=0)
        except Exception:
            try:
                df = pd.read_excel(path, header=0)
            except Exception as e:
                st.warning(f"读取本地 Excel 时出错 ({path})：{e}")
                return None
        df.columns = [str(c).strip() for c in df.columns]
        first_col_name = df.columns[0]
        first_vals = df.iloc[:, 0].dropna().astype(str)
        looks_like_emoji_col = False
        if not first_vals.empty:
            sample = first_vals.iloc[0]
            if any(ord(ch) > 1000 for ch in sample):
                looks_like_emoji_col = True
        if (first_col_name == "" or first_col_name.lower() in ("unnamed: 0", "nan")) and looks_like_emoji_col:
            df = df.rename(columns={df.columns[0]: "emoji"})
        if df.columns[0] != "emoji":
            col0_name = df.columns[0]
            if isinstance(col0_name, str) and any(ord(ch) > 1000 for ch in col0_name):
                df = df.rename(columns={df.columns[0]: "emoji"})
        df = df.dropna(how="all")
        if "emoji" in df.columns:
            df["emoji"] = df["emoji"].astype(str).str.strip()
        return df
    else:
        return None

if "st_radar_df" not in st.session_state:
    st.session_state["st_radar_df"] = load_radar_df_with_fallback(RADAR_XLSX_PATH)

_RADAR_DF = st.session_state["st_radar_df"]

if _RADAR_DF is None:
    st.warning(f"未找到本地问卷文件：{RADAR_XLSX_PATH}。请把 Excel 文件放到该路径或使用下方上传。")
    uploaded = st.file_uploader("或者把问卷 Excel（.xlsx）拖到这里上传", type=["xlsx", "xls"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df_uploaded = _read_excel_bytesio(uploaded)
            df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]
            first_col_name = df_uploaded.columns[0]
            first_vals = df_uploaded.iloc[:, 0].dropna().astype(str)
            looks_like_emoji_col = False
            if not first_vals.empty:
                sample = first_vals.iloc[0]
                if any(ord(ch) > 1000 for ch in sample):
                    looks_like_emoji_col = True
            if (first_col_name == "" or first_col_name.lower() in ("unnamed: 0", "nan")) and looks_like_emoji_col:
                df_uploaded = df_uploaded.rename(columns={df_uploaded.columns[0]: "emoji"})
            if df_uploaded.columns[0] != "emoji":
                col0_name = df_uploaded.columns[0]
                if isinstance(col0_name, str) and any(ord(ch) > 1000 for col0_name in col0_name):
                    df_uploaded = df_uploaded.rename(columns={df_uploaded.columns[0]: "emoji"})
            df_uploaded = df_uploaded.dropna(how="all")
            if "emoji" in df_uploaded.columns:
                df_uploaded["emoji"] = df_uploaded["emoji"].astype(str).str.strip()
            
            st.session_state["st_radar_df"] = df_uploaded
            _RADAR_DF = df_uploaded
            st.success("已加载上传的问卷数据。")
            st.rerun()
        except Exception as e:
            st.error(f"读取上传文件失败：{e}")
            _RADAR_DF = None

# =========================
# 雷达图绘制：CJK 字体与标签
# =========================
@st.cache_resource
def _find_cjk_font():
    candidates = [
        "Noto Sans CJK JP", "NotoSansCJKJP", "Source Han Sans JP", "Source Han Sans",
        "Arial Unicode MS", "Yu Gothic", "MS Gothic", "Hiragino Kaku Gothic ProN",
        "AppleGothic", "SimHei", "Microsoft YaHei", "Meiryo"
    ]
    for name in candidates:
        try:
            fp = fm.FontProperties(family=name)
            path = fm.findfont(fp, fallback_to_default=False)
            if path and "DejaVu" not in path:
                return fm.FontProperties(fname=path)
        except Exception:
            pass
    try:
        path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
        return fm.FontProperties(fname=path)
    except Exception:
        return None

def _apply_scale(vals):
    if RADAR_SCALE == "log1p":
        return [float(np.log1p(max(0.0, v))) for v in vals]
    elif RADAR_SCALE == "sqrt":
        return [float(np.sqrt(max(0.0, v))) for v in vals]
    else:
        return [float(v) for v in vals]

def plot_radar(values, labels):
    N = len(labels)
    if N == 0:
        return None
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals = values + [values[0]]
    angles_closed = angles + [angles[0]]
    if NORMALIZE_MODE is None:
        maxv = max(max(values) if values else 1.0, 1.0)
    else:
        maxv = 1.0
    fp = _find_cjk_font()
    fig = plt.figure(figsize=(3.8, 3.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(22)
    ax.set_ylim(0, maxv * 1.05)
    try:
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"])
    except Exception: pass
    ax.plot(angles_closed, vals, linewidth=1)
    ax.fill(angles_closed, vals, alpha=0.2)
    try:
        ax.tick_params(axis='y', labelsize=8)
    except Exception: pass
    label_pad = maxv * 1.12
    for angle, label in zip(angles, labels):
        x = angle
        y = label_pad
        if fp is not None:
            ax.text(x, y, label, size=10, ha='center', va='center',
                    fontproperties=fp, rotation=0, rotation_mode='anchor')
        else:
            ax.text(x, y, label, size=10, ha='center', va='center', rotation=0)
    ax.set_xticks([])
    plt.tight_layout(pad=1.0)
    return fig

# =========================
# 取雷达值
# =========================
def _get_radar_values_for_emoji(df: pd.DataFrame, emoji_key: str):
    if df is None:
        return None
    emoji_key = str(emoji_key).strip()
    row = None
    if "emoji" in df.columns:
        mask = df["emoji"].astype(str).str.strip() == emoji_key
        if mask.any():
            row = df[mask].iloc[0]
    if row is None:
        for col in df.columns:
            try:
                mask = df[col].astype(str).str.strip() == emoji_key
                if mask.any():
                    row = df[mask].iloc[0]
                    break
            except Exception:
                continue
    if row is None:
        return None
    vals = []
    for d in RADAR_DIMS:
        if d not in df.columns:
            st.error(f"问卷中缺少维度: {d}")
            return None
        raw = row[d]
        try:
            s = str(raw).strip()
            s = s.replace("％", "").replace("%", "")
            s = s.replace("，", ",")
            if "," in s and "." not in s:
                s = s.replace(",", ".")
            v = pd.to_numeric(s, errors="coerce")
            if pd.isna(v): v = 0.0
        except Exception:
            v = 0.0
        vals.append(float(v))
    vals = _apply_scale(vals)
    if NORMALIZE_MODE == "per_emoji":
        maxv = max(vals) if vals else 1.0
        if maxv > 0:
            vals = [v / maxv for v in vals]
    elif NORMALIZE_MODE == "global":
        try:
            _raw = df[RADAR_DIMS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            _scaled = _raw.apply(lambda col: pd.Series(_apply_scale(col.tolist())), axis=0)
            global_max = float(_scaled.max().max())
            if global_max > 0:
                vals = [v / global_max for v in vals]
        except Exception: pass
    if 0 < float(RADAR_MIN_VISIBLE) < 1:
        vals = [max(v, float(RADAR_MIN_VISIBLE)) for v in vals]
    base = float(RADAR_BASELINE)
    if 0 < base < 1:
        vals = [base + (1.0 - base) * v for v in vals]
    return vals

# =========================
# UI：侧边栏 (带 w_sem 滑块)
# =========================
with st.sidebar:
    st.header("⚙️ 设置 / Settings")
    lang = st.radio("Language / 言語", ["中国語", "英語", "日本語"], index=0, horizontal=False)
    
    w_sem_slider = st.slider(
        "BERT語義重量", 
        min_value=0.0, 
        max_value=1.0, 
        value=DEFAULT_W_SEM, # 默认值 0.3
        step=0.05,
        help="""
        「BERT 语義」行における BERT 语義の重みを調整する：
        - 0.0: 共起のみ
        - 0.5: 共起と语义が半分ずつ
        - 1.0: 语义のみ
- **推薦（共起）**  
  共起による推薦は，私たちが独自に収集したデータを基に行い，自作データベースとの照合によって絵文字を推薦している。

- **推薦（BERTモデル）**  
  上の「共起」による頻度だけでなく，BERT という言語モデルで計算した。
  文章の意味と絵文字の意味の近さ（類似度）も加味して並び替えた結果です。  

- **BERT語義重量**  
  共起による頻度と意味の近さをどれくらいの割合で混ぜるかを調整するための重みです。

        """
    )

# =========================
# UI：主区域
# =========================
st.title("Emoji推薦")
placeholder = {"中国語":"输入句子（中文）", "英語":"Type an English sentence", "日本語":"日本語の文を入力"}

# ==== 按照你“能追加的版本”修改的部分（只改这里） ====
# 使用普通 text_input + session_state，保证按钮追加生效
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

txt = st.text_input(
    "输入句子 / Sentence / 入力文",
    placeholder=placeholder[lang],
    key="input_text_field"
)

# 手动推荐按钮：触发 rerun（你也可以不点，输入时回车也会触发）
st.button("推薦") 

st.markdown("---")
# ==== 修改结束，其它逻辑保持不变 ====

# 3. 实时计算逻辑 (始终为三语计算 BERT，并传入 w_sem_slider)
files_co, labels_co = [], []
files_bert, labels_bert = [], []
current_text = txt.strip() if txt else ""
info_msg_co = ""
info_msg_bert = ""

if not current_text:
    info_msg_co = ""
    info_msg_bert = ""
else:
    if lang == "中国語":
        _, rec_co = recommend_cn(current_text, top_k=DEFAULT_TOP_K, use_semantic=False)
        files_co, labels_co = build_clickables("中国語", rec_co)
        
        _, rec_bert = recommend_cn(current_text, top_k=DEFAULT_TOP_K, use_semantic=True, w_sem=w_sem_slider)
        files_bert, labels_bert = build_clickables("中国語", rec_bert)
    
    elif lang == "英語":
        _, rec_co = recommend_en(current_text, top_k=DEFAULT_TOP_K, use_semantic=False)
        _f, labels_co = build_clickables("英語", rec_co)
        
        _, rec_bert = recommend_en(current_text, top_k=DEFAULT_TOP_K, use_semantic=True, w_sem=w_sem_slider)
        _f, labels_bert = build_clickables("英語", rec_bert)
    
    else: # 日语
        _, rec_co = recommend_ja(current_text, top_k=DEFAULT_TOP_K, use_semantic=False)
        _, rec_bert = recommend_ja(current_text, top_k=DEFAULT_TOP_K, use_semantic=True, w_sem=w_sem_slider)
        
        _f, labels_co = build_clickables("日本語", rec_co)
        _f, labels_bert = build_clickables("日本語", rec_bert)

    if not labels_co: info_msg_co = "共起：无匹配推荐"
    if not labels_bert: info_msg_bert = "BERT：无匹配推荐"

# 4. 插入回调
def _append_to_input(token: str):
    base = st.session_state.get("input_text_field", "").rstrip()
    newv = (base + (" " if base else "") + token) if token else base
    st.session_state["input_text_field"] = newv

# --- 5. UI 渲染 (统一逻辑) ---

# (此辅助函数仅在日语模式下需要)
if lang == "日本語":
    def try_match_label_to_row(df, label):
        if df is None: return None, "no_df"
        lab = str(label).strip()
        if "emoji" in df.columns:
            mask = df["emoji"].astype(str).str.strip() == lab
            if mask.any(): return df[mask].iloc[0], "exact_emoji_column"
        for col in df.columns:
            try:
                mask = df[col].astype(str).str.strip() == lab
                if mask.any(): return df[mask].iloc[0], f"exact_in_col:{col}"
            except Exception: continue
        try:
            dem = _name_from_emoji_char(lab)
            dem = str(dem).strip()
            if dem:
                for col in df.columns:
                    try:
                        mask = df[col].astype(str).str.contains(dem, na=False)
                        if mask.any(): return df[mask].iloc[0], f"demojize_match:{col}"
                    except Exception: continue
        except Exception: pass
        for col in df.columns:
            try:
                mask = df[col].astype(str).str.contains(lab, na=False)
                if mask.any(): return df[mask].iloc[0], f"contains_in_col:{col}"
            except Exception: continue
        return None, "no_match"

# --- 辅助函数：用于渲染一个推荐项 (图片/按钮 + 雷达图[仅日语]) ---
def render_item(col, item_key, label, files_dict, lang):
    with col:
        # Part 1: Show Image (CN) or Button (EN/JA)
        if lang == "中国語":
            src = files_dict.get(label) # 从 (label -> file) 字典中查找
            if src:
                st.image(src, caption=None, width=THUMB_H)
            st.button(f"「{label}」", key=item_key, on_click=_append_to_input, args=(label,))
        else:
            # EN or JA: Show text button
            st.button(label, key=item_key, on_click=_append_to_input, args=(label,))
        
        # Part 2: Show Radar Plot (JA only)
        if lang == "日本語":
            row, method = try_match_label_to_row(_RADAR_DF, label)
            vals = _get_radar_values_for_emoji(_RADAR_DF, label) if row is not None else None
            if vals is not None:
                fig = plot_radar(vals, RADAR_DIMS); st.pyplot(fig)
            else:
                st.info("问卷中未找到")
                if _RADAR_DF is not None and "emoji" in _RADAR_DF.columns:
                    try:
                        options = _RADAR_DF["emoji"].astype(str).tolist()
                        sel = st.selectbox("手动选择", options, key=f"{item_key}_sel", index=None)
                        if sel:
                            vals_sel = _get_radar_values_for_emoji(_RADAR_DF, sel)
                            if vals_sel: fig_sel = plot_radar(vals_sel, RADAR_DIMS); st.pyplot(fig_sel)
                    except Exception: pass

# --- 渲染“共起”行 (所有语言) ---
# --- 渲染“共起”行 (所有语言) ---
st.markdown("**推薦（共起）**")

# 日语模式下：折叠显示雷达图说明（点击展开）
if lang == "日本語":
    with st.expander("レーダーチャートの見方", expanded=False):
        st.markdown(
            """
- このレーダーチャートは，絵文字に対して回答者が感じた 6つの基本感情
  （怒り・嫌悪・恐怖・喜び・悲しみ・驚き）の強さを表しています．  
- 線が外側に近いほど，その絵文字に対して その感情が強く／よく選ばれている ことを意味します．   
- 本システムでは，日本人大学生を対象としたアンケート調査の結果をもとに，
  正規化した値をプロットしています．
            """
        )

if labels_co:
    # 为中文创建 label -> file 映射
    files_dict_co = {label: file for label, file in zip(labels_co, files_co)} if lang == "中国語" else {}
    cols_co = st.columns(len(labels_co))
    for i, lab in enumerate(labels_co):
        render_item(cols_co[i], f"ins_co_{i}", lab, files_dict_co, lang)
else:
    st.info(info_msg_co)

# --- 渲染“BERT”行 (所有语言) ---
st.markdown("**推薦（BERTモデル）**")
if labels_bert:
    # 为中文创建 label -> file 映射
    files_dict_bert = {label: file for label, file in zip(labels_bert, files_bert)} if lang == "中国語" else {}
    cols_bert = st.columns(len(labels_bert))
    for i, lab in enumerate(labels_bert):
        render_item(cols_bert[i], f"ins_bert_{i}", lab, files_dict_bert, lang)
else:
    st.info(info_msg_bert)

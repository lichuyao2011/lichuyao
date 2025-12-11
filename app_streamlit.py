# app_streamlit.py —— 三语共现 + BERT 语义重排（Streamlit 版）
import os, re, json, sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Emoji推薦", page_icon="", layout="wide")
# —— 隐藏“Press Enter to apply”提示（不影响功能，只去掉提示文本）——
def _kill_press_to_apply():
    st.markdown("""
    <style>
      /* 新旧版本通吃：把键盘提示 / Enter 提示统统隐藏 */
      [data-testid="stPressEnterToApply"],
      .st-keyboard-hint,
      div[aria-live="polite"] kbd[title="Enter"],
      div:has(> kbd[title="Enter"]) {
        display: none !important;
      }
    </style>
    """, unsafe_allow_html=True)

_kill_press_to_apply()

# =========================
# 工具与依赖
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

_HAVE_TORCH = _ensure("torch")
_HAVE_TRANS = _ensure("transformers")
_HAVE_EMOJI = _ensure("emoji")

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

if _HAVE_EMOJI:
    import emoji as _emoji
else:
    _emoji = None

# =========================
# 读取 emoji 映射（兼容 dict/list 两种结构）
# =========================
def _load_title2abs(json_path: Path, base_dir: Path):
    if not json_path.exists():
        st.warning(f"emojis.json 未找到: {json_path}")
        return {}
    data = json.loads(json_path.read_text("utf-8"))

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

TITLE2ABS = _load_title2abs(EMOJI_JSON, EMOJI_BASE_DIR)

def _title_variants(t: str):
    t = (t or "").strip()
    if not t: return []
    core = t.strip("_")
    return [t, f"_{t}_", f"_{t}", f"{t}_", core, f"_{core}_", f"_{core}", f"{core}_"]

# =========================
# BERT 语义向量（多语）
# =========================
BERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=True)
def _bert_init():
    if not (_HAVE_TORCH and _HAVE_TRANS):
        return None, None
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    mdl = AutoModel.from_pretrained(BERT_MODEL_NAME)
    mdl.eval()
    return tok, mdl

def _bert_ready(tok, mdl):
    return tok is not None and mdl is not None and torch is not None

def _embed_texts(texts, tok, mdl):
    if not _bert_ready(tok, mdl): return None
    if isinstance(texts, str): texts = [texts]
    batch = tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**batch)
        last = out.last_hidden_state  # [B, T, H]
        mask = batch["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (last * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        vec = (summed / count).cpu().numpy()
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec

_CN_EMOJI_EMB = {}

@st.cache_resource(show_spinner=False)
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
def recommend_cn(text: str, top_k=5, use_semantic=False, w_sem=0.3):
    s = (text or "").strip()
    if not s: return [], []
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

    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(s, tok, mdl)
            if qv is not None:
                qv = qv[0]
                rescored = []
                for title, co in scores.items():
                    if title in _CN_EMOJI_EMB:
                        sem = _cos(qv, _CN_EMOJI_EMB[title])
                    else:
                        sem = 0.0
                        for cand in _title_variants(title):
                            if cand in _CN_EMOJI_EMB:
                                sem = _cos(qv, _CN_EMOJI_EMB[cand]); break
                        if sem == 0.0:
                            dv = _embed_texts(title, tok, mdl)
                            if dv is not None: sem = _cos(qv, dv[0])
                    final = (1.0 - w_sem) * _log1p(scores[title]) + w_sem * sem
                    rescored.append((title, final))
                rescored.sort(key=lambda x: x[1], reverse=True)
                return [w for _, w in hits], [k for k, _ in rescored[:top_k]]

    top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [w for _, w in hits], [k for k, _ in top]

def recommend_en(text: str, top_k=5, use_semantic=False, w_sem=0.3):
    toks = _words_en(text)
    if not toks: return [], []
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

    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(text, tok, mdl)
            if qv is not None:
                qv = qv[0]
                items = list(scores.items())
                names = [_desc_for_token("英語", emo_char) for emo_char, _ in items]
                vecs = _embed_texts(names, tok, mdl)
                rescored = []
                for (emo_char, co), v in zip(items, vecs):
                    sem = _cos(qv, v)
                    final = (1.0 - w_sem) * _log1p(co) + w_sem * sem
                    rescored.append((emo_char, final))
                rescored.sort(key=lambda x: x[1], reverse=True)
                return [w for _, w in hits], [k for k, _ in rescored[:top_k]]

    top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [w for _, w in hits], [k for k, _ in top]

def recommend_ja(text: str, top_k=5, use_semantic=False, w_sem=0.3):
    toks = _tokenize_ja(text)
    if not toks: return [], []
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

    if use_semantic:
        tok, mdl = _bert_init()
        if _bert_ready(tok, mdl):
            qv = _embed_texts(text, tok, mdl)
            if qv is not None:
                qv = qv[0]
                items = list(scores.items())
                names = [_desc_for_token("日本語", emo_char) for emo_char, _ in items]
                vecs = _embed_texts(names, tok, mdl)
                rescored = []
                for (emo_char, co), v in zip(items, vecs):
                    sem = _cos(qv, v)
                    final = (1.0 - w_sem) * _log1p(co) + w_sem * sem
                    rescored.append((emo_char, final))
                rescored.sort(key=lambda x: x[1], reverse=True)
                return [w for _, w in hits], [k for k, _ in rescored[:top_k]]

    top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
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
# UI：侧边栏
# =========================
with st.sidebar:
    st.header("⚙️ 设置 / Settings")
    lang = st.radio("Language / 言語", ["中国語", "英語", "日本語"], index=0, horizontal=False)
    use_sem = st.checkbox("开启语义重排 (BERT)", value=True)
    w_sem = st.slider("语义权重 w_sem", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    top_k = 5
    

# =========================
# UI：主区域
# =========================
st.title("Emoji推薦")
placeholder = {"中国語":"输入句子（中文）", "英語":"Type an English sentence", "日本語":"日本語の文を入力"}
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
txt = st.text_input("输入句子 / Sentence / 入力文",
                    placeholder=placeholder[lang],
                    key="input_text_field")


# 推荐
def do_recommend(text, lang, use_sem, w_sem):
    if not (text or "").strip():
        hits_text = {"中国語":"（无）", "英語":"（None）", "日本語":"（なし）"}[lang]
        return hits_text, [], []
    if lang == "中国語":
        hit_words, rec = recommend_cn(text, top_k=top_k, use_semantic=use_sem, w_sem=w_sem)
    elif lang == "英語":
        hit_words, rec = recommend_en(text, top_k=top_k, use_semantic=use_sem, w_sem=w_sem)
    else:
        hit_words, rec = recommend_ja(text, top_k=top_k, use_semantic=use_sem, w_sem=w_sem)

    hits_text = ("、".join(hit_words) if lang != "英語" else ", ".join(hit_words)) or \
                {"中国語":"（无）", "英語":"（None）", "日本語":"（なし）"}[lang]
    files, labels = build_clickables(lang, rec)
    return hits_text, files, labels

hits_text, files, labels = do_recommend(st.session_state.get("input_text_field",""), lang, use_sem, w_sem)

st.markdown("**一致語 / Hit words**")
st.write(hits_text)

# 展示与点击插入
def _append_to_input(token: str):
    base = st.session_state.get("input_text_field", "").rstrip()
    newv = (base + (" " if base else "") + token) if token else base
    st.session_state["input_text_field"] = newv

st.markdown("---")

if lang == "中国語":
    # 图片九宫格（每个图片下方放一个“插入”按钮）
    if files:
        n = len(files)
        cols = st.columns(min(5, n))
        for i, (src, title) in enumerate(zip(files, labels)):
            with cols[i % len(cols)]:
                st.image(src, caption=None, width=None, use_container_width=False, output_format="auto", clamp=False)

                st.button(f"「{title}」", key=f"ins_cn_{i}", on_click=_append_to_input, args=(title,))
    else:
        st.info("暂无推荐图片")
else:
    # 英/日：一行按钮（显示 Unicode 表情）
    if labels:
        st.markdown("**点击按钮插入表情**")
        cols = st.columns(len(labels))
        for i, lab in enumerate(labels):
            with cols[i]:
                st.button(lab, key=f"ins_enja_{i}", on_click=_append_to_input, args=(lab,))
    else:
        st.info("暂无推荐")


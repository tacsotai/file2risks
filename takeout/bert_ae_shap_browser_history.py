# file: bert_ae_shap_browser_history.py
# -*- coding: utf-8 -*-
import os, json, re, random
from typing import List, Optional
from urllib.parse import urlparse, parse_qs, unquote_plus

import numpy as np
import pandas as pd
import tldextract

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap
# import matplotlib.pyplot as plt

# ============== 設定 ==============
TEXT_COLS = ["title", "url"]
TIMESTAMP_COL = "visited_at"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
HIDDEN = 256
EMBED_POOL = "mean"       # 'cls' or 'mean'
THRESHOLD_Q = 0.98
SEED = 42
PREFERRED_LOCALE = None   # "ja" / "en" で上書き可
DEFAULT_LOCALE = "en"
LOCALE_SAMPLE_N = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------- 言語推定ユーティリティ -------------
_JA_HIRA = r"\u3040-\u309F"; _JA_KATA = r"\u30A0-\u30FF"; _HAN = r"\u4E00-\u9FFF"; _KO_HANGUL = r"\uAC00-\uD7AF"
_TLD_TO_LANG = {".co.jp":"ja",".jp":"ja",".kr":"ko",".cn":"zh",".tw":"zh"}

def _as_str_list(x):
    """SHAPや各処理から来る入力を list[str] に正規化する"""
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, (np.ndarray, pd.Series)):
        return [str(v) for v in x.tolist()]
    return [str(x)]

def _score_text_lang(s: str) -> dict:
    s = s or ""
    scores = {"ja":0,"ko":0,"zh":0,"en":0}
    if re.search(f"[{_JA_HIRA}]", s): scores["ja"] += 3
    if re.search(f"[{_JA_KATA}]", s): scores["ja"] += 2
    if re.search(f"[{_HAN}]", s):
        if scores["ja"] == 0: scores["zh"] += 1
        else: scores["ja"] += 1
    if re.search(f"[{_KO_HANGUL}]", s): scores["ko"] += 3
    if re.search(r"[A-Za-z]", s): scores["en"] += 1
    return scores

def _lang_from_url_params(url: str) -> str:
    try:
        u = urlparse(url); q = parse_qs(u.query or "")
        for key in ("hl","lr","lang","locale"):
            if key in q and q[key]:
                v = (q[key][0] or "").lower()
                if "ja" in v: return "ja"
                if "en" in v: return "en"
                if "ko" in v: return "ko"
                if "zh" in v or "cn" in v: return "zh"
    except Exception:
        pass
    return ""

def _lang_from_tld(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
        for suf, lang in _TLD_TO_LANG.items():
            if host.endswith(suf): return lang
    except Exception:
        pass
    return ""

def decide_output_locale(df: pd.DataFrame) -> str:
    if PREFERRED_LOCALE:
        return PREFERRED_LOCALE
    votes = {"ja":0,"en":0,"ko":0,"zh":0}
    urls = df["url"].astype(str).tolist()[:LOCALE_SAMPLE_N]
    for u in urls:
        v = _lang_from_url_params(u)
        if v: votes[v] += 2
    titles = df["title"].astype(str).tolist()[:LOCALE_SAMPLE_N]
    for t in titles:
        s = _score_text_lang(t)
        for k,v in s.items(): votes[k] += v
    if max(votes.values()) == 0:
        for u in urls:
            v = _lang_from_tld(u)
            if v: votes[v] += 1
    lang = max(votes, key=lambda k: votes[k]) if any(votes.values()) else DEFAULT_LOCALE
    return lang if lang in {"ja","en"} else DEFAULT_LOCALE

# ------------- 入出力ユーティリティ -------------
def _chrome_time_to_iso(time_usec: Optional[object]) -> str:
    try:
        t = int(time_usec)
        ts = pd.to_datetime(t * 1000, utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        return ""

def _load_history_from_csv(path: str, text_cols: list, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in text_cols:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("")
    if ts_col in df.columns: df[ts_col] = df[ts_col].astype(str)
    return df

def _load_history_from_chrome_json(path: str, text_cols: list, ts_col: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("Browser History")
        if not isinstance(items, list):
            items = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    rows = []
    for it in items:
        if not isinstance(it, dict): continue
        title = str(it.get("title","") or "")
        url   = str(it.get("url","") or "")
        visited_at = _chrome_time_to_iso(it.get("time_usec"))
        rows.append({"title": title, "url": url, ts_col: visited_at})
    df = pd.DataFrame(rows)
    for c in text_cols:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("")
    if ts_col in df.columns: df[ts_col] = df[ts_col].astype(str)
    return df

def load_history(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        # デモデータ（ファイルが無いとき）
        return pd.DataFrame({
            "title": [
                "How to use pandas merge",
                "HuggingFace Transformers tutorial",
                "News: local weather today",
                "暗号通貨 市況 ニュース",
                "Adult casino bonus offer",
            ],
            "url": [
                "https://pandas.pydata.org/docs/user_guide/merging.html",
                "https://huggingface.co/transformers/",
                "https://news.example.com/weather/today",
                "https://jp-crypto.example.com/news",
                "http://shady-casino.xxx/bonus",
            ],
            TIMESTAMP_COL: [
                "2025-08-10T10:01:02Z",
                "2025-08-10T11:05:00Z",
                "2025-08-10T12:00:00Z",
                "2025-08-12T08:30:00Z",
                "2025-08-12T09:12:34Z",
            ]
        })
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        return _load_history_from_chrome_json(input_path, TEXT_COLS, TIMESTAMP_COL)
    if ext == ".csv":
        return _load_history_from_csv(input_path, TEXT_COLS, TIMESTAMP_COL)
    # 不明拡張子: JSON→CSVの順に試す
    try:
        return _load_history_from_chrome_json(input_path, TEXT_COLS, TIMESTAMP_COL)
    except Exception:
        try:
            return _load_history_from_csv(input_path, TEXT_COLS, TIMESTAMP_COL)
        except Exception:
            return pd.DataFrame(columns=TEXT_COLS + [TIMESTAMP_COL])

def extract_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        return ".".join([p for p in [ext.domain, ext.suffix] if p]) or url
    except Exception:
        return url

def _pick_first(params: dict, keys: list):
    for k in keys:
        if k in params and params[k]:
            v = params[k][0]
            if isinstance(v, str) and v.strip():
                return v
    return None

def _clean_query(q: str) -> str:
    q = unquote_plus(q)
    q = " ".join(q.split())
    return q[:200]

def parse_search_query(url: str, title: str):
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
        params = parse_qs(u.query or "")
        # Google
        if "google." in host:
            q = _pick_first(params, ["q"])
            tbm = _pick_first(params, ["tbm"])
            if not q and title:
                low = title.lower()
                if " - google 検索" in low: q = title[: low.rfind(" - google 検索")]
                elif " - google search" in low: q = title[: low.rfind(" - google search")]
            kind = "images" if tbm == "isch" else ("news" if tbm == "nws" else "web")
            return {"engine": "google", "query": _clean_query(q) if q else None, "kind": kind}
        # YouTube
        if "youtube.com" in host or "youtu.be" in host:
            if "/results" in path:
                q = _pick_first(params, ["search_query", "query"])
                return {"engine":"youtube","query":_clean_query(q) if q else None,"kind":"video"}
            return {"engine":"youtube","query":None,"kind":"video"}
        # Bing
        if "bing.com" in host and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine":"bing","query":_clean_query(q) if q else None,"kind":"web"}
        # Yahoo
        if ("search.yahoo" in host or "yahoo.co.jp" in host) and "/search" in path:
            q = _pick_first(params, ["p","q"])
            return {"engine":"yahoo","query":_clean_query(q) if q else None,"kind":"web"}
        # DuckDuckGo
        if "duckduckgo.com" in host:
            q = _pick_first(params, ["q"])
            return {"engine":"duckduckgo","query":_clean_query(q) if q else None,"kind":"web"}
        # Baidu
        if "baidu.com" in host:
            q = _pick_first(params, ["wd","word"])
            return {"engine":"baidu","query":_clean_query(q) if q else None,"kind":"web"}
        # Naver
        if "naver.com" in host and "/search" in path:
            q = _pick_first(params, ["query"])
            return {"engine":"naver","query":_clean_query(q) if q else None,"kind":"web"}
        # X (Twitter)
        if ("twitter.com" in host or "x.com" in host) and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine":"x","query":_clean_query(q) if q else None,"kind":"social"}
        # Reddit
        if "reddit.com" in host and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine":"reddit","query":_clean_query(q) if q else None,"kind":"forum"}
        # Amazon
        if "amazon." in host and (path.startswith("/s") or "/s?" in url):
            q = _pick_first(params, ["k"])
            return {"engine":"amazon","query":_clean_query(q) if q else None,"kind":"shopping"}
        return {"engine": None, "query": None, "kind": None}
    except Exception:
        return {"engine": None, "query": None, "kind": None}

# ------------- 説明生成 -------------
SITE_ALIASES = {
    "pandas.pydata.org": "Pandas Docs",
    "numpy.org": "NumPy",
    "scikit-learn.org": "scikit-learn",
    "huggingface.co": "Hugging Face",
    "github.com": "GitHub",
    "stackoverflow.com": "Stack Overflow",
    "youtube.com": "YouTube", "youtu.be": "YouTube",
    "nikkei.com": "日本経済新聞", "bloomberg.com": "Bloomberg", "reuters.com": "Reuters",
    "amazon.co.jp": "Amazon", "rakuten.co.jp": "楽天市場",
    "google.com/maps": "Google マップ", "maps.google.com": "Google マップ",
}

CATEGORY_RULES = [
    (r"(pandas|numpy|scikit|huggingface|docs\.python|api\s*reference)", "tech_docs", "positive"),
    (r"(news|cnn|bbc|nikkei|bloomberg|reuters|nytimes|wsj)", "news", "neutral"),
    (r"(github|gitlab|bitbucket|pull\s*request|issue|commit)", "code_hosting", "positive"),
    (r"(stackoverflow|serverfault|superuser|qiita|teratail)", "qna_forum", "positive"),
    (r"(youtube|youtu\.be|vimeo|tver|abema)", "video_streaming", "neutral"),
    (r"(maps|google\.com/maps|map\.)", "maps_travel", "positive"),
    (r"(invest|tradingview|portfolio|securities|brokerage)", "finance_invest", "neutral"),
    (r"(crypto|bitcoin|ethereum|defi|web3|nft)", "crypto", "neutral"),
    (r"(casino|gambl|bet|poker|odds)", "gambling", "caution"),
    (r"(adult|porn|xxx)", "adult", "caution"),
    (r"(shopping|cart|store|amazon|rakuten|yodobashi|biccamera)", "shopping", "neutral"),
    (r"(slack|discord|teams|zoom|meet\.google|webex)", "collaboration", "positive"),
    (r"(calendar|schedule|calendar\.google)", "productivity", "positive"),
    (r"(gmail|outlook|mail\.google|yahoo\.co\.jp/mail)", "email_webmail", "neutral"),
    (r"(sports|premier\s*league|mlb|nba|nfl|nhl|j\s*league|soccer)", "sports", "neutral"),
    (r"(ai|ml|machine\s*learning|deep\s*learning|llm|prompt)", "ai_ml", "positive"),
]

def _friendly_domain(domain: str) -> str:
    return SITE_ALIASES.get(domain.lower(), domain)

def guess_category(domain: str, title: str, top_tokens: List[str]):
    text = f"{domain} {title} {' '.join(top_tokens)}".lower()
    for pat, cat, tone in CATEGORY_RULES:
        if re.search(pat, text):
            return cat, tone
    return "other", "neutral"

def hour_bucket(ts_iso: Optional[str]) -> str:
    try:
        dt = pd.to_datetime(ts_iso, errors="coerce", utc=True).tz_convert("Asia/Tokyo")
        h = dt.hour
        if 0 <= h < 5: return "late_night"
        if 5 <= h < 9: return "early_morning"
        if 9 <= h < 18: return "daytime"
        return "evening"
    except Exception:
        return "unknown"

def clean_tokens(tokens: List[str]) -> List[str]:
    kept = []
    for t in tokens:
        t = re.sub(r"^##", "", str(t))
        if not re.search(r"[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]", t): continue
        if t in {"[CLS]","[SEP]","[PAD]"} or re.fullmatch(r"\d{1,4}", t): continue
        kept.append(t)
    return kept[:3]

def build_human_friendly_content(domain: str, title: str, top_tokens: List[str], ts_iso: Optional[str], locale: str="ja", row_url: str="") -> str:
    cat, _ = guess_category(domain, title, top_tokens)
    tokens = clean_tokens(top_tokens)
    hb = hour_bucket(ts_iso)
    name = _friendly_domain(domain)

    # 検索クエリ優先
    sq = parse_search_query(row_url, title)
    if sq.get("query"):
        engine = (sq.get("engine") or "web").capitalize()
        q = sq["query"]
        if locale == "ja":
            mapping = {
                "Google": "Google で",
                "Youtube": "YouTube で",
                "X": "X（Twitter）で",
                "Amazon": "Amazon で",
            }
            head = mapping.get(engine, f"{engine} で")
            return f"{head}「{q}」を検索しました。情報収集の一環です。"
        else:
            return f"Searched {engine} for “{q}” to gather context and evaluate sources."

    if locale == "ja":
        label = {
            "tech_docs": f"{name}の技術ドキュメントを閲覧しました。実装の参考になりそうです。",
            "news": f"{name}のニュースをチェックしました。トレンド把握に有効です。",
            "code_hosting": f"{name}でリポジトリ/イシューを確認しました。開発作業の一環です。",
            "qna_forum": f"{name}でQ&Aを参照しました。問題解決に役立ちます。",
            "video_streaming": f"{name}で動画を視聴しました。学習や情報収集の可能性があります。",
        }.get(cat, f"{name}でページを閲覧しました。")
        detail = f"（関連語: {', '.join(tokens)}）" if tokens else ""
        return f"{label}{detail}".strip()
    else:
        when = _hb_phrase_en(hb)
        when_clause = f" {when.capitalize()}." if when else ""

        base = {
            "tech_docs": f"Exploring {name}'s technical documentation to unblock implementation and validate best practices.",
            "news": f"Catching up on the latest coverage from {name} to stay on top of trends and potential impacts.",
            "code_hosting": f"Reviewing repositories, issues, or pull requests on {name} as part of active development.",
            "qna_forum": f"Searching {name} for explanations and proven fixes to troubleshoot a specific problem.",
            "video_streaming": f"Watching a video on {name} for learning and research.",
            "maps_travel": f"Checking routes and places on {name} to plan a visit or logistics.",
            "finance_invest": f"Monitoring markets on {name} to gauge price action, news, and risk.",
            "crypto": f"Scanning crypto-related updates on {name} to assess sentiment and security concerns.",
            "gambling": f"Visited a gambling-related page on {name}; treat with caution.",
            "adult": f"Visited adult content on {name}; flagged for caution.",
            "shopping": f"Browsing products on {name} to compare options, specs, and prices.",
            "collaboration": f"Using {name} to coordinate work and keep conversations moving.",
            "productivity": f"Reviewing schedules and tasks on {name} to organize the day.",
            "email_webmail": f"Reading or composing email on {name} to follow up and respond.",
            "sports": f"Following sports updates on {name} for scores and storylines.",
            "ai_ml": f"Researching AI/ML topics on {name} to inform experiments and designs.",
        }.get(cat, f"Visiting {name} for general browsing.")

        detail = _detail_sentence_en(tokens)
        # 2文目以降に「時間帯」や「キーワード」を自然に配置
        return f"{base}{when_clause}{detail}".strip()


def _to_date_str(x):
    try:
        ts = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(ts): return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None
    
def _top_token_contribs(sv, k=3, normalize=True, round_ndigits=3):
    """
    Text SHAP Explanation から上位 k トークンの寄与度を {token: weight} で返す。
    正の寄与のみを採用。normalize=True なら合計が 1 になるように正規化。
    """
    tokens = sv.data
    values = sv.values

    def _to_1d_list(x):
        if isinstance(x, list):
            if len(x) == 1 and isinstance(x[0], (list, np.ndarray)):
                return x[0]
            return x
        x = np.array(x, dtype=object)
        if x.ndim == 0:
            return [x.item()]
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        return list(x)

    toks = _to_1d_list(tokens)
    vals = _to_1d_list(values)

    pairs = []
    for t, v in zip(toks, vals):
        if t is None:
            continue
        tok = re.sub(r"^##", "", str(t).strip())
        if not tok or tok in {"[CLS]","[SEP]","[PAD]"}:
            continue
        if re.fullmatch(r"\d{1,4}", tok):  # 数字だけは除外（不要なら消してください）
            continue
        v = float(v)
        if v > 0:
            pairs.append((tok, v))

    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:k]

    if not pairs:
        return {}

    if normalize:
        s = sum(v for _, v in pairs)
        if s > 0:
            pairs = [(t, v / s) for t, v in pairs]

    return {t: round(v, round_ndigits) for t, v in pairs}


def _extract_top_tokens(sv, k=3):
    tokens = sv.data; values = sv.values
    def _to_1d_list(x):
        if isinstance(x, list):
            return x[0] if (len(x)==1 and isinstance(x[0], (list, np.ndarray))) else x
        x = np.array(x, dtype=object)
        if x.ndim == 0: return [x.item()]
        if x.ndim == 2 and x.shape[0] == 1: x = x[0]
        return list(x)
    tokens = _to_1d_list(tokens); values = _to_1d_list(values)
    pairs = []
    for t, v in zip(tokens, values):
        if t is None: continue
        tok = re.sub(r"^##", "", str(t).strip())
        if not tok or tok in {"[CLS]","[SEP]","[PAD]"}: continue
        pairs.append((tok, float(v)))
    pos = [(t, v) for t, v in pairs if v > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    seen, top = set(), []
    for t, _ in pos:
        if t not in seen:
            top.append(t); seen.add(t)
        if len(top) >= k: break
    return top

def _hb_phrase_en(hb: str) -> str:
    return {
        "late_night": "late at night",
        "early_morning": "early in the morning",
        "daytime": "during the day",
        "evening": "in the evening",
    }.get(hb, "")

def _detail_sentence_en(tokens: List[str]) -> str:
    return f" Key terms: {', '.join(tokens)}." if tokens else ""

def build_language_results(df, shap_values, sample_idx, threshold, locale="ja"):
    results = []
    for pos, i in enumerate(sample_idx):
        row = df.iloc[i]
        sv = shap_values[pos]

        # 追加: トップ寄与トークン -> {token: weight}
        reason_map = _top_token_contribs(sv, k=3, normalize=True)
        top_tokens = list(reason_map.keys())
        if not top_tokens:
            # フォールバック（既存の簡易版）
            top_tokens = _extract_top_tokens(sv, k=3)

        domain = extract_domain(row.get("url",""))
        date_str = _to_date_str(row.get(TIMESTAMP_COL)) if TIMESTAMP_COL in df.columns else None
        row_url = str(row.get("url",""))

        content = build_human_friendly_content(
            domain=domain,
            title=str(row.get("title","")),
            top_tokens=top_tokens,
            ts_iso=str(row.get(TIMESTAMP_COL)) if TIMESTAMP_COL in df.columns else None,
            locale=locale,
            row_url=row_url
        )
        if locale == "ja":
            title_txt = f"{domain}で潜在的なリスクを検知" if row.get("is_anomaly",0)==1 else f"{domain}で新しい発見"
        else:
            title_txt = (f"Potential risk detected on {domain}" if row.get("is_anomaly", 0) == 1 else f"New finding on {domain}")

        # ← ここで reason を追加
        results.append({
            "reason": reason_map,
            "title": title_txt,
            "content": content,
            "timing_at": date_str
        })
    return results

# ------------- 埋め込み/モデル -------------
def encode_texts(text_list, tokenizer, bert) -> np.ndarray:
    # ★ ここを追加：どんな型でも list[str] にそろえる
    text_list = _as_str_list(text_list)

    embs = []
    bert.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), BATCH_SIZE):
            batch = text_list[i:i+BATCH_SIZE]          # ← ここは必ず list[str]
            toks = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt"
            ).to(DEVICE)
            outputs = bert(**toks)
            last_hidden = outputs.last_hidden_state
            if EMBED_POOL == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                mask = toks.attention_mask.unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
            embs.append(pooled.cpu())
    return torch.cat(embs, dim=0).numpy()

class AEDataset(Dataset):
    def __init__(self, X): self.X = torch.from_numpy(X)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

class AE(nn.Module):
    def __init__(self, dim, hidden=256, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def recon_error(ae, Xnp):
    ae.eval()
    with torch.no_grad():
        Xtorch = torch.from_numpy(Xnp).to(DEVICE)
        recon = ae(Xtorch).cpu().numpy()
        return np.mean((recon - Xnp)**2, axis=1)

# ============== ここがエントリーポイント（呼び出し関数） ==============
def calc_risks(input_path: str) -> List[dict]:
    """
    履歴ファイル(JSON/CSV)を読み、BERT+AEで異常候補を推定し、
    SHAPで説明語を抽出、List[dict] を返す。
    """
    # 副次的に `anomaly_score_hist.png` と `shap_anomaly_texts.html` を生成します。

    # 1) 読み込み & ロケール決定
    df = load_history(input_path)
    output_locale = decide_output_locale(df)

    # 2) 文章化（検索クエリも反映）
    def compose_text_row(row) -> str:
        title = row.get("title",""); url = row.get("url",""); domain = extract_domain(url); ts = row.get(TIMESTAMP_COL,"")
        sq = parse_search_query(url, title)
        if sq.get("query"):
            engine = (sq.get("engine") or "web").capitalize()
            if output_locale == "ja":
                mapping = {"Google":"Google","Youtube":"YouTube","X":"X（Twitter）","Amazon":"Amazon"}
                head = mapping.get(engine, engine)
                return f"{head} で「{sq['query']}」を検索しました。"
            else:
                return f"Searched {engine} for “{sq['query']}”."
        return f"[TITLE] {title} [DOMAIN] {domain} [URL] {url} [TIME] {ts}"

    texts = df.apply(compose_text_row, axis=1).tolist()

    # 3) BERT埋め込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    X_raw = encode_texts(texts, tokenizer, bert)

    # 4) スケール
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    # 5) AE 学習
    dim = X.shape[1]
    ae = AE(dim=dim, hidden=HIDDEN, bottleneck=64).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    crit = nn.MSELoss()

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=SEED)
    train_loader = DataLoader(AEDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(AEDataset(X_val),   batch_size=BATCH_SIZE)

    best_val = float("inf")
    for _ in range(EPOCHS):
        ae.train(); tr_sum = 0.0
        for xb in train_loader:
            xb = xb.to(DEVICE); opt.zero_grad()
            loss = crit(ae(xb), xb); loss.backward(); opt.step()
            tr_sum += loss.item() * xb.size(0)
        ae.eval(); va_sum = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(DEVICE)
                va_sum += crit(ae(xb), xb).item() * xb.size(0)
        best_val = min(best_val, va_sum / max(len(val_loader.dataset),1))
    # 6) 異常スコア
    errors = recon_error(ae, X)
    thr = float(np.quantile(errors, THRESHOLD_Q))
    labels = (errors >= thr).astype(int)
    df["anomaly_score"] = errors; df["is_anomaly"] = labels

    # 7) SHAP（テキスト説明）
    # 5件ではなく一番確率が高い1件
    # sample_idx = df["anomaly_score"].nlargest(min(5, len(df))).index.tolist()
    if (df["is_anomaly"] == 1).any():
        top_idx = df.loc[df["is_anomaly"] == 1, "anomaly_score"].idxmax()
    else:
        top_idx = df["anomaly_score"].idxmax()
    sample_idx = [top_idx]  # ← 1件だけ
    # explain_texts = [texts[i] for i in sample_idx]
    explain_texts = [texts[top_idx]] # ← 1件だけ

    masker = shap.maskers.Text(tokenizer)

    def _model_for_shap(text_like):
        # ★ どんな入力でも list[str] にしてから通す
        _texts = _as_str_list(text_like)
        em = encode_texts(_texts, tokenizer, bert)
        ems = scaler.transform(em).astype(np.float32)
        return recon_error(ae, ems)

    explainer = shap.Explainer(_model_for_shap, masker)
    shap_values = explainer(explain_texts)

    # 8) 可視化の保存（副次的）
    # html_path = "shap_anomaly_texts.html"
    # with open(html_path, "w", encoding="utf-8") as f:
    #     for sv in shap_values:
    #         f.write(shap.plots.text(sv, display=False))
    # plt.figure(); plt.hist(errors, bins=30); plt.axvline(thr, linestyle="--")
    # plt.title("Anomaly score distribution (MSE)"); plt.xlabel("score"); plt.ylabel("count")
    # plt.tight_layout(); plt.savefig("anomaly_score_hist.png")

    # 9) 結果（List[dict]）
    results = build_language_results(
        df=df,
        shap_values=shap_values,
        sample_idx=sample_idx,
        threshold=thr,
        locale=output_locale
    )
    return results

# ------------- CLI 互換 -------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Browser history anomaly detector (callable).")
    p.add_argument("--input", "-i", default="history.json", help="入力パス (.json/.csv)")
    args = p.parse_args()
    try:
        res = calc_risks(args.input)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

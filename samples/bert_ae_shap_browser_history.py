# file: bert_ae_shap_browser_history.py
import os, json
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote_plus
import pandas as pd
import numpy as np
import random
import pandas as pd
import tldextract
from tqdm import tqdm
from typing import List, Optional
import json
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt

# ============== 設定 ==============
CSV_PATH = "history.json"  # あなたの履歴CSV
TEXT_COLS = ["title", "url"]  # どの列を埋め込みに使うか
TIMESTAMP_COL = "visited_at"  # 任意
MODEL_NAME = "bert-base-uncased"  # 純BERT（軽さ重視なら sentence-transformers も可）
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
HIDDEN = 256
EMBED_POOL = "mean"  # 'cls' or 'mean'
THRESHOLD_Q = 0.98  # 上位2%を異常とみなす(適宜調整)
SEED = 42
# 自動判定の上書きが必要な時は "ja" / "en" など文字列を入れる。None なら自動判定
PREFERRED_LOCALE = None
# 自動判定できなかった時のフォールバック
DEFAULT_LOCALE = "en"
# タイトル何件くらい見て判定するか（大きいほど精度↑、重さ↑）
LOCALE_SAMPLE_N = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --- 言語推定（スクリプト検出 + URLパラメータ + TLD） ---
_JA_HIRA = r"\u3040-\u309F"   # ひらがな
_JA_KATA = r"\u30A0-\u30FF"   # カタカナ
_HAN = r"\u4E00-\u9FFF"       # CJK 漢字
_KO_HANGUL = r"\uAC00-\uD7AF"

def _score_text_lang(s: str) -> dict:
    """与えられたテキストに含まれる各言語らしさを粗くスコア化"""
    import re
    s = s or ""
    scores = {"ja": 0, "ko": 0, "zh": 0, "en": 0}
    # 日本語：ひらがな・カタカナが強い証拠
    if re.search(f"[{_JA_HIRA}]", s): scores["ja"] += 3
    if re.search(f"[{_JA_KATA}]", s): scores["ja"] += 2
    # CJK漢字のみ→中国語/日本語のどちらか。ひら/カタが無ければ zh に+1
    if re.search(f"[{_HAN}]", s):
        if scores["ja"] == 0: scores["zh"] += 1
        else: scores["ja"] += 1
    # ハングル
    if re.search(f"[{_KO_HANGUL}]", s): scores["ko"] += 3
    # ラテン文字（英語らしさ）
    if re.search(r"[A-Za-z]", s): scores["en"] += 1
    return scores

_TLD_TO_LANG = {
    ".co.jp": "ja", ".jp": "ja",
    ".kr": "ko", ".cn": "zh", ".tw": "zh",
}

def _lang_from_url_params(url: str) -> str:
    """hl, lr, lang, locale などのURLクエリから推定（Google等）"""
    try:
        u = urlparse(url); q = parse_qs(u.query or "")
        for key in ("hl", "lr", "lang", "locale"):
            if key in q and len(q[key]) > 0:
                v = (q[key][0] or "").lower()
                # 例: hl=ja, lr=lang_ja, lang=en-US
                if "ja" in v: return "ja"
                if "en" in v: return "en"
                if "ko" in v: return "ko"
                if "zh" in v or "cn" in v: return "zh"
    except Exception:
        pass
    return ""

def _lang_from_tld(url: str) -> str:
    """TLDでざっくり。google.co.jp など"""
    try:
        host = (urlparse(url).netloc or "").lower()
        for suf, lang in _TLD_TO_LANG.items():
            if host.endswith(suf): return lang
    except Exception:
        pass
    return ""

def decide_output_locale(df: pd.DataFrame) -> str:
    """
    DataFrame（title/url等）から 全体の出力ロケールを推定。
    優先度: PREFERRED_LOCALE(手動) > URLの言語パラメータ > タイトルのスクリプト > TLD > DEFAULT_LOCALE
    """
    if PREFERRED_LOCALE:
        return PREFERRED_LOCALE

    # 1) URLパラメータから強い証拠を探す
    lang_votes = {"ja": 0, "en": 0, "ko": 0, "zh": 0}
    urls = df["url"].astype(str).tolist()[:LOCALE_SAMPLE_N]
    for u in urls:
        v = _lang_from_url_params(u)
        if v: lang_votes[v] += 2  # URLパラメータは強く重み付け

    # 2) タイトルのスクリプト検出（多数決）
    titles = df["title"].astype(str).tolist()[:LOCALE_SAMPLE_N]
    for t in titles:
        s = _score_text_lang(t)
        for k, v in s.items():
            lang_votes[k] += v

    # 3) パラメータ・スクリプトで決まらなければ TLD から補助
    if max(lang_votes.values()) == 0:
        for u in urls:
            v = _lang_from_tld(u)
            if v: lang_votes[v] += 1

    # 4) 票が最大の言語。全て0ならDEFAULT_LOCALE
    lang = max(lang_votes, key=lambda k: lang_votes[k]) if any(lang_votes.values()) else DEFAULT_LOCALE

    # 5) 未対応言語は英語にフォールバック（今はja/en対応、他は順次拡張）
    return lang if lang in {"ja", "en"} else DEFAULT_LOCALE


# ============== データ前処理 ==============
def _chrome_time_to_iso(time_usec: Optional[object]) -> str:
    """
    Chrome Takeout の time_usec (Unix epoch microseconds, UTC) → ISO8601文字列(UTC)
    例: 1752492402855767 -> "2025-08-13T01:26:42.855767Z"
    """
    try:
        # int/str どちらも許容
        t = int(time_usec)
        # pandas はナノ秒ベースなので *1000 で ns にする
        ts = pd.to_datetime(t * 1000, utc=True)  # microseconds -> nanoseconds
        return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        return ""

def _load_history_from_csv(path: str, text_cols: list, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    if ts_col in df.columns:
        df[ts_col] = df[ts_col].astype(str)
    return df

def _load_history_from_chrome_json(path: str, text_cols: list, ts_col: str) -> pd.DataFrame:
    """
    期待フォーマット:
    {
      "Browser History": [
        {"title": "...", "url": "...", "time_usec": 1752492402855767, ...},
        ...
      ]
    }
    ※ 上位キーがない/配列直書きの変種にもゆるく対応
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # "Browser History" キー or 最初の配列を探す
    if isinstance(data, dict):
        if "Browser History" in data and isinstance(data["Browser History"], list):
            items = data["Browser History"]
        else:
            # dict 内の最初の list を使う（安全側）
            items = None
            for v in data.values():
                if isinstance(v, list):
                    items = v; break
            if items is None:
                items = []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title", "") or "")
        url = str(it.get("url", "") or "")
        # time_usec を visited_at(ISO) に
        visited_at = _chrome_time_to_iso(it.get("time_usec"))
        rows.append({"title": title, "url": url, ts_col: visited_at})

    df = pd.DataFrame(rows)
    # 欠損補完
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    if ts_col in df.columns:
        df[ts_col] = df[ts_col].astype(str)
    return df

def load_history(input_path: str) -> pd.DataFrame:
    """
    拡張子で自動判定: .json -> Chrome Takeout / .csv -> CSV
    ファイルがない場合はデモデータを返す（従来どおり）
    """
    if not os.path.exists(input_path):
        # --- デモ用ダミーデータ（従来のまま） ---
        data = {
            "title": [
                "How to use pandas merge",
                "HuggingFace Transformers tutorial",
                "News: local weather today",
                "Football highlights premier league",
                "暗号通貨 市況 ニュース",
                "Adult casino bonus offer",
                "Free streaming movies now",
            ],
            "url": [
                "https://pandas.pydata.org/docs/user_guide/merging.html",
                "https://huggingface.co/transformers/",
                "https://news.example.com/weather/today",
                "https://sports.example.com/football/epl",
                "https://jp-crypto.example.com/news",
                "http://shady-casino.xxx/bonus",
                "http://free-streaming.zzz/watch"
            ],
            "visited_at": [
                "2025-08-10T10:01:02Z",
                "2025-08-10T11:05:00Z",
                "2025-08-10T12:00:00Z",
                "2025-08-11T21:20:00Z",
                "2025-08-12T08:30:00Z",
                "2025-08-12T09:12:34Z",
                "2025-08-12T09:50:10Z",
            ]
        }
        df = pd.DataFrame(data)
        return df

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        return _load_history_from_chrome_json(input_path, TEXT_COLS, TIMESTAMP_COL)
    elif ext == ".csv":
        return _load_history_from_csv(input_path, TEXT_COLS, TIMESTAMP_COL)
    else:
        # 拡張子不明ならJSONを試し、だめならCSV、最後にダミー
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
        domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
        return domain or url
    except Exception:
        return url
    
def _pick_first(params: dict, keys: list):
    for k in keys:
        if k in params and len(params[k]) > 0:
            v = params[k][0]
            if isinstance(v, str) and v.strip():
                return v
    return None

def _clean_query(q: str) -> str:
    # + や %xx を人間可読に、余計な空白を整理
    q = unquote_plus(q)
    q = " ".join(q.split())
    return q[:200]  # 長すぎるのはカット（任意）

def parse_search_query(url: str, title: str):
    """
    主要サイトの検索クエリを抽出する。
    戻り値: {"engine": str | None, "query": str | None, "kind": str | None}
      engine: "google" 等
      kind:   "web" | "images" | "video" | など（分かる範囲で）
    """
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
        params = parse_qs(u.query or "")

        # Google
        if "google." in host:
            q = _pick_first(params, ["q"])
            tbm = _pick_first(params, ["tbm"])  # isch=画像, nws=ニュース など
            # タイトル補完: 「 ... - Google 検索」
            if not q and title:
                # 日本語UIの例: "キーワード - Google 検索"
                if " - google 検索" in title.lower():
                    q = title[: title.lower().rfind(" - google 検索")]
                elif " - google search" in title.lower():
                    q = title[: title.lower().rfind(" - google search")]
            kind = "images" if tbm == "isch" else ("news" if tbm == "nws" else "web")
            return {"engine": "google", "query": _clean_query(q) if q else None, "kind": kind}

        # YouTube
        if "youtube.com" in host or "youtu.be" in host:
            if "/results" in path:
                q = _pick_first(params, ["search_query", "query"])
                return {"engine": "youtube", "query": _clean_query(q) if q else None, "kind": "video"}
            # /watch にはクエリが無い場合が多いのでスキップ
            return {"engine": "youtube", "query": None, "kind": "video"}

        # Bing
        if "bing.com" in host and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine": "bing", "query": _clean_query(q) if q else None, "kind": "web"}

        # Yahoo
        if ("search.yahoo" in host or "yahoo.co.jp" in host) and "/search" in path:
            q = _pick_first(params, ["p", "q"])
            return {"engine": "yahoo", "query": _clean_query(q) if q else None, "kind": "web"}

        # DuckDuckGo
        if "duckduckgo.com" in host:
            q = _pick_first(params, ["q"])
            return {"engine": "duckduckgo", "query": _clean_query(q) if q else None, "kind": "web"}

        # Baidu
        if "baidu.com" in host:
            q = _pick_first(params, ["wd", "word"])
            return {"engine": "baidu", "query": _clean_query(q) if q else None, "kind": "web"}

        # Naver
        if "naver.com" in host and "/search" in path:
            q = _pick_first(params, ["query"])
            return {"engine": "naver", "query": _clean_query(q) if q else None, "kind": "web"}

        # X (Twitter)
        if ("twitter.com" in host or "x.com" in host) and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine": "x", "query": _clean_query(q) if q else None, "kind": "social"}

        # Reddit
        if "reddit.com" in host and "/search" in path:
            q = _pick_first(params, ["q"])
            return {"engine": "reddit", "query": _clean_query(q) if q else None, "kind": "forum"}

        # Amazon 検索
        if "amazon." in host and (path.startswith("/s") or "/s?" in url):
            q = _pick_first(params, ["k"])
            return {"engine": "amazon", "query": _clean_query(q) if q else None, "kind": "shopping"}

        return {"engine": None, "query": None, "kind": None}
    except Exception:
        return {"engine": None, "query": None, "kind": None}


def compose_text(row) -> str:
    title = row.get("title","")
    url = row.get("url","")
    domain = extract_domain(url)
    ts = row.get(TIMESTAMP_COL, "")

    # --- 先頭に追加：検索クエリがあれば最優先で自然文を作る ---
    sq = parse_search_query(row_url if 'row_url' in locals() else "", title)
    # ↑ row_url を渡したいので、後述の build_language_results 変更で渡します

    if sq.get("query"):
        engine = sq.get("engine") or "web"
        q = sq["query"]
        name = _friendly_domain(domain) if '_friendly_domain' in globals() else domain
        if locale == "ja":
            # 種別ごとに言い分け（任意）
            if engine == "google":
                return f"Google で「{q}」を検索しました。必要な情報収集の一環です。"
            if engine == "youtube":
                return f"YouTube で「{q}」の動画を探しました。学習や情報収集の可能性があります。"
            if engine == "x":
                return f"X（Twitter）で「{q}」を検索しました。話題や動向の確認かもしれません。"
            if engine == "amazon":
                return f"Amazon で「{q}」を検索しました。購買検討の段階かもしれません。"
            return f"{engine.capitalize()} で「{q}」を検索しました。"
        else:
            if engine == "google":
                return f"Searched Google for “{q}”. Likely part of information gathering."
            if engine == "youtube":
                return f"Searched YouTube for “{q}”. Possibly tutorials or info."
            if engine == "x":
                return f"Searched X (Twitter) for “{q}”. Checking topics and trends."
            if engine == "amazon":
                return f"Searched Amazon for “{q}”. Possibly in consideration phase."
            return f"Searched {engine.capitalize()} for “{q}”."

    if sq.get("query"):
        # 検索クエリを含める（BERT への入力を豊かに）
        return f"[SEARCH] {sq['engine']}:{sq['query']} [TITLE] {title} [DOMAIN] {domain} [URL] {url} [TIME] {ts}"
    else:
        return f"[TITLE] {title} [DOMAIN] {domain} [URL] {url} [TIME] {ts}"


df = load_history(CSV_PATH)
texts = df.apply(compose_text, axis=1).tolist()

# ============== BERTエンコーダ ==============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert.eval()

@torch.no_grad()
def encode_texts(text_list):
    embs = []
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="BERT encoding"):
        batch = text_list[i:i+BATCH_SIZE]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        outputs = bert(**toks)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        if EMBED_POOL == "cls":
            pooled = last_hidden[:,0,:]  # [CLS]
        else:
            # mean-pooling (paddingを無視)
            mask = toks.attention_mask.unsqueeze(-1)  # (B, L, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        embs.append(pooled.cpu())
    return torch.cat(embs, dim=0).numpy()

X_raw = encode_texts(texts)  # shape: (N, H=768)

# スケーリング（AEに入れる前に安定化）
scaler = StandardScaler()
X = scaler.fit_transform(X_raw).astype(np.float32)

# ============== オートエンコーダ ==============
class AEDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

class AE(nn.Module):
    def __init__(self, dim, hidden=256, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

dim = X.shape[1]
ae = AE(dim=dim, hidden=HIDDEN, bottleneck=64).to(DEVICE)
opt = torch.optim.Adam(ae.parameters(), lr=LR)
crit = nn.MSELoss()

# 「正常データのみで学習」前提。実運用では初期ウィンドウを正常期間とみなす等の工夫を。
X_train, X_val = train_test_split(X, test_size=0.2, random_state=SEED)
train_loader = DataLoader(AEDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AEDataset(X_val), batch_size=BATCH_SIZE)

best_val = float("inf")
for ep in range(1, EPOCHS+1):
    ae.train()
    tr_loss = 0.0
    for xb in train_loader:
        xb = xb.to(DEVICE)
        opt.zero_grad()
        recon = ae(xb)
        loss = crit(recon, xb)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    ae.eval()
    with torch.no_grad():
        va_loss = 0.0
        for xb in val_loader:
            xb = xb.to(DEVICE)
            recon = ae(xb)
            loss = crit(recon, xb)
            va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

    print(f"Epoch {ep}/{EPOCHS}  train={tr_loss:.6f}  val={va_loss:.6f}")
    if va_loss < best_val:
        best_val = va_loss
        torch.save(ae.state_dict(), "ae_best.pth")

ae.load_state_dict(torch.load("ae_best.pth", map_location=DEVICE))
ae.eval()

# ============== 異常スコア算出（再構成誤差） ==============
@torch.no_grad()
def recon_error(Xnp):
    Xtorch = torch.from_numpy(Xnp).to(DEVICE)
    recon = ae(Xtorch).cpu().numpy()
    err = np.mean((recon - Xnp)**2, axis=1)
    return err

errors = recon_error(X)
thr = float(np.quantile(errors, THRESHOLD_Q))
labels = (errors >= thr).astype(int)  # 1: 異常候補
df["anomaly_score"] = errors
df["is_anomaly"] = labels

print(f"Threshold (q={THRESHOLD_Q}): {thr:.6f}")
print(df[["title","url","anomaly_score","is_anomaly"]].sort_values("anomaly_score", ascending=False).head(10))

# ============== SHAPで説明（テキスト→異常スコアの関数を説明） ==============
# SHAPは「関数: texts -> anomaly score」を説明できる。
# マスク（単語の置換）で入力テキストのどの部分がスコアを押し上げるかを可視化。
def _to_text_list(x):
    # SHAPやnumpy/Series経由でも必ず list[str] に揃える
    if isinstance(x, (np.ndarray, pd.Series)):
        return [str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]  # 単一文字列など

def anomaly_fn(text_like):
    texts = _to_text_list(text_like)
    # 1) BERT埋め込み
    em = encode_texts(texts)
    # 2) スケール
    ems = scaler.transform(em).astype(np.float32)
    # 3) AE再構成誤差
    return recon_error(ems)

# サンプルを小さめに（計算重いので）
sample_idx = df["anomaly_score"].nlargest(min(5, len(df))).index.tolist()
background_texts = random.sample(texts, min(20, len(texts)))
explain_texts = [texts[i] for i in sample_idx]

masker = shap.maskers.Text(tokenizer)  # トークン単位でマスク
explainer = shap.Explainer(anomaly_fn, masker)
shap_values = explainer(explain_texts)  # 各トークンが異常スコアに与える寄与

html_path = "shap_anomaly_texts.html"
with open(html_path, "w", encoding="utf-8") as f:
    for sv in shap_values:
        f.write(shap.plots.text(sv, display=False))
print(f"Saved SHAP HTML to: {html_path}")

plt.figure()
plt.hist(errors, bins=30)
plt.axvline(thr, linestyle="--")
plt.title("Anomaly score distribution (reconstruction MSE)")
plt.xlabel("score"); plt.ylabel("count")
plt.tight_layout()
plt.savefig("anomaly_score_hist.png")
print("Saved plot: anomaly_score_hist.png")

# ===== トークン抽出・日付整形 =====
def _extract_top_tokens(sv, k=3):
    import numpy as np, re
    tokens = sv.data; values = sv.values
    def _to_1d_list(x):
        if isinstance(x, list):
            return x[0] if (len(x)==1 and isinstance(x[0], (list, np.ndarray))) else x
        x = np.array(x, dtype=object)
        if x.ndim == 0: return [x.item()]
        if x.ndim == 2 and x.shape[0] == 1: x = x[0]
        return list(x)
    tokens = _to_1d_list(tokens); values = _to_1d_list(values)
    n = min(len(tokens), len(values)); tokens = tokens[:n]; values = values[:n]
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

def _to_date_str(x):
    try:
        ts = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(ts): return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None

# ドメインを人間向けに
SITE_ALIASES = {
    "pydata.org": "PyData",
    "pandas.pydata.org": "Pandas Docs",
    "numpy.org": "NumPy",
    "scikit-learn.org": "scikit-learn",
    "readthedocs.io": "Read the Docs",
    "huggingface.co": "Hugging Face",
    "transformers.huggingface.co": "Transformers Docs",
    "kaggle.com": "Kaggle",
    "github.com": "GitHub",
    "gitlab.com": "GitLab",
    "bitbucket.org": "Bitbucket",
    "stackoverflow.com": "Stack Overflow",
    "superuser.com": "Super User",
    "serverfault.com": "Server Fault",
    "arxiv.org": "arXiv",
    "medium.com": "Medium",
    "towardsdatascience.com": "TDS (Medium)",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "vimeo.com": "Vimeo",
    "netflix.com": "Netflix",
    "primevideo.com": "Prime Video",
    "bbc.com": "BBC",
    "cnn.com": "CNN",
    "nikkei.com": "日本経済新聞",
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "amazon.co.jp": "Amazon",
    "rakuten.co.jp": "楽天市場",
    "shopping.yahoo.co.jp": "Yahoo!ショッピング",
    "maps.google.com": "Google マップ",
    "google.com/maps": "Google マップ",
    "docs.python.org": "Python Docs",
    "colab.research.google.com": "Google Colab",
    "console.cloud.google.com": "Google Cloud Console",
    "portal.azure.com": "Azure Portal",
    "console.aws.amazon.com": "AWS Console",
    "classroom.google.com": "Google Classroom",
    "coursera.org": "Coursera",
    "edx.org": "edX",
    "udemy.com": "Udemy",
    "investing.com": "Investing.com",
    "tradingview.com": "TradingView",
    "coinmarketcap.com": "CoinMarketCap",
    "binance.com": "Binance",
    "openai.com": "OpenAI",
}

# 正規表現でカテゴリ推定（順番が重要：上からマッチ優先）
# (pattern, category, tone)
CATEGORY_RULES = [
    (r"(pydata|pandas|numpy|scikit|readthedocs|huggingface|transformers|docs\.python|api\s*reference)", "tech_docs", "positive"),
    (r"(kaggle|dataset|open\-?data|data\s*portal)", "open_data", "positive"),
    (r"(news|cnn|bbc|nikkei|bloomberg|reuters|nytimes|wsj)", "news", "neutral"),
    (r"(github|gitlab|bitbucket|pull\s*request|issue|commit)", "code_hosting", "positive"),
    (r"(stackoverflow|stack\s*overflow|serverfault|superuser|qiita|teratail)", "qna_forum", "positive"),
    (r"(youtube|youtu\.be|vimeo|netflix|primevideo|tver|abema)", "video_streaming", "neutral"),
    (r"(maps|google\.com/maps|map\.)", "maps_travel", "positive"),
    (r"(classroom|coursera|edx|udemy|khan\s*academy|mooc)", "education_mooc", "positive"),
    (r"(invest|tradingview|portfolio|securities|brokerage)", "finance_invest", "neutral"),
    (r"(crypto|bitcoin|ethereum|defi|web3|nft)", "crypto", "neutral"),
    (r"(casino|gambl|bet|poker|odds)", "gambling", "caution"),
    (r"(adult|porn|xxx)", "adult", "caution"),
    (r"(shopping|cart|store|amazon|rakuten|yodobashi|biccamera)", "shopping", "neutral"),
    (r"(slack|discord|teams|zoom|meet\.google|webex)", "collaboration", "positive"),
    (r"(calendar|schedule|calendar\.google)", "productivity", "positive"),
    (r"(gmail|outlook|mail\.google|yahoo\.co\.jp/mail)", "email_webmail", "neutral"),
    (r"(sports|premier\s*league|mlb|nba|nfl|nhl|j\s*league|soccer)", "sports", "neutral"),
    (r"(steam|epic\s*games|playstation|nintendo|xbox|gameplay)", "gaming", "neutral"),
    (r"(gov|go\.jp|met\.go\.jp|city\.|pref\.)", "government", "neutral"),
    (r"(health|medical|nih|who\.int|cdc\.gov|medic|drug)", "health_medical", "neutral"),
    (r"(recruit|indeed|linkedin/jobs|careers|green-japan)", "job_career", "positive"),
    (r"(ai|ml|machine\s*learning|deep\s*learning|LLM|prompt)", "ai_ml", "positive"),
    (r"(arxiv|doi\.org|ieee|acm|springer|nature\.com)", "academic_papers", "positive"),
    (r"(docs|document|manual|guide|how\s*to)", "docs", "positive"),
]

def _friendly_domain(domain: str) -> str:
    d = domain.lower()
    return SITE_ALIASES.get(d, domain)

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
        if not re.search(r"[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]", t):
            continue
        if t in {"[CLS]","[SEP]","[PAD]"} or re.fullmatch(r"\d{1,4}", t):
            continue
        kept.append(t)
    return kept[:3]

def build_human_friendly_content(
    domain: str,
    title: str,
    top_tokens: List[str],
    ts_iso: Optional[str],
    locale: str = "ja",
    row_url: str = "" 
) -> str:
    cat, tone = guess_category(domain, title, top_tokens)
    tokens = clean_tokens(top_tokens)
    hb = hour_bucket(ts_iso)
    name = _friendly_domain(domain)

    if locale == "ja":
        if cat in {"tech_docs","docs"}:
            base, add = f"{name}の技術ドキュメントを閲覧しました。", "実装や学習のヒントになりそうです。"
        elif cat == "open_data":
            base, add = f"{name}でオープンデータのリソースを見つけました。", "今後のデータ収集や分析がスムーズになりそうです。"
        elif cat == "news":
            base, add = f"{name}のニュースをチェックしました。", "最新トレンドの把握に役立ちます。"
        elif cat == "code_hosting":
            base, add = f"{name}でリポジトリやイシューを確認しました。", "開発作業の進行・調査の一環です。"
        elif cat == "qna_forum":
            base, add = f"{name}で技術的なQ&Aを参照しました。", "問題解決やノウハウ整理に有益です。"
        elif cat == "video_streaming":
            base, add = f"{name}で動画コンテンツを視聴しました。", "チュートリアルや情報収集の可能性があります。"
        elif cat == "maps_travel":
            base, add = f"{name}で場所や経路を確認しました。", "移動や外出計画の準備かもしれません。"
        elif cat == "education_mooc":
            base, add = f"{name}でオンライン学習コンテンツを確認しました。", "スキルアップの意欲が感じられます。"
        elif cat == "finance_invest":
            base, add = f"{name}で投資・相場情報を確認しました。", "意思決定には情報の信頼性にご留意ください。"
        elif cat == "crypto":
            base, add = f"{name}で暗号資産関連の情報を閲覧しました。", "価格変動が大きいため注意が必要です。"
        elif cat == "gambling":
            base, add = f"{name}にアクセスしました。", "ギャンブル関連の可能性があり、コンプライアンス上の注意が必要です。"
        elif cat == "adult":
            base, add = f"{name}にアクセスしました。", "成人向けの可能性があり、業務環境では注意が必要です。"
        elif cat == "shopping":
            base, add = f"{name}でショッピング関連のページを閲覧しました。", "比較検討のフェーズかもしれません。"
        elif cat == "collaboration":
            base, add = f"{name}でコミュニケーションを行いました。", "チーム連携の一環です。"
        elif cat == "productivity":
            base, add = f"{name}でスケジュールやタスクを確認しました。", "生産性向上の行動です。"
        elif cat == "email_webmail":
            base, add = f"{name}でメールを確認しました。", "やり取りの整理や返信の準備です。"
        elif cat == "sports":
            base, add = f"{name}でスポーツ関連情報を閲覧しました。", "試合結果やハイライトの確認かもしれません。"
        elif cat == "gaming":
            base, add = f"{name}でゲーム関連のページを閲覧しました。", "レビューや購入検討の可能性があります。"
        elif cat == "government":
            base, add = f"{name}の行政サイトを閲覧しました。", "手続きや公的情報の確認です。"
        elif cat == "health_medical":
            base, add = f"{name}で医療・健康情報を確認しました。", "情報の正確性をご確認ください。"
        elif cat == "job_career":
            base, add = f"{name}で求人・キャリア情報を確認しました。", "将来の選択肢を検討しているのかもしれません。"
        elif cat == "ai_ml":
            base, add = f"{name}でAI/機械学習に関する情報を閲覧しました。", "最新動向のキャッチアップに有効です。"
        elif cat == "academic_papers":
            base, add = f"{name}で学術論文・研究情報を確認しました。", "根拠に基づいた検討が行えます。"
        else:
            base, add = f"{name}で新しいページを閲覧しました。", "普段と少し異なる傾向が見られます。"

        detail = f"（関連語: {', '.join(tokens)}）" if tokens else ""
        nuance_map = {
            "late_night": "夜間の探索は集中しやすい一方、誤クリックにもご注意ください。",
            "early_morning": "朝の学習・情報収集に良い時間帯ですね。",
            "daytime": "業務時間帯の閲覧として適切か確認しましょう。",
            "evening": "一日の振り返りや調査に向いた時間帯です。",
            "unknown": ""
        }
        nuance = nuance_map.get(hb, "")
        return f"{base}{add}{detail} {nuance}".strip()

    # --- English templates ---
    if cat in {"tech_docs","docs"}:
        base, add = f"Visited {name} technical documentation.", "Looks helpful for learning and implementation."
    elif cat == "open_data":
        base, add = f"Found open data resources on {name}.", "Should streamline future data collection and analysis."
    elif cat == "news":
        base, add = f"Checked news on {name}.", "Useful for tracking current trends."
    elif cat == "code_hosting":
        base, add = f"Reviewed repositories/issues on {name}.", "Part of ongoing development or investigation."
    elif cat == "qna_forum":
        base, add = f"Looked up Q&A on {name}.", "Helpful for troubleshooting and know-how."
    elif cat == "video_streaming":
        base, add = f"Watched content on {name}.", "Could be tutorials or information gathering."
    elif cat == "maps_travel":
        base, add = f"Checked routes/places on {name}.", "Planning for travel or commute."
    elif cat == "education_mooc":
        base, add = f"Explored online learning on {name}.", "Great for upskilling."
    elif cat == "finance_invest":
        base, add = f"Viewed market/investment info on {name}.", "Mind the reliability for decisions."
    elif cat == "crypto":
        base, add = f"Browsed crypto-related info on {name}.", "Volatility warrants caution."
    elif cat == "gambling":
        base, add = f"Accessed {name}.", "Likely gambling-related; consider compliance risks."
    elif cat == "adult":
        base, add = f"Accessed {name}.", "Potentially adult content; be cautious in work contexts."
    elif cat == "shopping":
        base, add = f"Browsed shopping pages on {name}.", "Possibly in comparison stage."
    elif cat == "collaboration":
        base, add = f"Used {name} for communication.", "Part of team collaboration."
    elif cat == "productivity":
        base, add = f"Checked schedule/tasks on {name}.", "Supports productivity."
    elif cat == "email_webmail":
        base, add = f"Checked email on {name}.", "Organizing or preparing responses."
    elif cat == "sports":
        base, add = f"Viewed sports info on {name}.", "Maybe results or highlights."
    elif cat == "gaming":
        base, add = f"Viewed gaming pages on {name}.", "Reviews or purchase consideration."
    elif cat == "government":
        base, add = f"Visited a government site: {name}.", "Checking procedures or public info."
    elif cat == "health_medical":
        base, add = f"Checked health/medical info on {name}.", "Verify accuracy carefully."
    elif cat == "job_career":
        base, add = f"Explored jobs/career info on {name}.", "Considering future options."
    elif cat == "ai_ml":
        base, add = f"Read AI/ML information on {name}.", "Good for catching up on trends."
    elif cat == "academic_papers":
        base, add = f"Reviewed academic papers on {name}.", "Enables evidence-based discussion."
    else:
        base, add = f"Visited {name}.", "Slightly unusual pattern compared to usual behavior."

    detail = f" (key terms: {', '.join(tokens)})" if tokens else ""
    return f"{base} {add}{detail}".strip()



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
        if not re.search(r"[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]", t):  # 記号のみ除外
            continue
        if t in {"[CLS]","[SEP]","[PAD]"} or re.fullmatch(r"\d{1,4}", t):
            continue
        kept.append(t)
    return kept[:3]


def build_language_results(df, shap_values, sample_idx, threshold, locale="ja"):
    results = []

    for pos, i in enumerate(sample_idx):
        row = df.iloc[i]
        top_tokens = _extract_top_tokens(shap_values[pos], k=3)
        domain = extract_domain(row.get("url",""))
        date_str = _to_date_str(row.get(TIMESTAMP_COL)) if TIMESTAMP_COL in df.columns else None
        row_url = str(row.get("url",""))   # ← 追加
        content = build_human_friendly_content(
            domain=domain,
            title=str(row.get("title","")),
            top_tokens=top_tokens,
            ts_iso=str(row.get(TIMESTAMP_COL)) if TIMESTAMP_COL in df.columns else None,
            locale=locale,
            row_url=row_url
        )
        title_txt = f"{domain}で潜在的なリスクを検知" if row.get("is_anomaly",0)==1 else f"{domain}で新しい発見"
        results.append({
            "title": title_txt,
            "content": content,
            "timing_at": date_str
        })
    return results

# ====== ここで“呼び出す” ======
# 推定ロケールを決める
output_locale = decide_output_locale(df)

language_results = build_language_results(
    df=df,
    shap_values=shap_values,
    sample_idx=sample_idx,
    threshold=thr,
    locale=output_locale   # ← ここを自動判定に
)

print(json.dumps(language_results, ensure_ascii=False, indent=2))
with open("anomaly_language_results.json", "w", encoding="utf-8") as f:
    json.dump(language_results, f, ensure_ascii=False, indent=2)
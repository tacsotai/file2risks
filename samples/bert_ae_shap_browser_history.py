# file: bert_ae_shap_browser_history.py
import os
import math
import random
import numpy as np
import pandas as pd
import tldextract
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt

# ============== 設定 ==============
CSV_PATH = "history.csv"  # あなたの履歴CSV
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ============== データ前処理 ==============
def load_history(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # デモ用ダミーデータ
        data = {
            "title": [
                "How to use pandas merge",
                "HuggingFace Transformers tutorial",
                "News: local weather today",
                "Football highlights premier league",
                "暗号通貨 市況 ニュース",
                "Adult casino bonus offer",  # わざと怪しげ
                "Free streaming movies now", # わざと怪しげ
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
    else:
        df = pd.read_csv(csv_path)
    # 欠損対策
    for c in TEXT_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype(str)
    return df

def extract_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
        return domain or url
    except Exception:
        return url

def compose_text(row) -> str:
    title = row.get("title","")
    url = row.get("url","")
    domain = extract_domain(url)
    # ここで危険キーワードや時間帯特徴を含めてもよい
    ts = row.get(TIMESTAMP_COL, "")
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

# 可視化（HTMLを保存）
html_path = "shap_anomaly_texts.html"
with open(html_path, "w", encoding="utf-8") as f:
    for sv in shap_values:
        f.write(shap.plots.text(sv, display=False))
print(f"Saved SHAP HTML to: {html_path}")

# ついでにスコア分布の簡易プロット
plt.figure()
plt.hist(errors, bins=30)
plt.axvline(thr, linestyle="--")
plt.title("Anomaly score distribution (reconstruction MSE)")
plt.xlabel("score"); plt.ylabel("count")
plt.tight_layout()
plt.savefig("anomaly_score_hist.png")
print("Saved plot: anomaly_score_hist.png")

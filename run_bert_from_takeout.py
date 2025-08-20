#!/usr/bin/env python3
# save as: run_bert_from_takeout.py

import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
import subprocess
from typing import List, Optional
from typing import Any
import importlib.util

HISTORY_NAME_PATTERNS = [
    r"history",          # EN
    r"履歴",              # JA
    r"historial",        # ES
    r"história",         # PT
    r"historia",         # PL/ES
    r"historie",         # CS/NL/DK/NO
    r"historique",       # FR
    r"cronologia",       # IT
    r"chronik",          # DE
    r"история",          # RU
    r"歷史|历史",         # ZH
    r"istoria|istoric",  # RO
]

def guess_history_json_file(extract_dir: Path) -> Optional[Path]:
    """
    1) ファイル名に "history/履歴/各国語" を含む .json を優先
    2) ダメなら .json を総当りして「url キーを多く含む」もの（サイズも考慮）
    """
    json_files = [p for p in extract_dir.rglob("*.json") if p.is_file()]

    if not json_files:
        return None

    # 1) ファイル名によるスコアリング
    name_regex = re.compile("|".join(HISTORY_NAME_PATTERNS), re.IGNORECASE)
    name_hit = [p for p in json_files if name_regex.search(p.name)]
    if name_hit:
        # 一応サイズ大を優先（履歴が長いほど大きい想定）
        name_hit.sort(key=lambda p: p.stat().st_size, reverse=True)
        return name_hit[0]

    # 2) コンテンツを軽く覗いて "url" が多い（履歴アイテムにありがち）ものを選ぶ
    best = None
    best_score = -1
    for p in json_files:
        try:
            # 10万バイト程度までざっくり読み取って "url" の出現回数を指標に
            with open(p, "rb") as f:
                chunk = f.read(100_000).decode("utf-8", errors="ignore")
            score = chunk.count('"url"') + chunk.count("'url'")
            # サイズも少し加点
            score += min(p.stat().st_size // 50_000, 20)
            if score > best_score:
                best_score = score
                best = p
        except Exception:
            continue
    return best

def extract_zip(zip_path: Path, to_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(to_dir)

def run_anomaly_detection(zip_path: str, bert_path: Optional[str] = None) -> List[dict]:
    """
    Google Takeout の ZIP を展開して history.json を用意し、
    bert_ae_shap_browser_history.py の calc_risks(history.json) を直接呼び出して
    List[dict] を返す。
    - bert_path: bert_ae_shap_browser_history.py のファイルパス（未指定なら samples/ を自動探索）
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="takeout_extract_")
    try:
        workdir = Path(temp_dir.name)
        extract_dir = workdir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # bert モジュールの場所を決定
        if bert_path is None:
            candidate = Path(__file__).resolve().parent / "samples/bert_ae_shap_browser_history.py"
            if not candidate.exists():
                raise FileNotFoundError("bert_ae_shap_browser_history.py が見つかりません（samples/ 配下を確認してください）。")
            bert_module_path = candidate
        else:
            bert_module_path = Path(bert_path).resolve()
            if not bert_module_path.exists():
                raise FileNotFoundError(f"指定の bert モジュールが見つかりません: {bert_module_path}")

        # ZIP 展開
        zp = Path(zip_path).resolve()
        if not zp.exists():
            raise FileNotFoundError(f"ZIP が見つかりません: {zp}")
        extract_zip(zp, extract_dir)

        # history.json を特定して作業ディレクトリ直下に配置
        hist_json = guess_history_json_file(extract_dir)
        if not hist_json:
            raise FileNotFoundError("履歴 JSON が見つかりません")
        target_history = workdir / "history.json"
        shutil.copy2(hist_json, target_history)

        # 動的 import で calc_risks をロード
        spec = importlib.util.spec_from_file_location("bert_ae_shap_browser_history", str(bert_module_path))
        if spec is None or spec.loader is None:
            raise ImportError("bert_ae_shap_browser_history のロードに失敗しました。")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "calc_risks"):
            raise AttributeError("bert_ae_shap_browser_history に calc_risks がありません。")

        # 直接呼び出し
        results: List[dict] = module.calc_risks(str(target_history))
        return results

    finally:
        # 作業ディレクトリを後始末
        temp_dir.cleanup()

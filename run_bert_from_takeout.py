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

def main():
    parser = argparse.ArgumentParser(
        description="Google Takeout のブラウザ履歴 ZIP を展開し、history.json を作ってから "
                    "bert_ae_shap_browser_history.py を実行します。"
    )
    parser.add_argument("zip_path", type=Path, help="Google Takeout の ZIP パス (例: xxx/yyy/zzz.zip)")
    parser.add_argument("--bert", type=Path, default=None,
                        help="bert_ae_shap_browser_history.py のパス（未指定なら同ディレクトリ検索）")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="history.json を置いて実行する作業ディレクトリ（未指定なら一時ディレクトリ）")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="終了後も作業ディレクトリを残す（デバッグ用）")
    args = parser.parse_args()

    # bert スクリプトの特定
    if args.bert is None:
        # このラッパースクリプトと同じディレクトリにある想定
        candidate = Path(__file__).resolve().parent / "samples/bert_ae_shap_browser_history.py"
        if not candidate.exists():
            print("ERROR: --bert で bert_ae_shap_browser_history.py のパスを指定してください。", file=sys.stderr)
            sys.exit(1)
        bert_path = candidate
    else:
        bert_path = args.bert.resolve()
        if not bert_path.exists():
            print(f"ERROR: 指定の bert スクリプトが見つかりません: {bert_path}", file=sys.stderr)
            sys.exit(1)

    zip_path = args.zip_path.resolve()
    if not zip_path.exists():
        print(f"ERROR: ZIP が見つかりません: {zip_path}", file=sys.stderr)
        sys.exit(1)

    # 作業ディレクトリ
    temp_dir = None
    if args.workdir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="takeout_extract_")
        workdir = Path(temp_dir.name)
    else:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    extract_dir = workdir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting: {zip_path} -> {extract_dir}")
    try:
        extract_zip(zip_path, extract_dir)
    except zipfile.BadZipFile:
        print("ERROR: ZIP が壊れている可能性があります。", file=sys.stderr)
        if temp_dir: temp_dir.cleanup()
        sys.exit(1)

    print("[2/4] Searching for history JSON (言語差対応)")
    hist_json = guess_history_json_file(extract_dir)
    if not hist_json:
        print("ERROR: 履歴 JSON が見つかりませんでした。ZIP 内容を確認してください。", file=sys.stderr)
        if temp_dir and not args.keep_workdir:
            temp_dir.cleanup()
        sys.exit(1)

    print(f"    -> candidate: {hist_json}")

    # 出力 history.json を作成
    target_history = workdir / "history.json"
    print(f"[3/4] Writing normalized history.json -> {target_history}")
    try:
        # そのままコピー（bert スクリプトが 'history.json' を読む想定）
        shutil.copy2(hist_json, target_history)
    except Exception as e:
        print(f"ERROR: history.json の作成に失敗しました: {e}", file=sys.stderr)
        if temp_dir and not args.keep_workdir:
            temp_dir.cleanup()
        sys.exit(1)

    # bert スクリプト実行（カレントを workdir にして history.json を見つけやすくする）
    print(f"[4/4] Running bert script: {bert_path.name} (cwd={workdir})")
    try:
        proc = subprocess.run(
            [sys.executable, str(bert_path)],
            cwd=str(workdir),
            check=False
        )
        code = proc.returncode
    except FileNotFoundError:
        print("ERROR: Python 実行に失敗しました。環境を確認してください。", file=sys.stderr)
        code = 1

    # 一時ディレクトリ掃除
    if temp_dir and not args.keep_workdir:
        temp_dir.cleanup()

    if code != 0:
        print(f"bert スクリプトがエラー終了しました (exit={code})", file=sys.stderr)
        sys.exit(code)

    print("Done.")

if __name__ == "__main__":
    main()

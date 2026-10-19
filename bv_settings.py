"""アプリ設定（保存先フォルダ内の settings.json）とバックアップ処理。"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import db

SETTINGS_PATH = db.BASE_DIR / "settings.json"
DEFAULTS = {
    "backup_dir": "",  # バックアップ先(未設定なら自動バックアップは動かない)
    "auto_backup": True,  # 起動した日に1回、自動でバックアップ
    "keep": 14,  # DBバックアップを残す世代数
    "include_csv": True,  # 患者別フォルダの元CSVも複製する
}
DB_PREFIX = "bv_research_"


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg: dict) -> None:
    db.BASE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def check_dest(dest: str) -> tuple[bool, str]:
    """バックアップ先として使えるか。(OK?, メッセージ)"""
    if not dest.strip():
        return False, "バックアップ先が未設定です"
    p = Path(dest.strip().strip('"'))
    try:
        p_res = p.resolve()
    except OSError:
        return False, "パスが不正です"
    base = db.BASE_DIR.resolve()
    if p_res == base or base in p_res.parents or p_res in base.parents:
        return False, "データ保存先と同じ(または包含関係の)フォルダは指定できません"
    if not p.exists():
        parent = p.parent
        if not parent.exists():
            return False, "親フォルダが存在しません（ドライブが接続されているか確認してください）"
    return True, ""


def list_backups(dest: str) -> list[Path]:
    p = Path(dest.strip().strip('"')) if dest.strip() else None
    if not p or not p.exists():
        return []
    return sorted(p.glob(f"{DB_PREFIX}*.db"))


def last_backup_time(dest: str) -> datetime | None:
    files = list_backups(dest)
    return datetime.fromtimestamp(files[-1].stat().st_mtime) if files else None


def backup_now(dest: str, keep: int, include_csv: bool) -> dict:
    """DBを整合性を保ったままコピーし、古い世代を削除。元CSVは未コピー分のみ複製。"""
    ok, msg = check_dest(dest)
    if not ok:
        raise ValueError(msg)
    d = Path(dest.strip().strip('"'))
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"{DB_PREFIX}{datetime.now():%Y%m%d_%H%M%S}.db"
    src = sqlite3.connect(db.DB_PATH)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)  # 使用中でも一貫性のあるコピーになる
    finally:
        dst.close()
        src.close()
    removed = 0
    files = list_backups(dest)
    for f in files[:-max(1, keep)]:
        f.unlink(missing_ok=True)
        removed += 1
    copied = 0
    if include_csv and db.PATIENT_DIR.exists():
        for f in db.PATIENT_DIR.rglob("*"):
            if not f.is_file():
                continue
            out = d / "patients" / f.relative_to(db.PATIENT_DIR)
            if not out.exists() or out.stat().st_size != f.stat().st_size:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                copied += 1
    return {"file": target, "size_mb": target.stat().st_size / 1e6, "removed": removed, "csv_copied": copied}


def auto_backup_if_due(cfg: dict) -> str | None:
    """自動バックアップが有効で、今日まだ作っていなければ実行。結果メッセージ(なければNone)。"""
    if not cfg.get("auto_backup") or not cfg.get("backup_dir"):
        return None
    ok, _ = check_dest(cfg["backup_dir"])
    if not ok:
        return None
    last = last_backup_time(cfg["backup_dir"])
    if last and last.date() == datetime.now().date():
        return None
    try:
        r = backup_now(cfg["backup_dir"], int(cfg.get("keep", 14)), bool(cfg.get("include_csv", True)))
        return f"自動バックアップを作成しました（{r['file'].name}）"
    except Exception as e:  # noqa: BLE001
        return f"自動バックアップに失敗しました: {e}"

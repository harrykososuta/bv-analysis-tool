"""SQLite 永続化層。患者ごとのフォルダに元CSVを保存し、解析済みデータをDBへ蓄積する。"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from bv_parser import TS_COLUMNS, ParsedSession, meta_to_json

# 既定の保存先はクラウド同期されないホーム直下(OneDrive配下は避ける)。環境変数 BV_DATA_DIR で変更可。
BASE_DIR = Path(os.environ.get("BV_DATA_DIR", Path.home() / "BV_Tool_Data"))
DB_PATH = BASE_DIR / "bv_research.db"
PATIENT_DIR = BASE_DIR / "patients"

CLINICAL_COLS = ["hypo_event", "symptom", "intervention", "memo"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER);
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    alias TEXT, note TEXT, name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    session_dt TEXT NOT NULL,
    source_filename TEXT, stored_path TEXT,
    file_hash TEXT UNIQUE,
    treat_sec INTEGER, n_rows INTEGER,
    device TEXT, version TEXT, header_json TEXT,
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
    pre_weight REAL, post_weight REAL, dry_weight REAL, target_uf REAL,
    hypo_event INTEGER, symptom TEXT, intervention TEXT, memo TEXT,
    treat_mode INTEGER, mode_set TEXT, console_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_sessions_patient ON sessions(patient_id, session_dt);
CREATE TABLE IF NOT EXISTS timeseries (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    row_no INTEGER NOT NULL,
    clock TEXT,
    {", ".join(f"{c} REAL" for c in TS_COLUMNS)},
    PRIMARY KEY (session_id, row_no)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ufc_settings (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    saved_at TEXT DEFAULT (datetime('now', 'localtime')),
    settings_json TEXT NOT NULL, memo TEXT
);
INSERT OR IGNORE INTO meta(k, v) VALUES ('revision', 0);
"""


@contextmanager
def conn():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        cols = [r[1] for r in c.execute("PRAGMA table_info(patients)")]
        if "name" not in cols:  # 旧版DBからの移行
            c.execute("ALTER TABLE patients ADD COLUMN name TEXT")
        scols = [r[1] for r in c.execute("PRAGMA table_info(sessions)")]
        for col, typ in (("treat_mode", "INTEGER"), ("mode_set", "TEXT"), ("console_id", "TEXT")):
            if col not in scols:
                c.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typ}")
        tcols = [r[1] for r in c.execute("PRAGMA table_info(timeseries)")]
        for col in TS_COLUMNS:  # 解析列の追加(旧版DBからの移行)
            if col not in tcols:
                c.execute(f"ALTER TABLE timeseries ADD COLUMN {col} REAL")


def migrate_bp_clean() -> int:
    """血圧の無効値(未測定の -1)を欠測(NULL)にする。一度だけ実行し、実行前にDBを同じフォルダへコピーする。"""
    with conn() as c:
        done = c.execute("SELECT v FROM meta WHERE k='bp_clean_v1'").fetchone()
        if done and done[0] == 1:
            return 0
        n = c.execute("SELECT COUNT(*) FROM timeseries WHERE sbp<=0 OR dbp<=0 OR pulse<=0").fetchone()[0]
        if n:
            shutil.copy2(DB_PATH, BASE_DIR / "bv_research_before_bp_clean.db")
            c.execute("UPDATE timeseries SET sbp=NULL, dbp=NULL, pulse=NULL WHERE sbp<=0 OR dbp<=0 OR pulse<=0")
            _bump(c)
        c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('bp_clean_v1', 1)")
    return n


def backfill_modes() -> int:
    """治療モード未設定の症例を、保存済みCSVから補完する。補完した件数を返す。"""
    from bv_parser import read_treat_modes
    with conn() as c:
        rows = c.execute("SELECT session_id, stored_path FROM sessions WHERE treat_mode IS NULL "
                         "AND stored_path IS NOT NULL").fetchall()
        n = 0
        for sid, path in rows:
            try:
                mode, mset = read_treat_modes(Path(path).read_bytes())
            except Exception:  # noqa: BLE001  (ファイル欠損・形式不正は飛ばす)
                continue
            if mode is not None:
                c.execute("UPDATE sessions SET treat_mode=?, mode_set=? WHERE session_id=?", (mode, mset, sid))
                n += 1
        if n:
            _bump(c)
    return n


def backfill_ufc_steps() -> int:
    """BV-UFCステップ列を、保存済みCSVから補完する(一度だけ)。補完した症例数を返す。"""
    from bv_parser import parse_csv
    with conn() as c:
        done = c.execute("SELECT v FROM meta WHERE k='ufc_step_v1'").fetchone()
        if done and done[0] == 1:
            return 0
        rows = c.execute("SELECT session_id, stored_path, source_filename, n_rows FROM sessions "
                         "WHERE stored_path IS NOT NULL").fetchall()
        if rows:
            shutil.copy2(DB_PATH, BASE_DIR / "bv_research_before_ufc_step.db")
        n = 0
        for sid, path, fname, n_rows in rows:
            try:
                ts = parse_csv(Path(path).read_bytes(), fname or Path(path).name).ts
            except Exception:  # noqa: BLE001  (ファイル欠損・形式不正は飛ばす)
                continue
            if len(ts) != n_rows:  # 行の対応が取れない場合は書き込まない
                continue
            c.executemany("UPDATE timeseries SET ufc_step=? WHERE session_id=? AND row_no=?",
                          [(None if pd.isna(v) else float(v), sid, int(i))
                           for i, v in zip(ts["row_no"], ts["ufc_step"])])
            n += 1
        c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('ufc_step_v1', 1)")
        if n:
            _bump(c)
    return n


def migrate_patient_by_name() -> dict:
    """旧版ではCSVの「ID」(実際は装置の番号)を患者IDにしていたため、同じIDに複数の患者が混ざっていた。
    保存済みCSVの氏名から患者を分け直し、装置IDは console_id に移す。元CSVは患者ごとのフォルダへ移動する。
    一度だけ実行し、実行前にDBを同じフォルダへコピーする。"""
    from bv_parser import META_ID_IDX, META_NAME_IDX, N_DATA_COLS, decode, patient_key
    with conn() as c:
        done = c.execute("SELECT v FROM meta WHERE k='patient_by_name_v1'").fetchone()
        if done and done[0] == 1:
            return {}
        rows = c.execute("SELECT session_id, patient_id, stored_path FROM sessions").fetchall()
        if rows:
            shutil.copy2(DB_PATH, BASE_DIR / "bv_research_before_patient_split.db")
        moved, unknown = 0, 0
        for sid, old, path in rows:
            name, console = "", old
            try:
                first = decode(Path(path).read_bytes()).splitlines()[0].split(",")
                meta = first[N_DATA_COLS:]
                name = meta[META_NAME_IDX].strip() if len(meta) > META_NAME_IDX else ""
                console = (meta[META_ID_IDX].strip() if len(meta) > META_ID_IDX else "") or old
            except Exception:  # noqa: BLE001  (元CSVが無い・読めない症例は装置IDのまま)
                c.execute("UPDATE sessions SET console_id=COALESCE(console_id, ?) WHERE session_id=?", (old, sid))
                unknown += 1
                continue
            new = patient_key("" if "�" in name else name, console)
            c.execute("INSERT OR IGNORE INTO patients(patient_id) VALUES (?)", (new,))
            if name and "�" not in name:
                c.execute("UPDATE patients SET name=? WHERE patient_id=? AND (name IS NULL OR name='')", (name, new))
            new_path = path
            src = Path(path)
            dst_dir = PATIENT_DIR / safe_name(new)
            if src.exists() and src.parent != dst_dir:  # 元CSVを新しい患者フォルダへ移す
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if dst.exists():
                    dst = dst_dir / f"{src.stem}_{sid}{src.suffix}"
                src.replace(dst)
                new_path = str(dst)
            c.execute("UPDATE sessions SET patient_id=?, console_id=?, stored_path=? WHERE session_id=?",
                      (new, console, new_path, sid))
            moved += 1
        # 症例が無くなった旧患者(装置ID)は、手入力の情報・保存した設定が無ければ削除
        c.execute("""DELETE FROM patients WHERE patient_id NOT IN (SELECT DISTINCT patient_id FROM sessions)
                     AND COALESCE(alias,'')='' AND COALESCE(note,'')=''
                     AND patient_id NOT IN (SELECT patient_id FROM ufc_settings)""")
        for d in PATIENT_DIR.glob("*") if PATIENT_DIR.exists() else []:  # 空になった旧フォルダを片付ける
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('patient_by_name_v1', 1)")
        if rows:
            _bump(c)
    return {"moved": moved, "unknown": unknown}


def save_ufc_setting(patient_id: str, settings_json: str, memo: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO ufc_settings(patient_id, settings_json, memo) VALUES (?,?,?)",
                  (patient_id, settings_json, memo or None))
        _bump(c)


def ufc_settings_df(patient_id: str | None = None) -> pd.DataFrame:
    with conn() as c:
        q = "SELECT * FROM ufc_settings" + (" WHERE patient_id=?" if patient_id else "") +             " ORDER BY saved_at DESC, setting_id DESC"
        return pd.read_sql(q, c, params=(patient_id,) if patient_id else None)


def delete_ufc_setting(setting_id: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM ufc_settings WHERE setting_id=?", (setting_id,))
        _bump(c)


def session_uf() -> pd.DataFrame:
    """症例ごとの総除水量[L](時系列の最大値)。"""
    with conn() as c:
        return pd.read_sql("SELECT session_id, MAX(uf_vol) AS uf_total FROM timeseries GROUP BY session_id", c)


def _bump(c: sqlite3.Connection) -> None:
    c.execute("UPDATE meta SET v = v + 1 WHERE k = 'revision'")


def revision() -> int:
    with conn() as c:
        return c.execute("SELECT v FROM meta WHERE k='revision'").fetchone()[0]


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_") or "unknown"


def hash_exists(h: str) -> bool:
    with conn() as c:
        return c.execute("SELECT 1 FROM sessions WHERE file_hash=?", (h,)).fetchone() is not None


def session_spans() -> pd.DataFrame:
    """症例ごとの測定期間(最初と最後の時刻)と行数。同一症例の重複判定に使う。"""
    with conn() as c:
        return pd.read_sql(
            """SELECT s.session_id, s.patient_id, s.n_rows, MIN(t.clock) AS t0, MAX(t.clock) AS t1
               FROM sessions s JOIN timeseries t USING(session_id) GROUP BY s.session_id""", c)


# 置換時に旧症例から引き継ぐ手入力項目
CARRY_COLS = ["pre_weight", "post_weight", "dry_weight", "target_uf", *CLINICAL_COLS]


def add_session(p: ParsedSession, raw_to_store: bytes | None, store_name: bool = True,
                replace_ids: list[int] | None = None) -> int:
    """1症例を登録し session_id を返す。raw_to_store があれば患者フォルダへ保存する。
    replace_ids があれば、その症例の手入力項目を引き継いだうえで旧症例を削除する(元CSVの複製は残す)。"""
    stored = None
    if raw_to_store is not None:
        d = PATIENT_DIR / safe_name(p.patient_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / p.filename
        if path.exists():
            path = d / f"{path.stem}_{p.file_hash[:6]}{path.suffix}"
        path.write_bytes(raw_to_store)
        stored = str(path)
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO patients(patient_id) VALUES (?)", (p.patient_id,))
        if store_name and p.patient_name:  # 未登録の場合のみ。手動修正した氏名は上書きしない
            c.execute("UPDATE patients SET name=? WHERE patient_id=? AND (name IS NULL OR name='')",
                      (p.patient_name, p.patient_id))
        cur = c.execute(
            """INSERT INTO sessions(patient_id, session_dt, source_filename, stored_path, file_hash,
                   treat_sec, n_rows, device, version, header_json, treat_mode, mode_set, console_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (p.patient_id, p.session_dt.strftime("%Y-%m-%d %H:%M:%S"), p.filename, stored,
             p.file_hash, p.treat_sec, p.n_rows, p.device, p.version, meta_to_json(p.header_meta),
             p.treat_mode, p.mode_set, p.console_id),
        )
        sid = cur.lastrowid
        ts = p.ts.copy()
        ts.insert(0, "session_id", sid)
        ts.to_sql("timeseries", c, if_exists="append", index=False, chunksize=500)
        if replace_ids:
            marks = ",".join("?" * len(replace_ids))
            for col in CARRY_COLS:  # 旧症例のうち新しい登録の値を優先
                row = c.execute(f"SELECT {col} FROM sessions WHERE session_id IN ({marks}) AND {col} IS NOT NULL "
                                "ORDER BY session_id DESC LIMIT 1", replace_ids).fetchone()
                if row:
                    c.execute(f"UPDATE sessions SET {col}=? WHERE session_id=?", (row[0], sid))
            c.execute(f"DELETE FROM sessions WHERE session_id IN ({marks})", replace_ids)
        _bump(c)
    return sid


def patients_df() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql(
            """SELECT p.patient_id, p.name, p.alias, p.note, COUNT(s.session_id) AS n_sessions,
                      MIN(s.session_dt) AS first_dt, MAX(s.session_dt) AS last_dt,
                      GROUP_CONCAT(DISTINCT s.console_id) AS consoles
               FROM patients p LEFT JOIN sessions s USING(patient_id)
               GROUP BY p.patient_id ORDER BY p.patient_id""", c)


def sessions_df() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql(
            "SELECT * FROM sessions ORDER BY patient_id, session_dt", c, parse_dates=["session_dt"])


def timeseries(session_id: int) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql(
            "SELECT * FROM timeseries WHERE session_id=? ORDER BY row_no", c, params=(session_id,))


def patient_timeseries(patient_id: str) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql(
            """SELECT t.* FROM timeseries t JOIN sessions s USING(session_id)
               WHERE s.patient_id=? ORDER BY t.session_id, t.row_no""", c, params=(patient_id,))


def all_timeseries() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM timeseries ORDER BY session_id, row_no", c)


def update_clinical(rows: pd.DataFrame) -> None:
    """rows: session_id + CLINICAL_COLS"""
    with conn() as c:
        for r in rows.to_dict("records"):
            vals = [None if pd.isna(r[k]) else r[k] for k in CLINICAL_COLS]
            if vals[0] is not None:  # hypo_event
                vals[0] = int(vals[0])
            c.execute(
                f"UPDATE sessions SET {', '.join(k + '=?' for k in CLINICAL_COLS)} WHERE session_id=?",
                (*vals, int(r["session_id"])),
            )
        _bump(c)


def update_patients(rows: pd.DataFrame) -> None:
    """alias / note と、列があれば name を更新する。"""
    fields = [k for k in ("name", "alias", "note") if k in rows.columns]
    with conn() as c:
        for r in rows.to_dict("records"):
            c.execute(f"UPDATE patients SET {', '.join(k + '=?' for k in fields)} WHERE patient_id=?",
                      (*[None if pd.isna(r[k]) or r[k] == "" else r[k] for k in fields], r["patient_id"]))
        _bump(c)


def delete_session(session_id: int, delete_file: bool = False) -> None:
    with conn() as c:
        row = c.execute("SELECT stored_path FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        _bump(c)
    if delete_file and row and row[0]:
        Path(row[0]).unlink(missing_ok=True)


def delete_patient(patient_id: str, delete_files: bool = False) -> None:
    with conn() as c:
        paths = [r[0] for r in c.execute("SELECT stored_path FROM sessions WHERE patient_id=? AND stored_path IS NOT NULL",
                                         (patient_id,))]
        c.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
        _bump(c)
    if delete_files:  # 統合した患者などは別フォルダに元CSVがあるため、症例ごとの保存先も消す
        for p in paths:
            Path(p).unlink(missing_ok=True)
        shutil.rmtree(PATIENT_DIR / safe_name(patient_id), ignore_errors=True)


def merge_patients(src: str, dst: str) -> int:
    """患者 src の症例・保存した設定を dst へ移し、src を削除する(同一人物が別の氏名で登録された場合)。移した症例数を返す。"""
    with conn() as c:
        n = c.execute("UPDATE sessions SET patient_id=? WHERE patient_id=?", (dst, src)).rowcount
        c.execute("UPDATE ufc_settings SET patient_id=? WHERE patient_id=?", (dst, src))
        a = c.execute("SELECT alias, note FROM patients WHERE patient_id=?", (src,)).fetchone()
        if a:  # 統合先が空欄なら表示名・メモを引き継ぐ
            c.execute("UPDATE patients SET alias=COALESCE(NULLIF(alias,''), ?), note=COALESCE(NULLIF(note,''), ?) "
                      "WHERE patient_id=?", (a[0], a[1], dst))
        c.execute("DELETE FROM patients WHERE patient_id=?", (src,))
        _bump(c)
    return n

"""DCS-100NX (BV計) CSV の読み込み・整形。

- エンコーディングは Shift-JIS(cp932)
- 1行目 = 列名 + 装置/患者メタ情報(同一行に混在)
- 数値の多くは ×10 / ×100 の整数 → 実単位へ変換
- -9999 は欠測 → NaN
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

N_DATA_COLS = 41  # 'hour' ... 'ART press ave(mV)'
META_NAME_IDX = 8  # メタ情報内の氏名位置
META_ID_IDX = 9  # メタ情報内の「ID」位置。実際は装置(コンソール)の番号で、同じIDを複数の患者が使う

# 元列名 -> (新列名, 除数)
COLUMN_MAP = {
    "treat-time[sec]": ("treat_sec", 1),
    "dBV[%]*10": ("dbv", 10),
    "UFP-speed[L/h]*100": ("ufr", 100),
    "UF-volume[L]*100": ("uf_vol", 100),
    "BP-speed[mL/min]": ("qb_set", 1),
    "total-cond[mS/cm]*10": ("cond", 10),
    "sys-BP[mmHg]": ("sbp", 1),
    "dia-BP[mmHg]": ("dbp", 1),
    "pulse[bpm]": ("pulse", 1),
    "blood flow volume[L]*10": ("bf_vol", 10),
    "event": ("event", 1),
    "dBV reference area upper limit[%]*10": ("ref_upper", 10),
    "dBV reference area lower limit[%]*10": ("ref_lower", 10),
    "dBV AVR_5min(remove Max_Min)[%]*10": ("dbv_avg5", 10),
    "PRR[L/h]*100": ("prr", 100),
    "DP[mL/min]": ("dp", 1),
    "Estimate Blood Flow(mL/min)": ("est_qb", 1),
    "ART press ave(mV)": ("art_mv", 1),
    "BV-UFC step no": ("ufc_step", 1),
}
TS_COLUMNS = [v[0] for v in COLUMN_MAP.values()]  # DB に保存する時系列列

COLUMN_LABELS = {
    "dbv": "ΔBV (%)",
    "dbv_avg5": "ΔBV 5分平均 (%)",
    "ref_upper": "リファレンスエリア 上限 (%)",
    "ref_lower": "リファレンスエリア 下限 (%)",
    "ufr": "除水速度 (L/h)",
    "uf_vol": "除水量 (L)",
    "qb_set": "血液ポンプ流量 (mL/min)",
    "bf_vol": "処理血液量 (L)",
    "sbp": "収縮期血圧 (mmHg)",
    "dbp": "拡張期血圧 (mmHg)",
    "pulse": "脈拍 (bpm)",
    "cond": "総コンダクタンス (mS/cm)",
    "prr": "PRR (L/h)",
    "ufc_step": "BV-UFC ステップ",
}


# DCS-100NX 操作マニュアル 表12-1 より
EVENT_LABELS = {
    1: "ΔBV初期化", 2: "除水停止", 3: "除水再開",
    4: "再循環率測定開始(濃縮式)", 5: "再循環率測定終了(濃縮式)",
    6: "再循環率測定開始(希釈式)", 7: "再循環率測定終了(希釈式)", 99: "その他(手動入力)",
}
EVENT_GROUPS = {  # 画面での表示切替用
    "ΔBV初期化": {1}, "除水停止/再開": {2, 3}, "再循環率測定": {4, 5, 6, 7}, "その他": {99},
}
TREAT_MODE_LABELS = {
    0: "HD", 1: "O-HDF(pre)", 2: "O-HDF(post)", 3: "O-HF(pre)", 4: "O-HF(post)", 5: "ECUM",
    6: "I-HDF", 7: "HDF(pre)", 8: "HDF(post)", 9: "HF(pre)", 10: "HF(post)",
}


def decode_event(v) -> list[int]:
    """event値を個別コードへ分解。20秒以内に複数発生した場合は2桁ずつ連結される(例: 203 -> [2, 3], 20399 -> [2, 3, 99])。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    n = int(v)
    if n == 0:
        return []
    t = str(n)
    if len(t) % 2:
        t = "0" + t
    return [int(t[i:i + 2]) for i in range(0, len(t), 2)]


def event_text(v) -> str:
    return " + ".join(EVENT_LABELS.get(c, f"コード{c}") for c in decode_event(v))


def mode_label(m) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "不明"
    return TREAT_MODE_LABELS.get(int(m), f"モード{int(m)}")


def read_treat_modes(raw: bytes) -> tuple[int | None, str]:
    """CSVのtreat-mode列(Y列)から (最頻モード, 出現モード一覧) を返す。"""
    text = decode(raw)
    df = pd.read_csv(io.StringIO(text), index_col=False, usecols=["treat-mode"])
    m = pd.to_numeric(df["treat-mode"], errors="coerce").dropna().astype(int)
    if m.empty:
        return None, ""
    return int(m.mode().iloc[0]), ",".join(str(x) for x in sorted(m.unique()))


@dataclass
class ParsedSession:
    filename: str
    file_hash: str
    patient_id: str
    session_dt: datetime
    treat_sec: int
    n_rows: int
    device: str
    version: str
    header_meta: dict
    ts: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    patient_name: str = ""
    treat_mode: int | None = None
    mode_set: str = ""
    console_id: str = ""


def normalize_name(name: str) -> str:
    """氏名の表記ゆれ(全角/半角・空白)をそろえる。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name or ""))


def patient_key(name: str, console_id: str) -> str:
    """氏名から患者IDを作る(氏名そのものはIDに含めない)。氏名が無ければ装置IDで仮の患者とする。"""
    n = normalize_name(name)
    if not n:
        return f"不明_{console_id or 'unknown'}"
    return "P" + hashlib.sha1(n.encode("utf-8")).hexdigest()[:8].upper()


def decode(raw: bytes) -> str:
    try:
        return raw.decode("cp932")
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="replace")


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def anonymize_bytes(raw: bytes) -> bytes:
    """1行目の氏名フィールドを空にしたCSVを返す(患者フォルダ保存用)。"""
    text = decode(raw)
    nl = text.find("\n")
    first, rest = (text, "") if nl < 0 else (text[:nl], text[nl:])
    cr = first.endswith("\r")
    fields = first.rstrip("\r").split(",")
    idx = N_DATA_COLS + META_NAME_IDX
    if len(fields) > idx:
        fields[idx] = ""
    return (",".join(fields) + ("\r" if cr else "") + rest).encode("cp932", errors="replace")


def _session_start(filename: str, first_clock: tuple[int, int, int]) -> tuple[datetime, bool]:
    m = re.search(r"(\d{8})_(\d{6})", filename)
    h, mi, s = first_clock
    if m:
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        return datetime(d.year, d.month, d.day, h, mi, s), True
    d = datetime.now().date()
    return datetime(d.year, d.month, d.day, h, mi, s), False


def parse_csv(raw: bytes, filename: str, patient_id_override: str | None = None) -> ParsedSession:
    text = decode(raw)
    first_line = text.split("\n", 1)[0].rstrip("\r")
    fields = first_line.split(",")
    if len(fields) < N_DATA_COLS or fields[0].strip() != "hour" or "dBV[%]*10" not in fields:
        raise ValueError("DCS-100NX形式のCSVではありません(ヘッダー不一致)")

    df = pd.read_csv(io.StringIO(text), index_col=False, header=0)
    df = df.iloc[:, :N_DATA_COLS]
    df.columns = fields[:N_DATA_COLS]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("データ行がありません")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df == -9999)

    warnings: list[str] = []
    meta = fields[N_DATA_COLS:]
    device = meta[5].strip() if len(meta) > 5 else ""
    version = meta[6].strip() if len(meta) > 6 else ""
    console = meta[META_ID_IDX].strip() if len(meta) > META_ID_IDX else ""
    if console in ("", "-1", "0"):
        console = re.split(r"[_\.]", filename)[0]
    name = meta[META_NAME_IDX].strip() if len(meta) > META_NAME_IDX else ""
    if "�" in name:
        name = ""
        warnings.append("氏名の文字コードを判別できず未取得")
    pid = patient_key(name, console)
    if not name:
        warnings.append("氏名が無いため、装置IDで仮の患者として登録")
    if patient_id_override:
        pid = patient_id_override.strip()

    out = pd.DataFrame()
    for src, (dst, div) in COLUMN_MAP.items():
        out[dst] = df[src] / div if div != 1 else df[src]

    for c in ("sbp", "dbp", "pulse"):  # 未測定は -1 で出力される → 欠測として扱う
        out.loc[out[c] <= 0, c] = np.nan

    # 時刻(日跨ぎ対応)
    h, mi, s = int(df["hour"].iloc[0]), int(df["min"].iloc[0]), int(df["sec"].iloc[0])
    start, from_name = _session_start(filename, (h, mi, s))
    if not from_name:
        warnings.append("ファイル名から日付を取得できず、取り込み日を使用")
    secs = (df["hour"] * 3600 + df["min"] * 60 + df["sec"]).to_numpy()
    secs = np.unwrap(secs, period=86400)
    clock = [start + timedelta(seconds=float(x - secs[0])) for x in secs]
    out.insert(0, "clock", [c.strftime("%Y-%m-%d %H:%M:%S") for c in clock])
    out.insert(0, "row_no", np.arange(len(out)))

    tmode = pd.to_numeric(df["treat-mode"], errors="coerce").dropna().astype(int)
    meta_json = {"raw_meta": [m for i, m in enumerate(meta) if i != META_NAME_IDX]}
    return ParsedSession(
        filename=filename,
        file_hash=file_hash(raw),
        patient_id=pid,
        session_dt=start,
        treat_sec=int(out["treat_sec"].max()),
        n_rows=len(out),
        device=device,
        version=version,
        header_meta=meta_json,
        ts=out,
        warnings=warnings,
        patient_name=name,
        treat_mode=int(tmode.mode().iloc[0]) if len(tmode) else None,
        mode_set=",".join(str(x) for x in sorted(tmode.unique())),
        console_id=console,
    )


def meta_to_json(meta: dict) -> str:
    return json.dumps(meta, ensure_ascii=False)

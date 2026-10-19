"""BV(循環血液量変化率 ΔBV)と血圧低下の解析ロジック。UIに依存しない純関数のみ。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from bv_parser import mode_label

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HypoCriteria:
    """血圧低下の判定基準。None の項目は使用しない。有効な項目のいずれかを満たせば「低下」。"""
    sbp_drop: float | None = 20.0  # 治療前(最初の測定)SBPからの低下量 [mmHg]
    sbp_abs: float | None = 90.0  # SBPの絶対値下限 [mmHg]
    map_drop: float | None = None  # 治療前MAPからの低下量 [mmHg]
    prev_drop: float | None = None  # 前回測定からのSBP低下量 [mmHg]

    def describe(self) -> str:
        p = []
        if self.sbp_drop is not None:
            p.append(f"開始時SBPから{self.sbp_drop:g}mmHg以上低下")
        if self.sbp_abs is not None:
            p.append(f"SBP<{self.sbp_abs:g}")
        if self.map_drop is not None:
            p.append(f"開始時MAPから{self.map_drop:g}mmHg以上低下")
        if self.prev_drop is not None:
            p.append(f"前回測定から{self.prev_drop:g}mmHg以上低下")
        return " または ".join(p) if p else "（基準なし）"


PRESETS: dict[str, HypoCriteria] = {
    "標準: SBP 20↓ または <90": HypoCriteria(20, 90),
    "厳しめ: SBP 30↓ または <90": HypoCriteria(30, 90),
    "SBP <90 のみ": HypoCriteria(None, 90),
    "SBP <100 のみ": HypoCriteria(None, 100),
    "SBP 20↓ のみ": HypoCriteria(20, None),
    "MAP 10↓ または SBP 20↓ または <90": HypoCriteria(20, 90, 10),
    "前回比 SBP 20↓ または <90": HypoCriteria(None, 90, None, 20),
}

DUR_GROUPS = ["4時間群", "5時間群", "短時間(<210分)"]


def duration_group(treat_min) -> str:
    """治療時間による群分け: 5時間群(270分以上) / 4時間群(210〜270分) / 短時間。"""
    if treat_min is None or pd.isna(treat_min):
        return "不明"
    if treat_min >= 270:
        return "5時間群"
    if treat_min >= 210:
        return "4時間群"
    return "短時間(<210分)"


def dedupe_time(ts: pd.DataFrame) -> pd.DataFrame:
    """治療時間が重複する行(装置の一時停止など)は最後の行を採用。"""
    return ts.drop_duplicates("treat_sec", keep="last").reset_index(drop=True)


def bp_raw(ts: pd.DataFrame) -> pd.DataFrame:
    """血圧測定点の抽出(判定フラグなし)。

    装置は血圧を測定値のまま保持して出力するため、sbp/dbp/pulse のいずれかが
    前行から変化した行を「測定点」とみなす(先頭行は治療前値)。
    ※同一値が連続して測定された場合は検出できない。
    """
    raw = ts[["sbp", "dbp", "pulse"]].to_numpy(float)
    raw[raw <= 0] = np.nan  # 未測定(-1)は無効
    arr = pd.DataFrame(raw).ffill().to_numpy()  # 無効行は直前の測定値を保持して変化を判定
    ok = ~np.isnan(arr).any(axis=1) & ~np.isnan(raw).any(axis=1)
    changed = np.zeros(len(arr), bool)
    changed[1:] = (arr[1:] != arr[:-1]).any(axis=1) | np.isnan(arr[:-1]).any(axis=1)
    changed[0] = True
    changed &= ok
    bp = ts[changed].reset_index(drop=True)
    n = len(bp)
    if n == 0:
        return bp.assign(seq=[], treat_min=[], map=[], d_sbp_base=[], d_sbp_prev=[], d_map_base=[])
    sbp, dbp = bp["sbp"].to_numpy(float), bp["dbp"].to_numpy(float)
    mapv = dbp + (sbp - dbp) / 3
    return bp.assign(seq=np.arange(n), treat_min=bp["treat_sec"].to_numpy(float) / 60, map=mapv,
                     d_sbp_base=sbp - sbp[0], d_map_base=mapv - mapv[0],
                     d_sbp_prev=np.r_[np.nan, np.diff(sbp)])


def hypo_flags(bp: pd.DataFrame, crit: HypoCriteria) -> pd.Series:
    """各測定点が低下基準を満たすか(治療前値は常にFalse)。"""
    flag = pd.Series(False, index=bp.index)
    if bp.empty:
        return flag
    if crit.sbp_drop is not None:
        flag |= (-bp["d_sbp_base"]) >= crit.sbp_drop
    if crit.sbp_abs is not None:
        flag |= bp["sbp"] < crit.sbp_abs
    if crit.map_drop is not None:
        flag |= (-bp["d_map_base"]) >= crit.map_drop
    if crit.prev_drop is not None:
        flag |= (-bp["d_sbp_prev"]) >= crit.prev_drop
    return flag & (bp["seq"] > 0)


def bp_points(ts: pd.DataFrame, crit: HypoCriteria) -> pd.DataFrame:
    bp = bp_raw(ts)
    bp["hypo"] = hypo_flags(bp, crit)
    return bp


def minute_grid(ts: pd.DataFrame, cols=("dbv", "ufr", "uf_vol")) -> pd.DataFrame:
    """1分刻みへ線形補間した時系列(傾き・ウィンドウ計算・症例間比較用)。"""
    d = dedupe_time(ts)
    t = d["treat_sec"].to_numpy(float) / 60
    if len(t) < 2:
        return pd.DataFrame({"min": t})
    grid = np.arange(0, np.floor(t.max()) + 1)
    out = {"min": grid}
    for c in cols:
        ok = d[c].notna().to_numpy()
        out[c] = np.interp(grid, t[ok], d[c].to_numpy()[ok]) if ok.sum() >= 2 else np.full(len(grid), np.nan)
    return pd.DataFrame(out)


def slope_per_hour(t_min: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(t_min) | np.isnan(y))
    if ok.sum() < 3 or np.ptp(t_min[ok]) < 1:
        return np.nan
    return float(np.polyfit(t_min[ok], y[ok], 1)[0] * 60)


def bv_metrics(ts: pd.DataFrame, window_min: int = 30) -> dict:
    """ΔBV の要約指標。"""
    d = dedupe_time(ts)
    t = d["treat_sec"].to_numpy(float) / 60
    y = d["dbv"].to_numpy(float)
    m: dict = {"treat_min": float(np.nanmax(t)) if len(t) else np.nan}
    if np.isnan(y).all():
        return m
    i = int(np.nanargmin(y))
    m.update(
        min_dbv=float(y[i]), min_dbv_time=float(t[i]),
        end_dbv=float(y[~np.isnan(y)][-1]),
        dbv_slope_per_h=slope_per_hour(t, y),
        uf_total=float(np.nanmax(d["uf_vol"])) if d["uf_vol"].notna().any() else np.nan,
        uf_rate_mean=float(d["ufr"].mean()) if d["ufr"].notna().any() else np.nan,
    )
    # 基準範囲(下限)を下回った時間
    dt = np.diff(d["treat_sec"].to_numpy(float), prepend=d["treat_sec"].iloc[0]) / 60
    below = (d["dbv"] < d["ref_lower"]).fillna(False).to_numpy()
    m["below_lower_min"] = float(dt[below].sum())
    m["below_lower_pct"] = m["below_lower_min"] / m["treat_min"] * 100 if m["treat_min"] else np.nan
    # 最大低下速度(window_min 分あたりの ΔBV 変化の最小値)
    g = minute_grid(ts)
    if len(g) > window_min:
        diff = g["dbv"].diff(window_min)
        j = diff.idxmin()
        m[f"max_drop_{window_min}min"] = float(diff.loc[j])
        m[f"max_drop_{window_min}min_end"] = float(g["min"].loc[j])
    return m


def bp_metrics(bp: pd.DataFrame, ts: pd.DataFrame, window_min: int = 30) -> dict:
    """血圧の要約指標と、最初の血圧低下時点のBV状況。"""
    if bp.empty:
        return {"n_bp": 0}
    after = bp[bp["seq"] > 0]
    m: dict = {
        "n_bp": int(len(bp)),
        "sbp_base": float(bp["sbp"].iloc[0]),
        "sbp_min": float(after["sbp"].min()) if len(after) else np.nan,
        "sbp_drop_max": float(-after["d_sbp_base"].min()) if len(after) else np.nan,
        "hypo_auto": bool(bp["hypo"].any()),
    }
    if len(after):
        r = after.loc[after["sbp"].idxmin()]
        m["sbp_min_time"] = float(r["treat_min"])
        m["dbv_at_sbp_min"] = float(r["dbv"]) if pd.notna(r["dbv"]) else np.nan
    h = bp[bp["hypo"]]
    if len(h):
        r = h.iloc[0]
        g = minute_grid(ts)
        tm = int(round(r["treat_min"]))
        m["first_hypo_time"] = float(r["treat_min"])
        m["dbv_at_first_hypo"] = float(r["dbv"]) if pd.notna(r["dbv"]) else np.nan
        m["uf_at_first_hypo"] = float(r["uf_vol"]) if pd.notna(r["uf_vol"]) else np.nan
        # 低下直前window_min分のΔBV傾き
        lo = max(0, tm - window_min)
        seg = g[(g["min"] >= lo) & (g["min"] <= tm)]
        m["dbv_slope_before_hypo"] = slope_per_hour(seg["min"].to_numpy(), seg["dbv"].to_numpy())
    return m


def phase_table(ts: pd.DataFrame, bp: pd.DataFrame, n_phase: int = 4) -> pd.DataFrame:
    """治療時間を等分して、段階ごとのBV・除水・血圧を集計。"""
    d = dedupe_time(ts)
    t = d["treat_sec"].to_numpy(float) / 60
    total = t.max()
    edges = np.linspace(0, total, n_phase + 1)
    rows = []
    for k in range(n_phase):
        lo, hi = edges[k], edges[k + 1]
        sel = d[(t >= lo) & (t <= hi if k == n_phase - 1 else t < hi)]
        if sel.empty:
            continue
        tt = sel["treat_sec"].to_numpy(float) / 60
        b = bp[(bp["treat_min"] >= lo) & (bp["treat_min"] <= hi if k == n_phase - 1 else bp["treat_min"] < hi)
               & (bp["seq"] > 0)] if len(bp) else bp
        rows.append({
            "段階": f"{k + 1}/{n_phase}",
            "開始(分)": round(lo), "終了(分)": round(hi),
            "ΔBV開始(%)": sel["dbv"].iloc[0], "ΔBV終了(%)": sel["dbv"].iloc[-1],
            "ΔBV最小(%)": sel["dbv"].min(),
            "ΔBV傾き(%/h)": slope_per_hour(tt, sel["dbv"].to_numpy(float)),
            "平均除水速度(L/h)": sel["ufr"].mean(),
            "除水量増加(L)": sel["uf_vol"].iloc[-1] - sel["uf_vol"].iloc[0],
            "血圧測定数": len(b),
            "最低SBP": b["sbp"].min() if len(b) else np.nan,
            "低下あり": bool(b["hypo"].any()) if len(b) else False,
        })
    return pd.DataFrame(rows)


def bp_context(bp: pd.DataFrame, ts: pd.DataFrame, window_min: int = 30) -> pd.DataFrame:
    """各血圧測定点での、BV・除水の状況(どの時点でBVがどうなっていたか)。"""
    if bp.empty:
        return bp
    g = minute_grid(ts)
    rows = []
    for _, r in bp.iterrows():
        tm = int(round(r["treat_min"]))
        prev = g.loc[g["min"] == max(0, tm - window_min), "dbv"]
        now = g.loc[g["min"] == min(tm, g["min"].max()), "dbv"]
        rows.append({
            "測定": int(r["seq"]),
            "経過(分)": round(r["treat_min"]),
            "時刻": str(r["clock"])[11:16],
            "SBP": r["sbp"], "DBP": r["dbp"], "脈拍": r["pulse"], "MAP": round(r["map"], 1),
            "ΔSBP(開始比)": r["d_sbp_base"], "ΔSBP(前回比)": r["d_sbp_prev"],
            "ΔBV(%)": r["dbv"],
            f"ΔBV変化/{window_min}分(%)": (now.iloc[0] - prev.iloc[0]) if len(prev) and len(now) else np.nan,
            "除水量(L)": r["uf_vol"], "除水速度(L/h)": r["ufr"],
            "血圧低下": bool(r["hypo"]),
        })
    return pd.DataFrame(rows)


def session_summary(sessions: pd.DataFrame, ts_all: pd.DataFrame, crit: HypoCriteria,
                    window_min: int = 30) -> pd.DataFrame:
    """全症例の要約テーブル(症例比較・統計用)。"""
    rows = []
    groups = {sid: g for sid, g in ts_all.groupby("session_id")}
    for s in sessions.itertuples(index=False):
        g = groups.get(s.session_id)
        if g is None:
            continue
        bp = bp_points(g, crit)
        row = {
            "session_id": s.session_id, "patient_id": s.patient_id, "session_dt": s.session_dt,
            **bv_metrics(g, window_min), **bp_metrics(bp, g, window_min),
        }
        row["dur_group"] = duration_group(row.get("treat_min"))
        row["treat_mode"] = s.treat_mode if pd.notna(s.treat_mode) else np.nan
        row["mode_label"] = mode_label(s.treat_mode)
        ec = event_counts(g)
        row["n_init"] = ec.get(1, 0)
        row["n_uf_stop"] = ec.get(2, 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    clin = sessions[["session_id", "hypo_event", "symptom", "intervention", "memo"]]
    out = out.merge(clin, on="session_id", how="left")
    # 臨床記録(手入力)があればそれを優先、無ければ自動判定
    out["hypo_final"] = np.where(out["hypo_event"].notna(), out["hypo_event"] == 1, out["hypo_auto"])
    return out


def curves_matrix(ts_all: pd.DataFrame, session_ids, col: str = "dbv", normalize: bool = False,
                  step: float = 2.0, n_points: int = 101):
    """症例ごとの曲線を共通の時間軸へ補間した行列を返す: (x, matrix[症例×x], 症例ID一覧)。

    normalize=True で治療時間を0-100%へ正規化。非正規化では各症例の終了後はNaN。
    """
    ids = set(int(i) for i in session_ids)
    groups = {int(k): g for k, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id")}
    prepared = {}
    t_max = 0.0
    for sid, g in groups.items():
        g = dedupe_time(g)
        t = g["treat_sec"].to_numpy(float) / 60
        y = g[col].to_numpy(float)
        ok = ~np.isnan(y)
        if ok.sum() < 2 or t.max() <= 0:
            continue
        prepared[sid] = (t, y, ok)
        t_max = max(t_max, t.max())
    x = np.linspace(0, 100, n_points) if normalize else np.arange(0, t_max + step, step)
    mat = np.full((len(prepared), len(x)), np.nan)
    for i, (t, y, ok) in enumerate(prepared.values()):
        xs = t / t.max() * 100 if normalize else t
        yy = np.interp(x, xs[ok], y[ok])
        if not normalize:
            yy[x > t.max()] = np.nan
        mat[i] = yy
    return x, mat, list(prepared.keys())


def summarize_curves(x: np.ndarray, mat: np.ndarray) -> pd.DataFrame:
    """曲線行列の統計(平均・SD・中央値・四分位・症例数)。"""
    if mat.shape[0] == 0:
        return pd.DataFrame(columns=["x", "mean", "sd", "median", "q25", "q75", "n"])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.DataFrame({
            "x": x, "mean": np.nanmean(mat, 0), "sd": np.nanstd(mat, 0), "median": np.nanmedian(mat, 0),
            "q25": np.nanpercentile(mat, 25, axis=0), "q75": np.nanpercentile(mat, 75, axis=0),
            "n": np.sum(~np.isnan(mat), 0)})


def mean_curve(ts_all: pd.DataFrame, session_ids: list[int], col: str = "dbv",
               normalize: bool = False, n_points: int = 100) -> pd.DataFrame:
    """症例群の平均±SD曲線。normalize=True で治療時間を0-100%に正規化。"""
    x, mat, _ = curves_matrix(ts_all, session_ids, col, normalize, n_points=n_points)
    return summarize_curves(x, mat)


def all_bp_points(ts_all: pd.DataFrame, session_ids, crit: HypoCriteria) -> pd.DataFrame:
    """複数症例の血圧測定点をまとめる(session_id, t_total[分]付き)。"""
    ids = set(int(i) for i in session_ids)
    frames = []
    for sid, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id"):
        bp = bp_points(g, crit)
        if len(bp):
            frames.append(bp.assign(session_id=int(sid), t_total=g["treat_sec"].max() / 60))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bp_bin_stats(bp_all: pd.DataFrame, normalize: bool, bin_size: float) -> pd.DataFrame:
    """治療時間帯ごとのSBP平均と血圧低下の頻度(治療前値は除く)。"""
    if bp_all.empty:
        return pd.DataFrame(columns=["x", "n", "sbp_mean", "hypo_n", "hypo_pct"])
    d = bp_all[bp_all["seq"] > 0].copy()
    d["x"] = d["treat_min"] / d["t_total"] * 100 if normalize else d["treat_min"]
    d["bin"] = (d["x"] // bin_size) * bin_size
    g = d.groupby("bin").agg(n=("sbp", "size"), sbp_mean=("sbp", "mean"), hypo_n=("hypo", "sum")).reset_index()
    g["x"] = g["bin"] + bin_size / 2
    g["hypo_pct"] = g["hypo_n"] / g["n"] * 100
    return g


def event_counts(ts: pd.DataFrame) -> dict[int, int]:
    """イベントコードごとの発生回数(複数同時発生は分解して数える)。"""
    from bv_parser import decode_event
    out: dict[int, int] = {}
    ev = ts["event"].dropna()
    ev = ev[ev != 0]
    for v, n in ev.value_counts().items():
        for c in decode_event(v):
            out[c] = out.get(c, 0) + int(n)
    return out


def event_table(ts: pd.DataFrame) -> pd.DataFrame:
    """治療中のイベント一覧(時刻・内容・その時点のΔBVと除水量)。"""
    from bv_parser import decode_event, event_text
    d = ts[ts["event"].fillna(0) != 0]
    rows = []
    for r in d.itertuples():
        rows.append({"経過(分)": round(r.treat_sec / 60, 1), "時刻": str(r.clock)[11:19],
                     "イベント": event_text(r.event), "コード": "+".join(str(c) for c in decode_event(r.event)),
                     "ΔBV(%)": r.dbv, "除水量(L)": r.uf_vol, "除水速度(L/h)": r.ufr})
    return pd.DataFrame(rows)


def bp_features(ts: pd.DataFrame, bp: pd.DataFrame, window_min: int = 30) -> pd.DataFrame:
    """各血圧測定点(治療前値を除く)の直前の状況。血圧低下の予測因子の検討用。"""
    cols = ["seq", "treat_min", "sbp", "hypo", "dbv", "dbv_avg5", "dbv_change", "dbv_slope", "dbv_margin",
            "uf_vol", "ufr"]
    pts = bp[bp["seq"] > 0]
    if pts.empty:
        return pd.DataFrame(columns=cols)
    g = minute_grid(ts)
    rows = []
    for r in pts.itertuples():
        tm = float(r.treat_min)
        change = slope = np.nan
        if tm >= window_min and len(g) > 2:
            seg = g[(g["min"] >= tm - window_min) & (g["min"] <= tm)]
            slope = slope_per_hour(seg["min"].to_numpy(), seg["dbv"].to_numpy())
            change = r.dbv - float(np.interp(tm - window_min, g["min"], g["dbv"]))
        rows.append({"seq": r.seq, "treat_min": tm, "sbp": r.sbp, "hypo": bool(r.hypo), "dbv": r.dbv,
                     "dbv_avg5": r.dbv_avg5, "dbv_change": change, "dbv_slope": slope,
                     "dbv_margin": r.dbv - r.ref_lower, "uf_vol": r.uf_vol, "ufr": r.ufr})
    return pd.DataFrame(rows, columns=cols)


def all_features(ts_all: pd.DataFrame, crit: HypoCriteria, window_min: int = 30) -> pd.DataFrame:
    """全症例の血圧測定点の特徴量(session_id付き)。"""
    frames = []
    for sid, g in ts_all.groupby("session_id"):
        bp = bp_points(g, crit)
        f = bp_features(g, bp, window_min)
        if len(f):
            frames.append(f.assign(session_id=int(sid)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def roc_analysis(y: np.ndarray, x: np.ndarray) -> dict | None:
    """x が高い(または低い)ほど y=True になりやすいかを ROC/AUC で評価。陽性・陰性が各3件未満なら None。"""
    from scipy.stats import rankdata
    ok = ~np.isnan(x)
    y = np.asarray(y, bool)[ok]
    x = np.asarray(x, float)[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 3 or n0 < 3:
        return None
    ranks = rankdata(x)
    auc_hi = (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    direction = ">=" if auc_hi >= 0.5 else "<="
    score = x if direction == ">=" else -x
    order = np.argsort(-score, kind="stable")
    ys, sc = y[order], score[order]
    tps, fps = np.cumsum(ys), np.cumsum(~ys)
    idx = np.r_[np.where(np.diff(sc))[0], len(sc) - 1]
    tpr, fpr, thr = np.r_[0, tps[idx] / n1], np.r_[0, fps[idx] / n0], np.r_[np.inf, sc[idx]]
    k = int(np.argmax(tpr - fpr))
    cutoff = thr[k] if direction == ">=" else -thr[k]
    return {"auc": float(max(auc_hi, 1 - auc_hi)), "direction": direction, "cutoff": float(cutoff),
            "sens": float(tpr[k]), "spec": float(1 - fpr[k]), "fpr": fpr, "tpr": tpr, "n_pos": n1, "n_neg": n0}


def auc_cluster_ci(df: pd.DataFrame, feature: str, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """症例(session)単位のブートストラップによるAUCの95%区間(同一症例の測定点の相関を考慮)。"""
    rng = np.random.default_rng(seed)
    d = df[["session_id", feature, "hypo"]].dropna(subset=[feature])
    ys, xs = [], []
    for _, g in d.groupby("session_id"):
        ys.append(g["hypo"].to_numpy(bool))
        xs.append(g[feature].to_numpy(float))
    if len(ys) < 5:
        return (np.nan, np.nan)
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ys), len(ys))
        r = roc_analysis(np.concatenate([ys[i] for i in pick]), np.concatenate([xs[i] for i in pick]))
        if r:
            aucs.append(r["auc"])
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if len(aucs) > 20 else (np.nan, np.nan)


def aligned_traces(ts_all: pd.DataFrame, session_ids, crit: HypoCriteria, pre: int = 60, col: str = "dbv"):
    """血圧測定時点を0として、直前 pre 分の指標推移を、低下あり/なしの測定点で集める。

    戻り値: (低下ありの配列[n×(pre+1)], 低下なしの配列, 時間オフセット[-pre..0])
    """
    ids = set(int(i) for i in session_ids)
    offs = np.arange(-pre, 1)
    hyp, non = [], []
    for sid, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id"):
        bp = bp_points(g, crit)
        grid = minute_grid(g, cols=(col,))
        if bp.empty or len(grid) < 3:
            continue
        for r in bp[bp["seq"] > 0].itertuples():
            if r.treat_min < pre:
                continue
            y = np.interp(r.treat_min + offs, grid["min"], grid[col])
            (hyp if r.hypo else non).append(y)
    return np.array(hyp), np.array(non), offs


def hypo_by_preset(ts_all: pd.DataFrame, summary: pd.DataFrame, presets: dict[str, HypoCriteria]) -> pd.DataFrame:
    """判定基準ごとの血圧低下率(症例単位・測定点単位)。summary は session_id, dur_group を持つこと。"""
    groups = {int(k): g for k, g in ts_all.groupby("session_id")}
    raw = {sid: bp_raw(g) for sid, g in groups.items()}
    dur = summary.set_index("session_id")["dur_group"].to_dict()
    rows = []
    for name, crit in presets.items():
        sess_flag, pts_n, pts_h = {}, 0, 0
        for sid, bp in raw.items():
            if sid not in dur or bp.empty:
                continue
            f = hypo_flags(bp, crit)
            sess_flag[sid] = bool(f.any())
            pts_n += int((bp["seq"] > 0).sum())
            pts_h += int(f.sum())
        sf = pd.Series(sess_flag)
        row = {"判定基準": name, "内容": crit.describe(), "症例数": len(sf), "低下あり症例": int(sf.sum()),
               "症例の低下率(%)": sf.mean() * 100 if len(sf) else np.nan,
               "測定点あたり低下率(%)": pts_h / pts_n * 100 if pts_n else np.nan}
        for gname in DUR_GROUPS:
            ids = [sid for sid in sf.index if dur.get(sid) == gname]
            row[f"{gname} 症例低下率(%)"] = sf.loc[ids].mean() * 100 if ids else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def preset_flags(ts_all: pd.DataFrame, presets: dict[str, HypoCriteria]) -> pd.DataFrame:
    """症例ごと×判定基準ごとの『低下あり』フラグ(index=session_id)。"""
    rows = {}
    for sid, g in ts_all.groupby("session_id"):
        bp = bp_raw(g)
        if bp.empty:
            continue
        rows[int(sid)] = {name: bool(hypo_flags(bp, crit).any()) for name, crit in presets.items()}
    return pd.DataFrame.from_dict(rows, orient="index")

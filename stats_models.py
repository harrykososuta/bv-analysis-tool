"""患者の違いを考慮した解析(患者内比較・調整モデル)と、除水量の目安(概算)。UIに依存しない純関数。

いずれも観察データの探索的解析であり、因果や臨床上の指示を示すものではない。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def paired_by_patient(summ: pd.DataFrame, flag_col: str = "hypo_final", group_col: str = "dur_group",
                      a: str = "4時間群", b: str = "5時間群", min_each: int = 3) -> pd.DataFrame:
    """両群を各 min_each 件以上持つ患者について、患者ごとの血圧低下率(%)を並べる。"""
    rows = []
    for pid, g in summ.groupby("patient_id"):
        ga, gb = g[g[group_col] == a], g[g[group_col] == b]
        if len(ga) >= min_each and len(gb) >= min_each:
            rows.append({"patient_id": pid, f"{a} 症例数": len(ga), f"{a} 低下数": int(ga[flag_col].sum()),
                         f"{a} 低下率(%)": ga[flag_col].mean() * 100,
                         f"{b} 症例数": len(gb), f"{b} 低下数": int(gb[flag_col].sum()),
                         f"{b} 低下率(%)": gb[flag_col].mean() * 100})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["差(b−a, %pt)"] = d[f"{b} 低下率(%)"] - d[f"{a} 低下率(%)"]
    return d


def mantel_haenszel(summ: pd.DataFrame, flag_col: str = "hypo_final", group_col: str = "dur_group",
                    a: str = "4時間群", b: str = "5時間群") -> dict | None:
    """患者で層別したMantel-Haenszel共通オッズ比(b群 vs a群)。患者の違いによる混ざりを除く。"""
    from statsmodels.stats.contingency_tables import StratifiedTable
    tables = []
    for _, g in summ.groupby("patient_id"):
        ga, gb = g[g[group_col] == a][flag_col].astype(bool), g[g[group_col] == b][flag_col].astype(bool)
        if len(ga) and len(gb):
            tables.append(np.array([[gb.sum(), (~gb).sum()], [ga.sum(), (~ga).sum()]], float) + 0.0)
    tables = [t for t in tables if t.sum(axis=1).min() > 0]
    if len(tables) < 2:
        return None
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    return {"or": float(st.oddsratio_pooled), "lo": float(lo), "hi": float(hi), "n_strata": len(tables),
            "p": float(st.test_null_odds().pvalue), "n_sessions": int(sum(t.sum() for t in tables))}


def gee_logit(summ: pd.DataFrame, outcome: str, predictors: list[str], categorical: list[str],
              cluster: str = "patient_id") -> pd.DataFrame:
    """患者をクラスターとするGEEロジスティック回帰(交換可能相関)。オッズ比と95%区間を返す。"""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = summ.dropna(subset=[outcome, *predictors, *categorical, cluster]).copy()
    d[outcome] = d[outcome].astype(float)
    for c in predictors:  # 係数を解釈しやすい単位に(標準化はしない)
        d[c] = d[c].astype(float)
    terms = [*predictors, *[f"C({c})" for c in categorical]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.gee(f"{outcome} ~ " + " + ".join(terms), groups=cluster, data=d,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    out = pd.DataFrame({"項目": m.params.index, "OR": np.exp(m.params.values), "下限": np.exp(ci[0].values),
                        "上限": np.exp(ci[1].values), "p値": m.pvalues.values})
    out.attrs["n"] = int(len(d))
    out.attrs["n_clusters"] = int(d[cluster].nunique())
    return out[out["項目"] != "Intercept"].reset_index(drop=True)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict | None:
    """単変量ロジスティック回帰 p = sigmoid(a + b·x)。陽性・陰性が各5件未満、または完全分離なら None。"""
    import statsmodels.api as sm
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok].astype(float), y[ok].astype(float)
    if len(y) < 15 or y.sum() < 5 or (1 - y).sum() < 5 or np.ptp(x) == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    except Exception:  # noqa: BLE001  (完全分離・非収束)
        return None
    return {"a": float(m.params[0]), "b": float(m.params[1]), "p_slope": float(m.pvalues[1]), "n": int(len(y)),
            "n_pos": int(y.sum())}


def guideline(fit: dict | None, target: float) -> float:
    """血圧低下の確率が target になる x。傾きが正でなければ NaN。"""
    if not fit or fit["b"] <= 0:
        return np.nan
    return (_logit(target) - fit["a"]) / fit["b"]


def guideline_ci(df: pd.DataFrame, xcol: str, ycol: str, target: float, cluster: str,
                 n_boot: int = 300, seed: int = 0) -> tuple[float, float, int]:
    """クラスター(患者/症例)単位のブートストラップによる目安の95%区間。(下限, 上限, 有効回数)"""
    rng = np.random.default_rng(seed)
    groups = {k: g[[xcol, ycol]].to_numpy(float) for k, g in df.groupby(cluster)}
    keys = list(groups)
    if len(keys) < 5:
        return (np.nan, np.nan, 0)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        arr = np.vstack([groups[keys[i]] for i in pick])
        v = guideline(fit_logistic(arr[:, 0], arr[:, 1]), target)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 30:
        return (np.nan, np.nan, len(vals))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals))


def uf_stop_after_hypo(ts_all: pd.DataFrame, session_ids, crit, window: int = 30) -> pd.DataFrame:
    """血圧測定点ごとに、測定後 window 分以内に『除水停止』イベントがあったか。"""
    from analysis import bp_points
    from bv_parser import decode_event
    ids = set(int(i) for i in session_ids)
    rows = []
    for sid, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id"):
        bp = bp_points(g, crit)
        if bp.empty:
            continue
        ev = g[g["event"].fillna(0) != 0]
        stop_t = np.array([r.treat_sec / 60 for r in ev.itertuples() if 2 in decode_event(r.event)])
        for r in bp[bp["seq"] > 0].itertuples():
            hit = bool(len(stop_t) and ((stop_t > r.treat_min) & (stop_t <= r.treat_min + window)).any())
            rows.append({"session_id": int(sid), "treat_min": r.treat_min, "hypo": bool(r.hypo), "uf_stop": hit})
    return pd.DataFrame(rows)

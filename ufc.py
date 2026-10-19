"""BV-UFC(ΔBVを見て除水速度を自動調節する機能)の設定・予測シミュレーション・実績集計。UIに依存しない純関数のみ。

装置の制御は公開されていないため、ここでの予測は次の簡易モデルによる概算である。
- ΔBVの変化(%/h) = b0 + b1·除水速度 + b2·ΔBV + b3·経過時間(h)   … 患者の過去データから推定
  (b1 = −100/循環血液量, b2・b0・b3 は refilling(PRR) の大きさを表す)
- UFC中は2分ごとに、ΔBVの変化率が目標カーブに沿うよう除水速度を更新し、上限・下限で制限する
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from analysis import bp_points, dedupe_time, minute_grid
from bv_parser import decode_event

# CSVのstep no。3〜6は治療時間を4等分した区間で、終了時に目標ΔBVの 指数1/2/3/100% に達するよう制御される
# (取り込み済み症例の実績から推定)
STEP_LABELS = {0: "停止/その他", 1: "開始", 2: "ステップ2", 3: "区間1（→指数1）", 4: "区間2（→指数2）",
               5: "区間3（→指数3）", 6: "区間4（→終了時ΔBV）", 7: "最終"}
STEP_COLORS = {1: "rgba(80,190,230,.13)", 2: "rgba(150,150,150,.10)", 3: "rgba(230,90,200,.10)",
               4: "rgba(90,180,90,.10)", 5: "rgba(90,180,90,.16)", 6: "rgba(90,180,90,.22)",
               7: "rgba(240,150,60,.15)", 0: "rgba(120,120,120,.08)"}
TARGET_LEAD, TARGET_TAIL = 5.0, 5.0  # 実績準拠の配置で、目標カーブの前後に置く時間(分)
PLACEMENTS = {"whole": "治療全体を4等分（取り込み実績から推定）", "ufc": "UFC期間を4等分（装置画面の表示どおり）"}
CONTROL_EVERY = 2  # 除水速度の更新間隔(分)
CATCH_UP = 10.0  # 目標からのずれを何分かけて戻すか
BASELINE_INIT_MAX = 15.0  # この時刻(分)までのΔBV初期化は、開始時の基準取り(通常5分ごろ)とみなす


# ----------------------------------------------------------------- 設定
@dataclass
class UFCSettings:
    total_min: float = 240.0  # 治療時間
    target_uf: float = 2.0  # 目標除水量 (L)
    start_min: float = 10.0  # 開始
    start_mult: float = 1.00
    const_min: float = 60.0  # 定倍除水
    const_mult: float = 1.30
    end_min: float = 20.0  # 終了前
    ufr_max: float = 2.00  # UFC期間の除水速度 上限 (L/h)
    ufr_min: float = 0.00  # 下限 (L/h)
    start_dbv: float = 0.0  # 開始時ΔBV (%)
    idx1: float = 50.0  # 指数(終了時ΔBVに対する%)
    idx2: float = 80.0
    idx3: float = 95.0
    end_dbv: float = -4.0  # 終了時ΔBV (%)
    placement: str = "whole"

    @property
    def mean_rate(self) -> float:
        return self.target_uf / (self.total_min / 60) if self.total_min > 0 else 0.0

    @property
    def ufc_min(self) -> float:
        return max(0.0, self.total_min - self.start_min - self.const_min - self.end_min)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "UFCSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, s: str) -> "UFCSettings":
        return cls.from_dict(json.loads(s))


def planned_from_meta(header_json: str) -> dict:
    """CSV 1行目の設定値(治療時間・目標除水量・開始/定倍/終了前)。読めない項目は含めない。"""
    try:
        m = json.loads(header_json)["raw_meta"]
    except Exception:  # noqa: BLE001
        return {}

    def num(i, div=1.0):
        try:
            v = float(m[i])
        except (IndexError, ValueError, TypeError):
            return None
        return None if v < 0 else v / div

    keys = {"total_min": (15, 1), "target_uf": (16, 100), "start_mult": (17, 100), "start_min": (18, 1),
            "const_mult": (19, 100), "const_min": (20, 1), "end_min": (21, 1)}
    out = {k: num(i, d) for k, (i, d) in keys.items()}
    out = {k: v for k, v in out.items() if v is not None}
    if not out.get("total_min"):
        out.pop("total_min", None)
    if not out.get("target_uf"):
        out.pop("target_uf", None)
    return out


# ----------------------------------------------------------------- 目標カーブ
def target_points(s: UFCSettings) -> tuple[np.ndarray, np.ndarray]:
    """目標ΔBVの5点(開始・指数1・指数2・指数3・終了)の (時刻[分], ΔBV[%])。"""
    if s.placement == "ufc":
        a, b = s.start_min + s.const_min, s.total_min - s.end_min
    else:
        a, b = TARGET_LEAD, s.total_min - TARGET_TAIL
    b = max(b, a + 4)
    t = a + (b - a) * np.array([0, .25, .5, .75, 1])
    frac = np.array([0, s.idx1, s.idx2, s.idx3, 100]) / 100
    return t, s.start_dbv + (s.end_dbv - s.start_dbv) * frac


def target_curve(s: UFCSettings, t: np.ndarray) -> np.ndarray:
    """目標ΔBV(5点を単調な滑らかな曲線でつなぐ)。前後は端の値で一定。"""
    from scipy.interpolate import PchipInterpolator
    tp, vp = target_points(s)
    y = PchipInterpolator(tp, vp)(np.clip(t, tp[0], tp[-1]))
    return np.asarray(y, float)


def phase_at(s: UFCSettings, t: float) -> str:
    if t < s.start_min:
        return "開始"
    if t < s.start_min + s.const_min:
        return "定倍除水"
    if t < s.total_min - s.end_min:
        return "UFC期間"
    return "終了前"


# ----------------------------------------------------------------- ΔBV応答モデル
# ΔBVの変化(%/h) = −100/BV × (除水速度 − PRR)、PRR = p0 + p1·ΔBV + p2·経過時間(h)
#   BV : 装置がPRRの計算に使っている循環血液量(記録された PRR・除水速度・ΔBVの傾きから逆算)
#   PRR: 装置が記録したPRRを、ΔBVと経過時間で回帰
BV_RANGE = (2.5, 8.0)


@dataclass
class ResponseModel:
    bv: float  # 循環血液量 (L)
    prr_coef: np.ndarray  # p0, p1(ΔBV), p2(時間h)
    n_sessions: int
    rmse: float  # 過去症例のΔBV経過の再現誤差(%)
    source: str

    @property
    def coef(self) -> np.ndarray:
        """ΔBVの変化(%/h) = b0 + b1·除水速度 + b2·ΔBV + b3·時間h の係数"""
        k = 100 / self.bv
        p0, p1, p2 = self.prr_coef
        return np.array([k * p0, -k, k * p1, k * p2])

    def slope(self, ufr, dbv, t_h):
        return -100 / self.bv * (ufr - self.prr(dbv, t_h))

    @property
    def bv_l(self) -> float:
        return self.bv

    def prr(self, dbv, t_h):
        p0, p1, p2 = self.prr_coef
        return p0 + p1 * np.asarray(dbv) + p2 * np.asarray(t_h)

    def ufr_for(self, slope, dbv, t_h):
        """ΔBVの変化が slope(%/h) になる除水速度"""
        return self.prr(dbv, t_h) - slope * self.bv / 100

    def shifted(self, d_prr: float) -> "ResponseModel":
        c = self.prr_coef.copy()
        c[0] += d_prr
        return ResponseModel(self.bv, c, self.n_sessions, self.rmse, self.source)


def trajectories(ts_all: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """症例ごとの1分刻みの (除水速度, ΔBV 5分平均, PRR)。
    開始直後(通常5分ごろ)のΔBV初期化を起点とし、治療途中の再初期化以降は使わない。"""
    out = {}
    for sid, g in ts_all.groupby("session_id"):
        d = dedupe_time(g)
        if len(d) < 30:
            continue
        grid = minute_grid(d, cols=("dbv_avg5", "ufr", "prr"))
        if "dbv_avg5" not in grid or grid["dbv_avg5"].notna().sum() < 60:
            continue
        init_t = sorted(r.treat_sec / 60 for r in d[d["event"].fillna(0) != 0].itertuples()
                        if 1 in decode_event(r.event))
        a = int(np.ceil(init_t[0])) if init_t and init_t[0] <= BASELINE_INIT_MAX else 0
        later = [t for t in init_t if t > BASELINE_INIT_MAX]
        b = int(later[0]) if later else len(grid)
        if b - a < 60:
            continue
        y = grid["dbv_avg5"].ffill().fillna(0).to_numpy(float)[a:b]
        u = grid["ufr"].fillna(0).to_numpy(float)[a:b]
        p = grid["prr"].to_numpy(float)[a:b].copy()
        # 装置のPRRは開始直後は出ない。補間で埋まった区間や外れ値は使わない
        raw_t = d.loc[d["prr"].notna(), "treat_sec"].to_numpy(float) / 60
        first = raw_t.min() if len(raw_t) else np.inf
        p[(np.arange(a, b) < first + 2) | (np.abs(p) > 3)] = np.nan
        out[int(sid)] = (u, y - y[0], p)
    return out


def _implied_bv(u: np.ndarray, y: np.ndarray, p: np.ndarray, w: int = 20) -> float:
    """装置がPRRの計算に使った循環血液量: BV = 100·(PRR − 除水速度) / ΔBVの傾き"""
    if len(y) <= w + 10:
        return np.nan
    sl = (y[w:] - y[:-w]) * 60 / w
    um = pd.Series(u).rolling(w).mean().to_numpy()[w:]
    pm = pd.Series(p).rolling(w).mean().to_numpy()[w:]
    ok = (np.abs(sl) > 2) & ~np.isnan(pm) & ~np.isnan(um)
    if ok.sum() < 5:
        return np.nan
    v = float(np.median(100 * (pm[ok] - um[ok]) / sl[ok]))
    return v if BV_RANGE[0] <= v <= BV_RANGE[1] + 1 else np.nan


def _prr_points(trajs: dict, sids) -> np.ndarray:
    rows = []
    for s in sids:
        u, y, p = trajs[s]
        i = np.arange(0, len(y), 2)
        ok = ~np.isnan(p[i])
        rows.append(np.c_[y[i][ok], i[ok] / 60, p[i][ok]])
    return np.vstack(rows) if rows else np.zeros((0, 3))


def _fit_prr(pts: np.ndarray) -> np.ndarray | None:
    """PRR = p0 + p1·ΔBV + p2·時間h。ΔBVが下がるとPRRが減る(p1>0)推定は不安定になるため p1=0 とする。"""
    if len(pts) < 30 or np.ptp(pts[:, 0]) < 2:
        return None
    X = np.c_[np.ones(len(pts)), pts[:, :2]]
    coef, *_ = np.linalg.lstsq(X, pts[:, 2], rcond=None)
    if coef[1] > 0:
        c, *_ = np.linalg.lstsq(X[:, [0, 2]], pts[:, 2], rcond=None)
        coef = np.array([c[0], 0.0, c[1]])
    return coef


def _calibrate_level(trajs: dict, sids, bv: float, prr: np.ndarray) -> np.ndarray:
    """PRRの水準(p0)を、各症例の除水量の収支(実際の除水量 − BV×ΔBVの低下)に合うよう合わせる(中央値)。"""
    vals = []
    for s in sids:
        u, y, _ = trajs[s]
        n = len(y) - 1
        if n < 60:
            continue
        t = np.arange(n) / 60
        rest = float(np.sum(u[:n]) / 60 - bv * (y[0] - y[-1]) / 100 - np.sum(prr[1] * y[:n] + prr[2] * t) / 60)
        vals.append(rest / (n / 60))
    c = prr.copy()
    if vals:
        c[0] = float(np.median(vals))
    return c


def _replay_mat(coef: np.ndarray, U: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3 = coef
    D = np.zeros_like(U)
    for i in range(U.shape[1] - 1):
        D[:, i + 1] = np.minimum(D[:, i] + (b0 + b1 * U[:, i] + b2 * D[:, i] + b3 * i / 60) / 60, 0.0)
    return D


def _rmse(model: ResponseModel, trajs: dict, sids) -> float:
    errs = [np.mean((replay(model, trajs[s][0]) - trajs[s][1]) ** 2) for s in sids]
    return float(np.sqrt(np.mean(errs))) if errs else np.nan


def fit_pooled(trajs: dict) -> ResponseModel | None:
    """全患者のデータでのモデル(患者のデータが少ないときの補完に使う)。"""
    sids = list(trajs)
    bvs = [v for v in (_implied_bv(*trajs[s]) for s in sids) if not np.isnan(v)]
    pc = _fit_prr(_prr_points(trajs, sids))
    if pc is None:
        return None
    bv = float(np.median(bvs)) if bvs else 4.5
    m = ResponseModel(bv, _calibrate_level(trajs, sids, bv, pc), len(sids), np.nan, "全患者のデータ")
    m.rmse = _rmse(m, trajs, sids[::5])
    return m


def fit_response(trajs: dict, patient_sids, pooled: ResponseModel, k: float = 5.0) -> ResponseModel:
    """患者の過去症例から BV と PRR の傾向を推定。症例数 n の患者は 重み n/(n+k) で患者自身の値を使い、
    残りを全患者の値で補う(少数例での不安定さを抑える)。"""
    sids = [int(x) for x in patient_sids if int(x) in trajs]
    if len(sids) < 2:
        return pooled
    bvs = [v for v in (_implied_bv(*trajs[s]) for s in sids) if not np.isnan(v)]
    pc = _fit_prr(_prr_points(trajs, sids))
    w = len(sids) / (len(sids) + k)
    bv = w * float(np.median(bvs)) + (1 - w) * pooled.bv if len(bvs) >= 2 else pooled.bv
    bv = float(np.clip(bv, *BV_RANGE))
    if pc is not None:
        pc = _calibrate_level(trajs, sids, bv, pc)
    prr = w * pc + (1 - w) * pooled.prr_coef if pc is not None else pooled.prr_coef
    m = ResponseModel(bv, prr, len(sids), np.nan,
                      f"この患者 {w * 100:.0f}% ＋ 全患者 {100 - w * 100:.0f}%")
    m.rmse = _rmse(m, trajs, sids)
    return m


def day_models(trajs: dict, model: ResponseModel, patient_sids, n_max: int = 40,
               seed: int = 0) -> list[ResponseModel]:
    """日による違いを表すモデル群。過去の各症例の除水量の収支から求めたPRRの水準(refillingの多い/少ない日)に置き換えたもの。
    患者の症例が3件未満なら全患者の症例から抜き出す。"""
    sids = [int(x) for x in patient_sids if int(x) in trajs]
    if len(sids) < 3:
        rng = np.random.default_rng(seed)
        allk = list(trajs)
        sids = [int(x) for x in rng.choice(allk, min(n_max, len(allk)), replace=False)]
    out = []
    for s in sids[-n_max:]:
        p0 = _calibrate_level(trajs, [s], model.bv, model.prr_coef)[0]
        out.append(model.shifted(p0 - model.prr_coef[0]))
    return out


# ----------------------------------------------------------------- シミュレーション
def simulate(s: UFCSettings, model: ResponseModel, mode: str = "ufc") -> pd.DataFrame:
    """1分刻みで治療をシミュレーション。mode='ufc' はBV-UFC、'const' は目標除水量を一定速度で除水。"""
    n = int(round(s.total_min))
    t = np.arange(n + 1, dtype=float)
    tgt = target_curve(s, t)
    tgt_ahead = target_curve(s, t + CATCH_UP)
    dbv = np.zeros(n + 1)
    ufr = np.zeros(n + 1)
    uf = np.zeros(n + 1)
    phase = []
    cur = 0.0
    for i in range(n + 1):
        ph = phase_at(s, t[i])
        phase.append(ph)
        t_h = t[i] / 60
        if mode == "const":
            cur = s.mean_rate
        elif ph == "開始":
            cur = s.mean_rate * s.start_mult
        elif ph == "定倍除水":
            cur = s.mean_rate * s.const_mult
        elif i % CONTROL_EVERY == 0 or phase[i - 1] in ("開始", "定倍除水"):
            # 目標カーブの傾き + ずれを CATCH_UP 分で戻す分 を実現する除水速度
            need = (tgt_ahead[i] - dbv[i]) * 60 / CATCH_UP
            cur = float(np.clip(model.ufr_for(need, dbv[i], t_h), s.ufr_min, s.ufr_max))
        ufr[i] = cur
        if i < n:
            sl = model.slope(cur, dbv[i], t_h)
            if dbv[i] >= 0:  # refillingで循環血液量が開始時より増えることはない(PRRは除水量を超えない)
                sl = min(sl, 0.0)
            dbv[i + 1] = min(dbv[i] + sl / 60, 0.0) if dbv[i] <= 0 else dbv[i] + sl / 60
            uf[i + 1] = uf[i] + cur / 60
    prr = model.prr(dbv, t / 60)
    prr = np.where(dbv >= 0, np.minimum(prr, ufr), prr)  # ΔBVが0%の間は、除水量以上には戻らない
    return pd.DataFrame({"min": t, "dbv": dbv, "target": tgt, "ufr": ufr, "uf_vol": uf, "phase": phase, "prr": prr})


def sim_summary(s: UFCSettings, sim: pd.DataFrame) -> dict:
    ufc = sim[sim["phase"].isin(["UFC期間", "終了前"])]
    eps = 1e-6
    return {
        "uf_total": float(sim["uf_vol"].iloc[-1]),
        "end_dbv": float(sim["dbv"].iloc[-1]),
        "min_dbv": float(sim["dbv"].min()),
        "at_max_pct": float((ufc["ufr"] >= s.ufr_max - eps).mean() * 100) if len(ufc) else np.nan,
        "at_min_pct": float((ufc["ufr"] <= s.ufr_min + eps).mean() * 100) if len(ufc) else np.nan,
        "max_dev": float((sim["dbv"] - sim["target"]).abs().max()),
    }


def band(s: UFCSettings, models: list[ResponseModel], lo: float = 10, hi: float = 90) -> pd.DataFrame:
    """モデル群でのシミュレーション結果の幅(ΔBV・除水量の lo〜hi パーセンタイル)。"""
    if not models:
        return pd.DataFrame()
    sims = [simulate(s, m) for m in models]
    t = sims[0]["min"].to_numpy()
    D = np.vstack([x["dbv"].to_numpy() for x in sims])
    U = np.vstack([x["uf_vol"].to_numpy() for x in sims])
    return pd.DataFrame({"min": t, "dbv_lo": np.percentile(D, lo, 0), "dbv_hi": np.percentile(D, hi, 0),
                         "uf_lo": np.percentile(U, lo, 0), "uf_hi": np.percentile(U, hi, 0)})


def solve_end_dbv(s: UFCSettings, model: ResponseModel, uf_goal: float | None = None,
                  lo: float = -30.0) -> float:
    """目標除水量に届く、もっとも浅い終了時ΔBV(%)。上限速度などで届かなければ NaN。"""
    goal = s.target_uf if uf_goal is None else uf_goal
    for e in np.arange(-1.0, lo - 0.01, -0.5):
        x = UFCSettings.from_dict({**asdict(s), "end_dbv": float(e)})
        if simulate(x, model)["uf_vol"].iloc[-1] >= goal - 0.05:
            return float(e)
    return np.nan


def replay(model: ResponseModel, ufr_grid: np.ndarray) -> np.ndarray:
    """実際の除水速度(1分刻み)を与えたときのΔBVの予測。"""
    return _replay_mat(model.coef, np.asarray(ufr_grid, float)[None, :])[0]


def uf_from_dbv(model: ResponseModel, dbv: np.ndarray) -> float:
    """ΔBVが dbv(1分刻み)の経過をたどったときの除水量(L) = BV×ΔBVの低下 + PRRの積算。
    BV-UFCではΔBVが目標カーブに沿うよう制御されるため、除水量はこの関係でほぼ決まる。"""
    t = np.arange(len(dbv)) / 60
    return float(model.bv * (dbv[0] - dbv[-1]) / 100 + np.sum(model.prr(dbv[:-1], t[:-1])) / 60)


def backtest(trajs: dict, patient_sids_by_date: list[int], pooled: ResponseModel,
             n_last: int = 10, min_prior: int = 3) -> pd.DataFrame:
    """その患者の直近症例を、それより前の症例だけで作ったモデルで検証する。
    実際のΔBVの経過から除水量を予測し、実際の除水量と比べる(全患者モデルでの予測も並べる)。"""
    sids = [int(s) for s in patient_sids_by_date if int(s) in trajs]
    rows = []
    for j in range(max(min_prior, len(sids) - n_last), len(sids)):
        m = fit_response(trajs, sids[:j], pooled)
        u, y, _ = trajs[sids[j]]
        actual = float(np.sum(u[:-1]) / 60)
        if actual < 0.3:
            continue
        rows.append({"session_id": sids[j], "uf_actual": actual, "uf_pred": uf_from_dbv(m, y),
                     "uf_pred_pooled": uf_from_dbv(pooled, y), "end_dbv": float(y[-1])})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["err"] = d["uf_pred"] - d["uf_actual"]
        d["err_pooled"] = d["uf_pred_pooled"] - d["uf_actual"]
    return d


# ----------------------------------------------------------------- 指数の目安
def session_shapes(trajs: dict) -> pd.DataFrame:
    """症例ごとのΔBVの経過の形: 治療を4等分した時点のΔBVが、終了時ΔBVの何%だったか(=実績の指数1〜3)。
    終了時ΔBVが浅い症例(−3%より浅い)や短い症例は、割合が不安定なため除く。"""
    rows = []
    for sid, (u, y, _) in trajs.items():
        n = len(y)
        if n < 120 or y[-1] > -3:
            continue
        e = y[-1]
        rows.append({"session_id": int(sid), "end_dbv": float(e),
                     **{f"idx{k}": float(np.clip(y[int(n * k / 4)] / e * 100, -50, 150)) for k in (1, 2, 3)}})
    return pd.DataFrame(rows, columns=["session_id", "end_dbv", "idx1", "idx2", "idx3"])


def index_guide(shapes: pd.DataFrame, ok_ids, bad_ids, min_n: int = 5) -> dict | None:
    """血圧低下なく終えた症例(ok_ids)と、低下があった症例(bad_ids)の指数の実績(中央値)と、目安の指数。"""
    ok = shapes[shapes["session_id"].isin(set(int(x) for x in ok_ids))]
    bad = shapes[shapes["session_id"].isin(set(int(x) for x in bad_ids))]
    if len(ok) < min_n:
        return None
    med = ok[["idx1", "idx2", "idx3"]].median()
    rec, prev = [], 0.0
    for k in ("idx1", "idx2", "idx3"):  # 5刻みに丸め、指数1 ≤ 指数2 ≤ 指数3 ≤ 100 にそろえる
        v = float(np.clip(round(med[k] / 5) * 5, max(prev, 5.0), 100.0))
        rec.append(v)
        prev = v
    return {"ok": med.to_dict(), "bad": bad[["idx1", "idx2", "idx3"]].median().to_dict() if len(bad) else {},
            "n_ok": len(ok), "n_bad": len(bad), "rec": rec, "end_ok": float(ok["end_dbv"].median())}


# ----------------------------------------------------------------- 血圧低下リスク
def risk_fit(features: pd.DataFrame, patient_sids) -> dict | None:
    """血圧測定点の ΔBV と血圧低下の関係(ロジスティック)。患者で推定できなければ全患者で推定。"""
    from stats_models import fit_logistic
    f = features.dropna(subset=["dbv"])
    sids = set(int(x) for x in patient_sids)
    own = f[f["session_id"].isin(sids)]
    for d, src in ((own, "この患者"), (f, "全患者")):
        r = fit_logistic(-d["dbv"].to_numpy(float), d["hypo"].to_numpy(float))
        if r and r["b"] > 0:
            return {**r, "source": src}
    return None


def risk_at(fit: dict | None, dbv) -> np.ndarray:
    if not fit:
        return np.full(np.shape(dbv), np.nan)
    z = fit["a"] + fit["b"] * (-np.asarray(dbv, float))
    return 1 / (1 + np.exp(-z))


def dbv_for_risk(fit: dict | None, p: float) -> float:
    """血圧低下の確率(測定1回あたり)が p になる ΔBV。"""
    if not fit or fit["b"] <= 0:
        return np.nan
    return -((np.log(p / (1 - p)) - fit["a"]) / fit["b"])


# ----------------------------------------------------------------- 実績
def ufc_used(ts_all: pd.DataFrame) -> pd.Series:
    """症例ごとにBV-UFCが使われたか(step no が1以上の行がある)。"""
    if "ufc_step" not in ts_all:
        return pd.Series(dtype=bool)
    return ts_all.groupby("session_id")["ufc_step"].max().fillna(0) > 0


def step_spans(ts: pd.DataFrame) -> pd.DataFrame:
    """1症例のステップ区間(連続する同じ step no をまとめる)。"""
    d = dedupe_time(ts)
    if "ufc_step" not in d or d["ufc_step"].fillna(0).max() <= 0:
        return pd.DataFrame(columns=["step", "t0", "t1"])
    s = d["ufc_step"].fillna(0).astype(int).to_numpy()
    t = d["treat_sec"].to_numpy(float) / 60
    cut = np.r_[0, np.where(np.diff(s) != 0)[0] + 1, len(s)]
    rows = [{"step": int(s[a]), "t0": t[a], "t1": t[min(b, len(t) - 1)]} for a, b in zip(cut[:-1], cut[1:])]
    return pd.DataFrame(rows)


def step_table(ts_all: pd.DataFrame, session_ids, crit) -> pd.DataFrame:
    """BV-UFC使用症例のステップごとの集計(時間・ΔBV・除水速度・PRR・血圧低下)。"""
    ids = set(int(i) for i in session_ids)
    rows = []
    for sid, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id"):
        d = dedupe_time(g)
        bp = bp_points(d, crit)
        bp = bp[bp["seq"] > 0] if len(bp) else bp
        for r in step_spans(d).itertuples():
            sel = d[(d["treat_sec"] / 60 >= r.t0) & (d["treat_sec"] / 60 <= r.t1)]
            b = bp[(bp["treat_min"] >= r.t0) & (bp["treat_min"] <= r.t1)] if len(bp) else bp
            rows.append({"session_id": int(sid), "step": r.step, "dur": r.t1 - r.t0,
                         "dbv0": sel["dbv_avg5"].iloc[0], "dbv1": sel["dbv_avg5"].iloc[-1],
                         "ufr": sel["ufr"].mean(), "prr": sel["prr"].mean(),
                         "n_bp": len(b), "n_hypo": int(b["hypo"].sum()) if len(b) else 0})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    g = d.groupby("step").agg(症例数=("session_id", "nunique"), 平均時間_分=("dur", "mean"),
                              開始ΔBV=("dbv0", "mean"), 終了ΔBV=("dbv1", "mean"),
                              平均除水速度=("ufr", "mean"), 平均PRR=("prr", "mean"),
                              血圧測定数=("n_bp", "sum"), 低下数=("n_hypo", "sum")).reset_index()
    g["低下の頻度(%)"] = g["低下数"] / g["血圧測定数"].replace(0, np.nan) * 100
    g.insert(1, "ステップ", g["step"].map(lambda x: STEP_LABELS.get(x, f"step{x}")))
    return g.drop(columns="step").rename(columns={"平均時間_分": "平均時間(分)", "開始ΔBV": "開始時ΔBV(%)",
                                                  "終了ΔBV": "終了時ΔBV(%)", "平均除水速度": "平均除水速度(L/h)",
                                                  "平均PRR": "平均PRR(L/h)"})


def index_achievement(ts_all: pd.DataFrame, session_ids) -> pd.DataFrame:
    """各UFC症例で、区間1〜3の終了時のΔBVが、区間4終了時のΔBVの何%だったか(装置の指数の実績)。"""
    ids = set(int(i) for i in session_ids)
    rows = []
    for sid, g in ts_all[ts_all["session_id"].isin(ids)].groupby("session_id"):
        d = dedupe_time(g)
        sp = step_spans(d)
        ends = {int(k): v for k, v in sp.groupby("step")["t1"].max().items()}
        if not {3, 4, 5, 6} <= set(ends):
            continue
        t = d["treat_sec"].to_numpy(float) / 60
        y = d["dbv_avg5"].to_numpy(float)
        ok = ~np.isnan(y)
        e = float(np.interp(ends[6], t[ok], y[ok]))
        if e > -2:  # 終了時ΔBVが浅すぎる(初期化など)症例は割合が不安定なため除く
            continue
        rows.append({"session_id": int(sid), "end_dbv": e,
                     **{f"idx{k - 2}": float(np.interp(ends[k], t[ok], y[ok])) / e * 100 for k in (3, 4, 5)}})
    return pd.DataFrame(rows)

"""BV計(DCS-100NX)研究用ツール  —  streamlit run app.py"""
from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as an
import db
import bv_settings as bset
import stats_models as sm_
import ufc
from bv_parser import (COLUMN_LABELS, EVENT_GROUPS, EVENT_LABELS, META_NAME_IDX, N_DATA_COLS, TREAT_MODE_LABELS,
                       anonymize_bytes, decode, decode_event, event_text, mode_label, parse_csv)

st.set_page_config(page_title="BV計 解析ツール", page_icon="🩸", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.4rem; padding-bottom: 3rem;}
[data-testid="stMetricLabel"] p {white-space: normal; font-size: .82rem; opacity: .8;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stMetricDelta"] {font-size: .78rem;}
h1, h2 {margin-bottom: .2rem;}
</style>""", unsafe_allow_html=True)
db.init()


@st.cache_resource(show_spinner="データを更新中…")
def _startup() -> int:
    db.migrate_bp_clean()  # 血圧の無効値(-1)を欠測化(初回のみ)
    db.backfill_ufc_steps()  # BV-UFCステップ列を元CSVから補完(初回のみ)
    db.migrate_patient_by_name()  # 装置IDごとに混ざっていた患者を、氏名で分け直す(初回のみ)
    return db.backfill_modes()  # 旧データの治療モードを補完(初回のみ)


_startup()
if "auto_bk_done" not in st.session_state:  # 画面を開いたとき、今日まだなら自動バックアップ
    st.session_state["auto_bk_done"] = True
    if _msg := bset.auto_backup_if_due(bset.load()):
        st.session_state["toast"] = _msg

PAGES = ["① 取り込み", "② 症例解析", "③ 患者別", "④ 症例間比較・統計", "⑤ 血圧低下とΔBV", "⑥ BV-UFC", "⑦ データ管理",
         "⑧ 設定"]
HYPO_OPTS = ["未記入", "なし", "あり"]
PLOT_CONFIG = {"displaylogo": False,
               "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "bv_chart"}}

# ----------------------------------------------------------------- loaders
REV = db.revision()


@st.cache_data(show_spinner=False)
def load_sessions(rev: int) -> pd.DataFrame:
    d = db.sessions_df()
    d["treat_min"] = d["treat_sec"] / 60
    d["dur_group"] = d["treat_min"].map(an.duration_group)
    d["mode_label"] = d["treat_mode"].map(mode_label)
    return d


@st.cache_data(show_spinner=False)
def load_patients(rev: int) -> pd.DataFrame:
    return db.patients_df()


@st.cache_data(show_spinner=False)
def load_ts(rev: int, sid: int) -> pd.DataFrame:
    return db.timeseries(sid)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_all_ts(rev: int) -> pd.DataFrame:
    """全症例の時系列(約55万行)。コピーを避けるためcache_resourceを使う。読み取り専用として扱うこと。"""
    return db.all_timeseries()


@st.cache_data(show_spinner=False)
def load_uf(rev: int) -> pd.DataFrame:
    return db.session_uf()


@st.cache_data(show_spinner="全症例を集計中…")
def load_summary(rev: int, crit: an.HypoCriteria, window: int) -> pd.DataFrame:
    return an.session_summary(db.sessions_df(), load_all_ts(rev), crit, window)


@st.cache_data(show_spinner="全症例の血圧測定点を解析中…")
def load_features(rev: int, crit: an.HypoCriteria, window: int) -> pd.DataFrame:
    return an.all_features(load_all_ts(rev), crit, window)


@st.cache_data(show_spinner="判定基準ごとに集計中…")
def load_preset_table(rev: int, window: int) -> pd.DataFrame:
    summ = load_summary(rev, an.PRESETS["標準: SBP 20↓ または <90"], window)
    return an.hypo_by_preset(load_all_ts(rev), summ, an.PRESETS)


@st.cache_data(show_spinner="判定基準ごとに再判定中…")
def load_preset_flags(rev: int) -> pd.DataFrame:
    return an.preset_flags(load_all_ts(rev), an.PRESETS)


@st.cache_data(show_spinner="除水停止との関連を集計中…")
def load_uf_stop(rev: int, crit: an.HypoCriteria, ids: tuple, window: int) -> pd.DataFrame:
    return sm_.uf_stop_after_hypo(load_all_ts(rev), ids, crit, window)


@st.cache_data(show_spinner=False)
def curve_stats(rev: int, ids: tuple, col: str, normalize: bool):
    """症例群の平均曲線の統計(DataFrame)と、使用した症例数。"""
    x, m, used = an.curves_matrix(load_all_ts(rev), ids, col, normalize)
    return an.summarize_curves(x, m), len(used)


@st.cache_data(show_spinner=False)
def load_patient_ts(rev: int, pid: str) -> pd.DataFrame:
    return db.patient_timeseries(pid)


@st.cache_data(show_spinner="集計中…")
def load_patient_summary(rev: int, pid: str, crit: an.HypoCriteria, window: int) -> pd.DataFrame:
    sess = db.sessions_df()
    return an.session_summary(sess[sess.patient_id == pid], db.patient_timeseries(pid), crit, window)


@st.cache_data(show_spinner=False)
def parse_cached(name: str, raw: bytes, pid_override: str):
    """(ParsedSession | None, エラー文)"""
    try:
        return parse_csv(raw, name, pid_override or None), ""
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def summary_labels(w: int) -> dict[str, str]:
    return {
        "treat_min": "治療時間(分)", "uf_total": "除水量(L)", "uf_rate_mean": "平均除水速度(L/h)",
        "min_dbv": "最小ΔBV(%)", "min_dbv_time": "最小ΔBV到達時間(分)", "end_dbv": "終了時ΔBV(%)",
        "dbv_slope_per_h": "ΔBV傾き(%/h)", "below_lower_min": "リファレンス下限以下(分)",
        "below_lower_pct": "リファレンス下限以下(%)",
        f"max_drop_{w}min": f"最大ΔBV低下/{w}分(%)",
        "sbp_base": "開始時SBP", "sbp_min": "最低SBP", "sbp_drop_max": "最大SBP低下量",
        "first_hypo_time": "初回血圧低下時間(分)", "dbv_at_first_hypo": "初回血圧低下時ΔBV(%)",
        "dbv_slope_before_hypo": "低下直前ΔBV傾き(%/h)", "n_bp": "血圧測定回数",
        "n_init": "ΔBV初期化回数", "n_uf_stop": "除水停止回数",
    }


def pid_label(pid: str, patients: pd.DataFrame) -> str:
    """氏名（患者ID / 表示名）。氏名表示がOFFのときは 患者ID（表示名）のみ。"""
    r = patients.loc[patients["patient_id"] == pid]
    name = alias = ""
    if len(r):
        name = r["name"].iloc[0] if pd.notna(r["name"].iloc[0]) else ""
        alias = r["alias"].iloc[0] if pd.notna(r["alias"].iloc[0]) else ""
    if SHOW_NAMES and name:
        return f"{name}（{pid}{' / ' + alias if alias else ''}）"
    return f"{pid}（{alias}）" if alias else pid


# ----------------------------------------------------------------- sidebar / navigation
sessions = load_sessions(REV)
patients = load_patients(REV)
ACTIVE_PIDS = patients.loc[patients.n_sessions > 0, "patient_id"].tolist()  # 症例のある患者のみ選択肢にする
if "page" not in st.session_state:  # 初回: データがあれば解析画面、無ければ取り込み画面
    st.session_state["page"] = PAGES[1] if len(sessions) else PAGES[0]
if toast := st.session_state.pop("toast", None):
    st.toast(toast, icon="✅")


def goto(page: str) -> None:
    st.session_state["page"] = page


st.sidebar.title("🩸 BV計 解析ツール")
PAGE = st.sidebar.radio("メニュー", PAGES, key="page", label_visibility="collapsed")
SHOW_NAMES = st.sidebar.toggle("氏名を表示", True,
                               help="OFFの間は画面上で氏名を伏せます(画面共有・スクリーンショット用)。データは削除されません。")
if "onedrive" in str(db.BASE_DIR).lower():
    st.sidebar.warning("データ保存先がOneDrive配下です。患者情報がクラウドに同期されます。")
with st.sidebar.expander("血圧低下の判定基準", expanded=False):
    PRESET = st.selectbox("判定基準", [*an.PRESETS, "カスタム"], key="hypo_preset")
    if PRESET == "カスタム":
        def _opt(label, key, default, lo, hi):
            on = st.checkbox(label, value=default is not None, key=f"{key}_on")
            if not on:
                return None
            return float(st.number_input(label + "（mmHg）", lo, hi, int(default or 20), 5, key=f"{key}_v"))

        CRIT = an.HypoCriteria(_opt("開始時SBPから低下", "c_drop", 20, 5, 80), _opt("SBPが次の値未満", "c_abs", 90, 50, 130),
                               _opt("開始時MAPから低下", "c_map", None, 5, 60), _opt("前回測定からSBP低下", "c_prev", None, 5, 80))
    else:
        CRIT = an.PRESETS[PRESET]
    st.caption("判定: " + CRIT.describe() + "（治療前の最初の測定を基準）。臨床記録を入力した症例は、そちらを優先して集計します。")
    WINDOW = st.number_input("ΔBV変化の評価窓 (分)", 10, 120, 30, 5)
st.sidebar.divider()
st.sidebar.caption(f"登録: 患者 {len(patients)} 名 / 症例 {len(sessions)} 件")
st.sidebar.caption("🔒 このPC内のみで動作（外部通信なし）")


def group_filter(key: str):
    """治療時間群・治療モードの絞り込みUI。(選択された時間群, 選択されたモード) を返す。"""
    order = (sessions.drop_duplicates("mode_label").assign(_k=lambda d: d["treat_mode"].fillna(99))
             .sort_values("_k")["mode_label"].tolist())
    c1, c2 = st.columns(2)
    dur = c1.multiselect("治療時間群", an.DUR_GROUPS, default=an.DUR_GROUPS, key=f"{key}_dur",
                         help="5時間群: 270分以上 / 4時間群: 210〜270分 / 短時間: 210分未満")
    modes = c2.multiselect("治療モード", order, default=order, key=f"{key}_mode")
    return dur, modes


def apply_filter(df: pd.DataFrame, dur: list[str], modes: list[str]) -> pd.DataFrame:
    return df[df["dur_group"].isin(dur) & df["mode_label"].isin(modes)]


def empty_state() -> None:
    st.info("まだデータがありません。CSVを取り込むと、ここに解析結果が表示されます。")
    st.button("① CSV取り込みへ", on_click=goto, args=(PAGES[0],), type="primary")


# --- 患者/症例の選択状態（ページをまたいで保持）
def _newest_sid(pid: str) -> int:
    sub = sessions[sessions.patient_id == pid].sort_values("session_dt")
    return int(sub.session_id.iloc[-1])


def _on_pid() -> None:
    pid = st.session_state["w_pid"]
    st.session_state["sel_pid"], st.session_state["sel_sid"] = pid, _newest_sid(pid)


def _on_pid_p() -> None:
    pid = st.session_state["w_pid_p"]
    st.session_state["sel_pid"], st.session_state["sel_sid"] = pid, _newest_sid(pid)


def _on_sid() -> None:
    st.session_state["sel_sid"] = st.session_state["w_sid"]


def _step(sids_asc: list[int], delta: int) -> None:
    i = sids_asc.index(st.session_state["sel_sid"]) + delta
    if 0 <= i < len(sids_asc):
        st.session_state["sel_sid"] = sids_asc[i]


def current_selection():
    """(patient_id, session_id, その患者の症例DF[日付昇順])。未選択なら最新の症例を選ぶ。"""
    pid = st.session_state.get("sel_pid")
    if pid not in set(ACTIVE_PIDS):
        last = sessions.sort_values("session_dt").iloc[-1]
        pid = last.patient_id
        st.session_state["sel_sid"] = int(last.session_id)
    sub = sessions[sessions.patient_id == pid].sort_values("session_dt")
    if st.session_state.get("sel_sid") not in set(sub.session_id):
        st.session_state["sel_sid"] = int(sub.session_id.iloc[-1])
    st.session_state["sel_pid"] = pid
    return pid, int(st.session_state["sel_sid"]), sub


# ----------------------------------------------------------------- charts
def session_figure(ts: pd.DataFrame, bp: pd.DataFrame, n_phase: int, show_phase: bool,
                   event_groups: list[str], show_ref: bool = True, steps: pd.DataFrame | None = None) -> go.Figure:
    d = ts.copy()
    d["treat_min"] = d["treat_sec"] / 60
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.5, 0.25, 0.25],
                        specs=[[{}], [{}], [{"secondary_y": True}]])
    # --- 1: ΔBV
    if show_ref:
        fig.add_trace(go.Scatter(x=d.treat_min, y=d.ref_upper, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=d.treat_min, y=d.ref_lower, fill="tonexty", name="リファレンスエリア(装置出力)",
                                 fillcolor="rgba(110,180,110,.20)", line=dict(width=0),
                                 hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d.treat_min, y=d.dbv, name="ΔBV", line=dict(color="#9bb0c9", width=1),
                             customdata=d.clock.str[11:19],
                             hovertemplate="%{y:.1f}%  (%{customdata})"), 1, 1)
    fig.add_trace(go.Scatter(x=d.treat_min, y=d.dbv_avg5, name="ΔBV 5分平均",
                             line=dict(color="#1f5fbf", width=2.6),
                             hovertemplate="%{y:.1f}%"), 1, 1)
    if len(bp):
        col = np.where(bp.hypo, "#d62728", "#2b2b2b")
        fig.add_trace(go.Scatter(
            x=bp.treat_min, y=bp.dbv, mode="markers+text", name="血圧測定時点",
            marker=dict(size=11, color=col, symbol="diamond", line=dict(color="white", width=1)),
            text=[f"{int(s)}/{int(dd)}" for s, dd in zip(bp.sbp, bp.dbp)],
            textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(bp))],
            textfont=dict(size=11), hovertemplate="BP %{text}  ΔBV %{y:.1f}%"), 1, 1)
    if event_groups:
        want = set().union(*(EVENT_GROUPS[g] for g in event_groups))
        ev = d[d.event.fillna(0) != 0]
        pts = [(r.treat_min, r.dbv, event_text(r.event)) for r in ev.itertuples()
               if set(decode_event(r.event)) & want]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name="イベント",
                marker=dict(symbol="x", size=9, color="#e08a00"),
                text=[p[2] for p in pts], hovertemplate="%{text}"), 1, 1)
        if "ΔBV初期化" in event_groups:
            for r in ev.itertuples():
                if 1 in decode_event(r.event):
                    fig.add_vline(x=r.treat_min, line=dict(color="rgba(130,80,170,.6)", dash="dot", width=1.5),
                                  row=1, col=1, annotation_text="ΔBV初期化", annotation_font_size=10,
                                  annotation_position="top")
    # --- 2: 血圧・脈拍
    if len(bp):
        for key, name, color in (("sbp", "収縮期血圧", "#d62728"), ("dbp", "拡張期血圧", "#e68a8a"),
                                 ("pulse", "脈拍", "#2a9d8f")):
            fig.add_trace(go.Scatter(x=bp.treat_min, y=bp[key], mode="lines+markers", name=name,
                                     line=dict(color=color, width=1.8), marker=dict(size=7),
                                     hovertemplate="%{y:.0f}"), 2, 1)
        if CRIT.sbp_abs is not None:
            fig.add_hline(y=CRIT.sbp_abs, line=dict(color="rgba(214,39,40,.4)", dash="dot"), row=2, col=1)
    # --- 3: 除水
    fig.add_trace(go.Scatter(x=d.treat_min, y=d.ufr, name="除水速度", line=dict(color="#e08a00", width=1.8,
                  shape="hv"), hovertemplate="%{y:.2f} L/h"), 3, 1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d.treat_min, y=d.uf_vol, name="除水量", line=dict(color="#6c757d", width=1.8),
                             hovertemplate="%{y:.2f} L"), 3, 1, secondary_y=True)
    # --- 装飾
    for tm in bp[bp.hypo].treat_min if len(bp) else []:
        fig.add_vline(x=tm, line=dict(color="rgba(214,39,40,.55)", dash="dash", width=1.5),
                      row="all", col=1)
    if steps is not None and len(steps):  # BV-UFCのステップ(CSVの step no)
        for r in steps.itertuples():
            if r.t1 - r.t0 < 0.5:
                continue
            fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor=ufc.STEP_COLORS.get(r.step, "rgba(0,0,0,.05)"), line_width=0,
                          layer="below", row="all", col=1)
            if r.t1 - r.t0 >= 8:
                fig.add_annotation(x=(r.t0 + r.t1) / 2, y=1.0, xref="x", yref="y domain", yanchor="bottom",
                                   text=ufc.STEP_LABELS.get(r.step, f"step{r.step}").split("（")[0], showarrow=False,
                                   font=dict(size=10, color="#777"))
    elif show_phase and n_phase > 1:
        edges = np.linspace(0, d.treat_min.max(), n_phase + 1)
        for k in range(n_phase):
            if k % 2 == 1:
                fig.add_vrect(x0=edges[k], x1=edges[k + 1], fillcolor="rgba(0,0,0,.045)", line_width=0,
                              layer="below", row="all", col=1)
            fig.add_annotation(x=(edges[k] + edges[k + 1]) / 2, y=1.0, xref="x", yref="y domain",
                               yanchor="bottom", text=f"段階{k + 1}", showarrow=False,
                               font=dict(size=11, color="#888"))
    fig.update_yaxes(title_text="ΔBV (%)", row=1, col=1)
    fig.update_yaxes(title_text="血圧 / 脈拍", row=2, col=1)
    fig.update_yaxes(title_text="除水速度 (L/h)", row=3, col=1, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="除水量 (L)", row=3, col=1, secondary_y=True, rangemode="tozero",
                     showgrid=False)
    fig.update_xaxes(title_text="治療経過時間 (分)", row=3, col=1, dtick=30)
    fig.update_layout(height=780, template="plotly_white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=90, b=10),
                      legend=dict(orientation="h", y=1.11, x=0))
    return fig


def wilson(k: int, n: int, z: float = 1.96):
    """比率のWilson区間。(割合%, 下限%, 上限%)"""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p * 100, max(0, c - h) * 100, min(1, c + h) * 100)


def fmt(v, f="{:.1f}", na="–"):
    return na if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


# ----------------------------------------------------------------- ① import
def page_import():
    st.header("① CSV取り込み")
    if msg := st.session_state.pop("flash", None):
        st.success(msg)
        st.button("② 取り込んだ症例を解析する →", on_click=goto, args=(PAGES[1],), type="primary")
    st.caption("DCS-100NXのCSVを複数まとめて取り込めます。患者はCSV内の**氏名**で区別し（CSVの「ID」は装置の番号のため）、"
               "患者ごとのフォルダに元CSVを保存します（すべてこのPC内）。")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        store_name = st.checkbox("氏名をこのPCのDBに保存", True,
                                 help="患者管理用。データはこのPC内にのみ保存され、外部には送信されません。")
        anonymize = st.checkbox("保存するCSV複製から氏名を除去", False)
    with c2:
        store_raw = st.checkbox("元CSVを患者フォルダへ保存", True)
    with c3:
        pid_override = st.text_input("患者IDを手動指定（空欄=自動）", "",
                                     help="全ファイルに同じIDを割り当てます。単一患者のファイルをまとめて入れる時に使用。"
                                          "空欄なら、CSVの氏名から患者IDを作ります。")

    tab_up, tab_dir = st.tabs(["ファイルを選択（複数可）", "フォルダを指定して一括取り込み"])
    items: list[tuple[str, bytes]] = []
    with tab_up:
        files = st.file_uploader("CSVをドラッグ＆ドロップ", type=["csv"], accept_multiple_files=True)
        items += [(f.name, f.getvalue()) for f in files or []]
    with tab_dir:
        d1, d2 = st.columns([4, 1])
        folder = d1.text_input("フォルダのパス", placeholder=r"D:\BVデータ")
        recursive = d2.checkbox("サブフォルダも", True)
        if st.button("フォルダをスキャン"):
            p = Path(folder.strip().strip('"'))
            if not p.is_dir():
                st.error("フォルダが見つかりません")
            else:
                pat = "**/*.csv" if recursive else "*.csv"
                st.session_state["scan"] = sorted(str(x) for x in p.glob(pat))
        scan = st.session_state.get("scan", [])
        if scan:
            st.info(f"{len(scan)} 個のCSVを検出")
            for path in scan:
                try:
                    items.append((Path(path).name, Path(path).read_bytes()))
                except OSError as e:
                    st.warning(f"{path}: {e}")
            if st.button("スキャン結果をクリア"):
                st.session_state["scan"] = []
                st.rerun()

    if not items:
        st.info("ファイルを選択するか、フォルダをスキャンしてください。")
        return

    st.caption("同じ患者で測定時間が重なるCSVは同一症例とみなし、自動で振り分けます："
               "より長い(行数が多い)データだけを取り込み、短い方はスキップします。"
               "登録済みより長いデータは置換し、入力済みの臨床記録・体重は引き継ぎます。")
    parsed = [(name, raw, *parse_cached(name, raw, pid_override)) for name, raw in items]
    decided = iter(auto_dedup([p for *_, p, _ in parsed if p is not None], load_spans(REV)))
    rows, ready = [], []
    for name, raw, p, err in parsed:
        if p is None:
            rows.append({"ファイル": name, "状態": "❌ エラー", "患者ID": "", "日時": "", "治療時間(分)": None,
                         "行数": None, "備考": err})
            continue
        status, note, replace_ids = next(decided)
        rows.append({"ファイル": name, "状態": status,
                     **({"氏名": p.patient_name or "（なし）"} if SHOW_NAMES else {}),
                     "患者ID": p.patient_id, "装置": p.console_id,
                     "日時": p.session_dt.strftime("%Y-%m-%d %H:%M"), "治療時間(分)": round(p.treat_sec / 60),
                     "時間群": an.duration_group(p.treat_sec / 60), "モード": mode_label(p.treat_mode),
                     "行数": p.n_rows, "備考": " / ".join([note, *p.warnings]).strip(" /")})
        if status in (ST_NEW, ST_REPLACE):
            ready.append((p, raw, replace_ids))
    prev = pd.DataFrame(rows)
    st.dataframe(prev, hide_index=True, width="stretch")
    vc = prev["状態"].value_counts()
    n_new, n_rep = vc.get(ST_NEW, 0), vc.get(ST_REPLACE, 0)
    st.write(f"新規 **{n_new}** 件 / 置換 **{n_rep}** 件 / 重複スキップ {vc.get(ST_DUP, 0)} 件 / "
             f"エラー {vc.get('❌ エラー', 0)} 件")

    label = f"新規 {n_new} 件" + (f"・置換 {n_rep} 件" if n_rep else "") + " を取り込む"
    if ready and st.button(label, type="primary"):
        bar = st.progress(0.0)
        last = None
        for i, (p, raw, replace_ids) in enumerate(ready, 1):
            stored = (anonymize_bytes(raw) if anonymize else raw) if store_raw else None
            last = (p.patient_id, db.add_session(p, stored, store_name, replace_ids))
            bar.progress(i / len(ready))
        if last:  # 解析画面で最後に取り込んだ症例を開く
            st.session_state["sel_pid"], st.session_state["sel_sid"] = last
        st.cache_data.clear()
        st.session_state["flash"] = f"{n_new} 件を取り込みました" + (f"（{n_rep} 件は既存症例を置換）" if n_rep else "")
        st.rerun()


ST_NEW, ST_REPLACE, ST_DUP = "🆕 新規", "🔄 置換(より長いデータ)", "⏭ 重複(スキップ)"


@st.cache_data(show_spinner=False)
def load_spans(rev: int) -> pd.DataFrame:
    return db.session_spans()


def auto_dedup(cands: list, spans: pd.DataFrame) -> list:
    """取り込み候補(ParsedSession)ごとに (状態, 備考, 置換する session_id) を返す。
    同一患者で測定時間が重なるものは同一症例とみなし、行数の多いデータを1つだけ残す。"""
    out: list = [None] * len(cands)
    seen_hash: set[str] = set()
    kept: list = []  # このバッチで採用した (patient_id, t0, t1)
    for i in sorted(range(len(cands)), key=lambda i: -cands[i].n_rows):  # 長いデータから採用
        p = cands[i]
        t0, t1 = p.ts["clock"].iloc[0], p.ts["clock"].iloc[-1]
        if p.file_hash in seen_hash or db.hash_exists(p.file_hash):
            out[i] = (ST_DUP, "同一ファイル", [])
            continue
        seen_hash.add(p.file_hash)
        if any(pid == p.patient_id and t0 <= k1 and k0 <= t1 for pid, k0, k1 in kept):
            out[i] = (ST_DUP, "同じ症例のより長いCSVを今回採用", [])
            continue
        olds = spans[(spans.patient_id == p.patient_id) & (spans.t0 <= t1) & (spans.t1 >= t0)]
        if len(olds) and p.n_rows <= olds.n_rows.max():
            out[i] = (ST_DUP, "同じ症例が登録済み", [])
            continue
        kept.append((p.patient_id, t0, t1))
        if len(olds):
            out[i] = (ST_REPLACE, f"登録済み {olds.n_rows.max()} 行 → {p.n_rows} 行",
                      [int(x) for x in olds.session_id])
        else:
            out[i] = (ST_NEW, "", [])
    # 同日に時間の重ならない別症例がある場合は、念のため備考に表示する
    db_days = set(zip(spans.patient_id, spans.t0.astype(str).str[:10]))
    for i, p in enumerate(cands):
        if out[i][0] != ST_NEW:
            continue
        key = (p.patient_id, p.ts["clock"].iloc[0][:10])
        n_batch = sum(pid == key[0] and k0[:10] == key[1] for pid, k0, _ in kept)
        if n_batch > 1 or key in db_days:
            out[i] = (ST_NEW, "同日に時間の重ならない別症例あり", [])
    return out


# ----------------------------------------------------------------- ② session
def pick_session(key=None):
    """患者・施行日の選択(前回/次回ボタン付き)。(patient_id, session_id) を返す。"""
    if sessions.empty:
        empty_state()
        return None
    pid, sid, sub = current_selection()
    sids_asc = [int(x) for x in sub.session_id]
    labels = {int(r.session_id): f"{r.session_dt:%Y-%m-%d %H:%M}　（{n}/{len(sub)}回目）"
              for n, r in enumerate(sub.itertuples(), 1)}
    st.session_state["w_pid"], st.session_state["w_sid"] = pid, sid
    c1, c2, c3, c4 = st.columns([2.2, 2.6, 1, 1], vertical_alignment="bottom")
    c1.selectbox("患者", ACTIVE_PIDS, key="w_pid", on_change=_on_pid,
                 format_func=lambda x: pid_label(x, patients))
    c2.selectbox("施行日", sids_asc[::-1], key="w_sid", on_change=_on_sid, format_func=labels.get)
    i = sids_asc.index(sid)
    c3.button("◀ 前回", on_click=_step, args=(sids_asc, -1), disabled=i == 0, use_container_width=True)
    c4.button("次回 ▶", on_click=_step, args=(sids_asc, 1), disabled=i == len(sids_asc) - 1,
              use_container_width=True)
    return pid, sid


def _txt(v) -> str:
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def clinical_form(row: pd.Series, sid: int) -> None:
    uf = load_uf(REV).set_index("session_id")["uf_total"].get(sid)
    st.caption(f"この症例の除水量（CSVより）: **{fmt(uf, '{:.2f}')} L**　／　治療時間 {row['treat_min']:.0f} 分"
               f"　／　平均除水速度 {fmt(uf / row['treat_min'] * 60 if uf is not None and row['treat_min'] else None, '{:.2f}')} L/h")
    with st.form(f"clin_{sid}"):
        cur = {0: "なし", 1: "あり"}.get(row["hypo_event"] if pd.notna(row["hypo_event"]) else None, "未記入")
        hy = st.radio("血圧低下（臨床記録）", HYPO_OPTS, index=HYPO_OPTS.index(cur), horizontal=True,
                      help="記録すると、集計では自動判定より優先されます。")
        c1, c2 = st.columns(2)
        sym = c1.text_input("症状", _txt(row["symptom"]))
        iv = c2.text_input("対応（補液・除水停止など）", _txt(row["intervention"]))
        memo = st.text_area("メモ", _txt(row["memo"]), height=80)
        if st.form_submit_button("保存", type="primary"):
            db.update_clinical(pd.DataFrame([{
                "session_id": sid, "hypo_event": {"なし": 0, "あり": 1}.get(hy),
                "symptom": sym or None, "intervention": iv or None, "memo": memo or None}]))
            st.cache_data.clear()
            st.session_state["toast"] = "臨床記録を保存しました"
            st.rerun()


def raw_view(row: pd.Series) -> None:
    """保存済みの元CSVを、装置の出力のまま表示する(氏名欄は表示設定に従う)。"""
    path = Path(row["stored_path"]) if _txt(row["stored_path"]) else None
    if not path or not path.exists():
        st.info("元CSVが保存されていません（取り込み時に「元CSVを患者フォルダへ保存」をOFFにした症例です）。")
        return
    raw = path.read_bytes()
    text = decode(raw)
    first = text.split("\n", 1)[0].rstrip("\r").split(",")
    meta = first[N_DATA_COLS:]
    if not SHOW_NAMES and len(meta) > META_NAME_IDX:
        meta[META_NAME_IDX] = "（非表示）"
    df = pd.read_csv(io.StringIO(text), index_col=False, header=0).iloc[:, :N_DATA_COLS]
    df.columns = first[:N_DATA_COLS]
    st.dataframe(df, width="stretch", height=360)
    st.caption(f"{path.name}　／　{len(df)} 行 × {len(df.columns)} 列。"
               f"ヘッダー右側のメタ情報（AW列以降）: {', '.join(m for m in meta if m != '')}")
    st.download_button("このCSVをダウンロード", raw, path.name, "text/csv")


def page_session():
    st.header("② 症例解析")
    sel = pick_session()
    if not sel:
        return
    pid, sid = sel
    ts = load_ts(REV, sid)
    bp = an.bp_points(ts, CRIT)
    bvm, bpm = an.bv_metrics(ts, WINDOW), an.bp_metrics(bp, ts, WINDOW)
    row = sessions[sessions.session_id == sid].iloc[0]
    nev = an.event_counts(ts)
    modes_seen = [mode_label(int(m)) for m in _txt(row["mode_set"]).split(",") if m != ""]
    mode_txt = f"**{row['mode_label']}**" + (f"（治療中に {' → '.join(modes_seen)} と変化）" if len(modes_seen) > 1 else "")
    ufc_on = "ufc_step" in ts and ts["ufc_step"].fillna(0).max() > 0
    st.caption(f"装置: {_txt(row.get('console_id'))}　／　治療モード: {mode_txt}　／　**{row['dur_group']}**　／　ΔBV初期化 {nev.get(1, 0)}回　／　除水停止 {nev.get(2, 0)}回"
               + ("　／　**BV-UFC使用**" if ufc_on else ""))
    if nev.get(1, 0) > 1:
        st.warning(f"治療中にΔBV初期化が {nev[1]} 回あります。初期化のたびにΔBVの基準(0%)が取り直されるため、"
                   "曲線が途中で不連続になります（グラフの紫の点線）。")

    with st.container(border=True):
        k = st.columns(6)
        k[0].metric("治療時間", f"{fmt(bvm.get('treat_min'), '{:.0f}')} 分")
        k[1].metric("除水量", f"{fmt(bvm.get('uf_total'), '{:.2f}')} L")
        k[2].metric("最小ΔBV", f"{fmt(bvm.get('min_dbv'))} %", f"{fmt(bvm.get('min_dbv_time'), '{:.0f}')} 分時点",
                    delta_color="off")
        k[3].metric("終了時ΔBV", f"{fmt(bvm.get('end_dbv'))} %")
        k[4].metric("ΔBV傾き", f"{fmt(bvm.get('dbv_slope_per_h'))} %/h", help="治療全体の直線回帰の傾き")
        k[5].metric("リファレンス下限以下の時間", f"{fmt(bvm.get('below_lower_min'), '{:.0f}')} 分",
                    f"{fmt(bvm.get('below_lower_pct'), '{:.0f}')} %", delta_color="off",
                    help="装置が出力したリファレンスエリアの下限を、ΔBVが下回っていた時間")
        k = st.columns(6)
        k[0].metric("開始時SBP", fmt(bpm.get("sbp_base"), "{:.0f}"))
        k[1].metric("最低SBP", fmt(bpm.get("sbp_min"), "{:.0f}"))
        k[2].metric("最大SBP低下", fmt(bpm.get("sbp_drop_max"), "{:.0f}"), help="開始時SBPからの最大低下量")
        k[3].metric("血圧測定回数", bpm.get("n_bp", 0))
        clin = row["hypo_event"]
        if pd.notna(clin):
            k[4].metric("血圧低下", "あり" if clin == 1 else "なし", "臨床記録", delta_color="off")
        else:
            k[4].metric("血圧低下", "あり" if bpm.get("hypo_auto") else "なし", "自動判定", delta_color="off",
                        help="サイドバーの判定基準による自動判定。臨床記録を入力すると優先されます。")
        ft = bpm.get("first_hypo_time")
        k[5].metric("初回低下時のΔBV", f"{fmt(bpm.get('dbv_at_first_hypo'))} %",
                    f"{fmt(ft, '{:.0f}')} 分時点" if ft is not None else None, delta_color="off")

    h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
    h1.caption("◆ = 血圧測定時点（赤 = 低下判定）／赤の破線 = 血圧低下の時間／グラフ右上のカメラで画像保存")
    with h2.popover("⚙ 表示", use_container_width=True):
        steps = ufc.step_spans(ts)
        show_steps = st.checkbox("BV-UFCのステップを表示", True, disabled=steps.empty,
                                 help="CSVの BV-UFC step no。この症例でBV-UFCを使っていない場合は選べません。")
        n_phase = st.number_input("段階の分割数", 1, 8, 4)
        show_phase = st.checkbox("段階を表示", True)
        show_ref = st.checkbox("リファレンスエリアを表示", True)
        ev_groups = st.multiselect("イベント表示", list(EVENT_GROUPS), default=["ΔBV初期化", "除水停止/再開"])
    st.plotly_chart(session_figure(ts, bp, int(n_phase), show_phase, ev_groups, show_ref,
                                   steps if show_steps and not steps.empty else None),
                    use_container_width=True, config=PLOT_CONFIG)

    t1, t2, t5, t3, t4 = st.tabs(["血圧測定点とBV", "治療段階別", "イベント", "臨床記録", "生データ"])
    with t5:
        et = an.event_table(ts)
        if et.empty:
            st.info("この症例にイベントはありません。")
        else:
            st.dataframe(et.style.format(precision=2, na_rep="–"), hide_index=True, width="stretch")
            st.caption("イベントコードはDCS-100NX取扱説明書（表12-1）に基づきます。20秒以内に複数発生した場合は「+」で連結して表示します。")
    with t1:
        st.dataframe(an.bp_context(bp, ts, WINDOW), hide_index=True, width="stretch")
        st.caption("血圧は装置が値を保持して出力するため、値が変化した行を測定点としています。")
    with t2:
        pt = an.phase_table(ts, bp, int(n_phase))
        st.dataframe(pt.style.format(precision=1, na_rep="–"), hide_index=True, width="stretch")
    with t3:
        clinical_form(row, sid)
    with t4:
        mode = st.radio("表示", ["変換後データ（実単位）", "元のCSV（装置の出力そのまま）"], horizontal=True,
                        label_visibility="collapsed")
        if mode.startswith("変換後"):
            st.dataframe(ts.rename(columns=COLUMN_LABELS), width="stretch", height=360)
            st.caption("装置の値を実単位に変換したもの（ΔBV÷10、除水量÷100など）。欠測(-9999)は空欄です。")
        else:
            raw_view(row)


# ----------------------------------------------------------------- ③ patient
def integrated_figure(ts_all: pd.DataFrame, sids: list[int], normalize: bool, show_lines: bool,
                      band: str, dbv_col: str):
    """複数症例を1枚に統合: ΔBV平均±バンド / 血圧測定点 / 血圧低下の頻度 / 除水。(fig, 時間帯集計) を返す。"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.46, 0.2, 0.14, 0.2],
                        specs=[[{}], [{}], [{}], [{"secondary_y": True}]])
    x, mat, used = an.curves_matrix(ts_all, sids, dbv_col, normalize)
    min_n = max(2, int(np.ceil(len(used) * 0.3)))  # 症例数が少ない終盤は省略

    def stats(col):
        xx, m, _ = an.curves_matrix(ts_all, sids, col, normalize)
        sm = an.summarize_curves(xx, m)
        sm.loc[sm.n < min_n, ["mean", "sd", "median", "q25", "q75"]] = np.nan
        return sm

    sm = stats(dbv_col)
    if show_lines:
        for row in mat:
            fig.add_trace(go.Scatter(x=x, y=row, mode="lines", line=dict(color="rgba(120,140,170,.22)", width=1),
                                     showlegend=False, hoverinfo="skip"), 1, 1)
    lo, hi = (sm["mean"] - sm["sd"], sm["mean"] + sm["sd"]) if band == "±SD" else (sm["q25"], sm["q75"])
    fig.add_trace(go.Scatter(x=sm.x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=sm.x, y=lo, fill="tonexty", fillcolor="rgba(31,95,191,.18)", line=dict(width=0),
                             name=f"バンド（{band}）", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=sm.x, y=sm["mean"], name=f"平均（{len(used)}症例）", line=dict(color="#1f5fbf", width=3),
                             customdata=sm.n, hovertemplate="%{y:.1f}%  (n=%{customdata})"), 1, 1)

    bp_all = an.all_bp_points(ts_all, sids, CRIT)
    bin_size = 10 if normalize else 30
    bs = an.bp_bin_stats(bp_all, normalize, bin_size)
    if not bp_all.empty:
        xb = bp_all.treat_min / bp_all.t_total * 100 if normalize else bp_all.treat_min
        hy = bp_all.hypo.astype(bool)
        fig.add_trace(go.Scatter(x=xb[hy], y=bp_all.dbv[hy], mode="markers", name="血圧低下の測定点",
                                 marker=dict(symbol="diamond", color="#d62728", size=10,
                                             line=dict(color="white", width=1)),
                                 hovertemplate="血圧低下  ΔBV %{y:.1f}%"), 1, 1)
        fig.add_trace(go.Scatter(x=xb[~hy], y=bp_all.sbp[~hy], mode="markers", name="SBP（各測定）",
                                 marker=dict(size=5, color="rgba(80,100,130,.45)"),
                                 hovertemplate="SBP %{y:.0f}"), 2, 1)
        fig.add_trace(go.Scatter(x=xb[hy], y=bp_all.sbp[hy], mode="markers", showlegend=False,
                                 marker=dict(size=7, color="#d62728"), hovertemplate="SBP %{y:.0f}（低下）"), 2, 1)
        fig.add_trace(go.Scatter(x=bs.x, y=bs.sbp_mean, mode="lines+markers", name="SBP 時間帯平均",
                                 line=dict(color="#d62728", width=2.4), hovertemplate="%{y:.0f}"), 2, 1)
        if CRIT.sbp_abs is not None:
            fig.add_hline(y=CRIT.sbp_abs, line=dict(color="rgba(214,39,40,.4)", dash="dot"), row=2, col=1)
        fig.add_trace(go.Bar(x=bs.x, y=bs.hypo_pct, width=bin_size * 0.85, name="血圧低下の頻度",
                             marker_color="#d62728", text=[f"{int(h)}/{int(n)}" for h, n in zip(bs.hypo_n, bs.n)],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{y:.0f}%（低下/測定）"), 3, 1)
        fig.update_yaxes(range=[0, max(10.0, float(bs.hypo_pct.max()) * 1.4 + 5)], row=3, col=1)
    uf = stats("ufr")
    fig.add_trace(go.Scatter(x=uf.x, y=uf["mean"], name="除水速度（平均）",
                             line=dict(color="#e08a00", width=2, shape="hv")), 4, 1, secondary_y=False)
    uv = stats("uf_vol")
    fig.add_trace(go.Scatter(x=uv.x, y=uv["mean"], name="除水量（平均）", line=dict(color="#6c757d", width=2)),
                  4, 1, secondary_y=True)
    fig.update_yaxes(title_text="ΔBV (%)", row=1, col=1)
    fig.update_yaxes(title_text="収縮期血圧", row=2, col=1)
    fig.update_yaxes(title_text="低下の頻度 (%)", row=3, col=1)
    fig.update_yaxes(title_text="除水速度 (L/h)", row=4, col=1, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="除水量 (L)", row=4, col=1, secondary_y=True, rangemode="tozero", showgrid=False)
    fig.update_xaxes(title_text="治療進行度 (%)" if normalize else "治療経過時間 (分)", row=4, col=1,
                     dtick=10 if normalize else 30)
    fig.update_layout(height=900, template="plotly_white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=1.07, x=0))
    return fig, bs, len(used)


def page_patient():
    st.header("③ 患者別")
    if sessions.empty:
        empty_state()
        return
    cur_pid, _, _ = current_selection()
    st.session_state["w_pid_p"] = cur_pid
    pid = st.selectbox("患者", ACTIVE_PIDS, key="w_pid_p", on_change=_on_pid_p,
                       format_func=lambda x: pid_label(x, patients))
    st.subheader(pid_label(pid, patients))
    dur_f, mode_f = group_filter("pt")
    sub = apply_filter(sessions[sessions.patient_id == pid], dur_f, mode_f).sort_values("session_dt")
    if sub.empty:
        st.info("この条件に合う症例がありません。")
        return
    ts_all = load_patient_ts(REV, pid)
    summ = load_patient_summary(REV, pid, CRIT, int(WINDOW)).sort_values("session_dt")
    summ = summ[summ.session_id.isin(sub.session_id)]
    st.write(f"{len(sub)} 症例  ({sub.session_dt.min():%Y-%m-%d} 〜 {sub.session_dt.max():%Y-%m-%d})　"
             + "／ ".join(f"{k} {v}" for k, v in sub.mode_label.value_counts().items()))

    tab_over, tab_int = st.tabs(["症例の重ね描き", "統合グラフ（全症例の平均）"])
    with tab_int:
        c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1.4])
        norm_i = c1.checkbox("治療時間を0–100%に正規化", False, key="int_norm")
        lines_i = c2.checkbox("各症例の曲線を薄く重ねる", True, key="int_lines")
        band_i = c3.radio("バンド", ["±SD", "四分位"], horizontal=True, key="int_band")
        col_i = c4.selectbox("BV指標", ["dbv_avg5", "dbv"], format_func=COLUMN_LABELS.get, key="int_col")
        figi, bs, n_used = integrated_figure(ts_all, sub.session_id.tolist(), norm_i, lines_i, band_i, col_i)
        st.plotly_chart(figi, use_container_width=True, config=PLOT_CONFIG)
        st.caption(f"{n_used} 症例を統合。症例数が全体の30%未満になる終盤の時間帯は平均を省略しています。"
                   "血圧低下の頻度 = その時間帯の血圧測定のうち低下と判定された割合（棒の上は 低下/測定 の件数、治療前値は除く）。")
        with st.expander("時間帯別の集計表"):
            tbl = bs.rename(columns={"x": "時間帯の中央", "n": "測定数", "sbp_mean": "SBP平均",
                                     "hypo_n": "低下数", "hypo_pct": "低下の頻度(%)"})
            st.dataframe(tbl[["時間帯の中央", "測定数", "SBP平均", "低下数", "低下の頻度(%)"]]
                         .style.format(precision=1, na_rep="–"), hide_index=True, width="stretch")
    with tab_over:
        norm = st.checkbox("治療時間を0–100%に正規化", False)
        fig = go.Figure()
        colors = px.colors.sequential.Viridis
        for i, s in enumerate(sub.itertuples()):
            g = an.dedupe_time(ts_all[ts_all.session_id == s.session_id])
            x = g.treat_sec / 60
            if norm:
                x = x / x.max() * 100
            c = colors[int(i / max(len(sub) - 1, 1) * (len(colors) - 1))]
            fig.add_trace(go.Scatter(x=x, y=g.dbv_avg5, name=f"{s.session_dt:%Y-%m-%d}", line=dict(color=c, width=2),
                                     hovertemplate="%{y:.1f}%"))
            b = an.bp_points(g, CRIT)
            b = b[b.hypo] if len(b) else b
            if len(b):
                bx = b.treat_min / (g.treat_sec.max() / 60) * 100 if norm else b.treat_min
                fig.add_trace(go.Scatter(x=bx, y=b.dbv, mode="markers", showlegend=False,
                                         marker=dict(symbol="diamond", color="#d62728", size=11,
                                                     line=dict(color="white", width=1)),
                                         hovertemplate=f"{s.session_dt:%m/%d} 血圧低下 ΔBV %{{y:.1f}}%"))
        fig.update_layout(height=480, template="plotly_white", hovermode="closest",
                          xaxis_title="治療進行度 (%)" if norm else "治療経過時間 (分)", yaxis_title="ΔBV 5分平均 (%)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.caption("◆ = その症例で血圧低下と判定された測定時点。症例の詳細は「② 症例解析」で確認できます。")

        L = summary_labels(int(WINDOW))
        cols = [c for c in ["min_dbv", "dbv_slope_per_h", "uf_total", "sbp_min"] if c in summ]
        if len(summ) and cols:
            trend = px.line(summ, x="session_dt", y=cols, markers=True, labels={"session_dt": "施行日", "value": "値"})
            trend.for_each_trace(lambda t: t.update(name=L.get(t.name, t.name)))
            trend.update_layout(height=340, template="plotly_white", legend_title="", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(trend, use_container_width=True)
            show = summ[["session_dt", *[c for c in L if c in summ]]].rename(columns={"session_dt": "施行日", **L})
            show["血圧低下"] = summ["hypo_final"].map({True: "あり", False: "なし"})
            st.dataframe(show.round(2), hide_index=True, width="stretch")


FILL = {"#d62728": "rgba(214,39,40,.12)", "#1f5fbf": "rgba(31,95,191,.12)"}


# ----------------------------------------------------------------- ④ compare
def page_compare():
    st.header("④ 症例間比較・統計")
    if sessions.empty:
        empty_state()
        return
    summ = load_summary(REV, CRIT, int(WINDOW))
    L = summary_labels(int(WINDOW))
    dur_f, mode_f = group_filter("cmp")
    summ = apply_filter(summ, dur_f, mode_f)
    if summ.empty:
        st.info("この条件に合う症例がありません。")
        return
    f1, f2 = st.columns([2, 2])
    pids = f1.multiselect("患者で絞り込み（空=全員）", ACTIVE_PIDS,
                          format_func=lambda x: pid_label(x, patients))
    dmin, dmax = summ.session_dt.min().date(), summ.session_dt.max().date()
    rng = f2.date_input("期間", (dmin, dmax), min_value=dmin, max_value=dmax)
    s = summ.copy()
    if pids:
        s = s[s.patient_id.isin(pids)]
    if isinstance(rng, tuple) and len(rng) == 2:
        s = s[(s.session_dt.dt.date >= rng[0]) & (s.session_dt.dt.date <= rng[1])]
    s["血圧低下"] = s.hypo_final.map({True: "あり", False: "なし"})
    st.write(f"対象 **{len(s)}** 症例（血圧低下あり {int((s.hypo_final == True).sum())} / なし {int((s.hypo_final == False).sum())}）")
    if s.empty:
        return

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["BV推移の比較", "散布図・相関", "群間比較", "一覧", "患者ごとの平均曲線",
                                          "治療時間群・モード別", "判定基準の比較"])
    with t1:
        c1, c2 = st.columns(2)
        norm = c1.checkbox("治療時間を0–100%に正規化", False, key="cmp_norm")
        col = c2.selectbox("項目", ["dbv_avg5", "dbv", "sbp", "ufr", "uf_vol"],
                           format_func=lambda c: COLUMN_LABELS.get(c, c))
        ts_all = load_all_ts(REV)
        fig = go.Figure()
        for lab, flag, color in (("血圧低下あり", True, "#d62728"), ("血圧低下なし", False, "#1f5fbf")):
            ids = s.loc[s.hypo_final == flag, "session_id"].tolist()
            mc, _ = curve_stats(REV, tuple(ids), col, norm)
            if mc.empty:
                continue
            mc = mc[mc.n >= max(2, int(len(ids) * 0.3))] if not norm else mc
            fig.add_trace(go.Scatter(x=np.r_[mc.x, mc.x[::-1]], y=np.r_[mc["mean"] + mc.sd, (mc["mean"] - mc.sd)[::-1]],
                                     fill="toself", fillcolor=FILL[color], line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mc.x, y=mc["mean"], name=f"{lab} (n={len(ids)})",
                                     line=dict(color=color, width=3)))
            fig.add_trace(go.Scatter(x=mc.x, y=mc["mean"] + mc.sd, line=dict(color=color, width=1, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mc.x, y=mc["mean"] - mc.sd, line=dict(color=color, width=1, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
        fig.update_layout(height=470, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="治療進行度 (%)" if norm else "治療経過時間 (分)",
                          yaxis_title=COLUMN_LABELS.get(col, col), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("実線 = 平均、点線 = ±1SD。非正規化表示では、症例数が全体の30%未満になる時間帯は省略しています。")
    with t2:
        if len(s) < 3:
            st.info(f"散布図・相関には3症例以上のデータが必要です（現在 {len(s)} 症例）")
        else:
            opts = [c for c in L if c in s.columns and s[c].notna().sum() > 1]
            a, b, c = st.columns(3)
            xv = a.selectbox("X軸", opts, index=opts.index("uf_total") if "uf_total" in opts else 0, format_func=L.get)
            yv = b.selectbox("Y軸", opts, index=opts.index("min_dbv") if "min_dbv" in opts else 0, format_func=L.get)
            cv = c.selectbox("色分け", ["血圧低下", "patient_id"], format_func=lambda x: {"血圧低下": "血圧低下の有無",
                                                                                      "patient_id": "患者"}[x])
            d = s.dropna(subset=[xv, yv])
            fig = px.scatter(d, x=xv, y=yv, color=cv, hover_data=["patient_id", "session_dt"],
                             labels={xv: L[xv], yv: L[yv]},
                             color_discrete_map={"あり": "#d62728", "なし": "#1f5fbf"})
            if len(d) >= 3 and d[xv].nunique() > 1:
                k, m = np.polyfit(d[xv], d[yv], 1)
                xs = np.array([d[xv].min(), d[xv].max()])
                fig.add_trace(go.Scatter(x=xs, y=k * xs + m, mode="lines", name="回帰直線",
                                         line=dict(color="gray", dash="dash")))
                r = np.corrcoef(d[xv], d[yv])[0, 1]
                msg = f"Pearson r = {r:.2f}（n={len(d)}）"
                try:
                    from scipy import stats
                    rho, p = stats.spearmanr(d[xv], d[yv])
                    msg += f" ／ Spearman ρ = {rho:.2f}, p = {p:.3g}"
                except Exception:  # noqa: BLE001
                    pass
                st.caption(msg)
            fig.update_layout(height=470, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("同一患者の複数症例は独立ではないため、相関のp値は参考値です（混合効果モデル等が必要）。")
    with t3:
        if len(s) < 3:
            st.info(f"群間比較には3症例以上のデータが必要です（現在 {len(s)} 症例）")
        else:
            opts = [c for c in L if c in s.columns and s[c].notna().sum() > 1]
            v = st.selectbox("比較する項目", opts, format_func=L.get,
                             index=opts.index("min_dbv") if "min_dbv" in opts else 0)
            d = s.dropna(subset=[v])
            fig = px.box(d, x="血圧低下", y=v, points="all", color="血圧低下", labels={v: L[v]},
                         color_discrete_map={"あり": "#d62728", "なし": "#1f5fbf"},
                         category_orders={"血圧低下": ["なし", "あり"]})
            fig.update_layout(height=420, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            g1, g0 = d.loc[d.hypo_final == True, v], d.loc[d.hypo_final == False, v]  # noqa: E712
            st.dataframe(pd.DataFrame({"群": ["血圧低下あり", "血圧低下なし"], "n": [len(g1), len(g0)],
                                       "平均": [g1.mean(), g0.mean()], "SD": [g1.std(), g0.std()],
                                       "中央値": [g1.median(), g0.median()]}).style.format(precision=2, na_rep="–"),
                         hide_index=True)
            if len(g1) >= 2 and len(g0) >= 2:
                try:
                    from scipy import stats
                    u = stats.mannwhitneyu(g1, g0, alternative="two-sided")
                    st.caption(f"Mann-Whitney U 検定: p = {u.pvalue:.3g}（参考値。患者内の反復測定は考慮していません）")
                except Exception:  # noqa: BLE001
                    pass
    with t4:
        show = s[["patient_id", "session_dt", *[c for c in L if c in s.columns], "血圧低下"]].rename(
            columns={"patient_id": "患者", "session_dt": "施行日", **L})
        show["患者"] = show["患者"].map(lambda x: pid_label(x, patients))
        st.dataframe(show.round(2), hide_index=True, width="stretch")

    with t5:
        c1, c2, c3 = st.columns(3)
        norm5 = c1.checkbox("治療時間を0–100%に正規化", False, key="pm_norm")
        min_s = c2.number_input("1患者あたりの最小症例数", 1, 50, 3, key="pm_min")
        col5 = c3.selectbox("項目", ["dbv_avg5", "dbv", "sbp", "ufr", "uf_vol"], key="pm_col",
                            format_func=lambda c: COLUMN_LABELS.get(c, c))
        ts_all = load_all_ts(REV)
        fig = go.Figure()
        for pid_, grp in s.groupby("patient_id"):
            ids = grp.session_id.tolist()
            if len(ids) < min_s:
                continue
            sm_, n_used = curve_stats(REV, tuple(ids), col5, norm5)
            sm_ = sm_[sm_.n >= max(2, int(np.ceil(n_used * 0.3)))]
            fig.add_trace(go.Scatter(x=sm_.x, y=sm_["mean"], name=f"{pid_label(pid_, patients)}（{n_used}症例）",
                                     mode="lines", line=dict(width=2)))
        sm_, n_used = curve_stats(REV, tuple(s.session_id.tolist()), col5, norm5)
        sm_ = sm_[sm_.n >= max(2, int(np.ceil(n_used * 0.3)))]
        fig.add_trace(go.Scatter(x=sm_.x, y=sm_["mean"], name=f"全体の平均（{n_used}症例）", mode="lines",
                                 line=dict(color="black", width=4, dash="dash")))
        fig.update_layout(height=520, template="plotly_white", hovermode="closest",
                          margin=dict(l=10, r=10, t=10, b=10), yaxis_title=COLUMN_LABELS.get(col5, col5),
                          xaxis_title="治療進行度 (%)" if norm5 else "治療経過時間 (分)",
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.caption("1本 = 1患者の全症例を平均した曲線。黒の破線 = 対象症例すべての平均。")
    with t6:
        strat = st.radio("層別", ["治療時間群（4時間群 / 5時間群）", "治療モード", "治療時間群 × モード"], horizontal=True,
                         key="gs_strat")
        s6 = s.copy()
        s6["群"] = (s6["dur_group"] if strat.startswith("治療時間群（") else
                   s6["mode_label"] if strat == "治療モード" else s6["dur_group"] + " / " + s6["mode_label"])
        min_n = st.number_input("1群あたりの最小症例数", 1, 100, 5, key="gs_minn")
        counts = s6["群"].value_counts()
        keep = counts[counts >= min_n].index.tolist()
        s6 = s6[s6["群"].isin(keep)]
        if len(keep) < 1:
            st.info("条件を満たす群がありません。最小症例数を下げてください。")
        else:
            palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark2
            g_order = sorted(keep)
            cA, cB = st.columns(2)
            norm6 = cA.checkbox("治療時間を0–100%に正規化", False, key="gs_norm")
            col6 = cB.selectbox("経時変化の項目", ["dbv_avg5", "dbv", "sbp", "ufr", "uf_vol"], key="gs_col",
                                format_func=lambda c: COLUMN_LABELS.get(c, c))
            ts_all = load_all_ts(REV)
            fig = go.Figure()
            for i, gname in enumerate(g_order):
                ids = s6.loc[s6["群"] == gname, "session_id"].tolist()
                sm_, n_used = curve_stats(REV, tuple(ids), col6, norm6)
                sm_ = sm_[sm_.n >= max(2, int(np.ceil(n_used * 0.3)))]
                color = palette[i % len(palette)]
                fig.add_trace(go.Scatter(x=sm_.x, y=sm_["mean"], name=f"{gname}（{n_used}症例）", mode="lines",
                                         line=dict(color=color, width=3)))
                se = sm_["sd"] / np.sqrt(sm_["n"])
                fig.add_trace(go.Scatter(x=np.r_[sm_.x, sm_.x[::-1]], y=np.r_[sm_["mean"] + se, (sm_["mean"] - se)[::-1]],
                                         fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                         showlegend=False, hoverinfo="skip"))
            fig.update_layout(height=430, template="plotly_white", hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10), yaxis_title=COLUMN_LABELS.get(col6, col6),
                              xaxis_title="治療進行度 (%)" if norm6 else "治療経過時間 (分)")
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption("実線 = 群の平均、帯 = 平均の標準誤差（±SE）。")

            # 血圧低下率(症例単位) と Wilson 95%区間
            rate = []
            for gname in g_order:
                d_ = s6[s6["群"] == gname]
                k_ = int((d_.hypo_final == True).sum())  # noqa: E712
                p_, lo_, hi_ = wilson(k_, len(d_))
                rate.append({"群": gname, "症例数": len(d_), "低下あり": k_, "低下率(%)": p_, "lo": lo_, "hi": hi_})
            rate = pd.DataFrame(rate)
            figb = go.Figure(go.Bar(
                x=rate["群"], y=rate["低下率(%)"], marker_color=[palette[i % len(palette)] for i in range(len(rate))],
                error_y=dict(type="data", symmetric=False, array=rate["hi"] - rate["低下率(%)"],
                             arrayminus=rate["低下率(%)"] - rate["lo"]),
                text=[f"{a}/{b}" for a, b in zip(rate["低下あり"], rate["症例数"])], textposition="outside"))
            figb.update_layout(height=340, template="plotly_white", yaxis_title="血圧低下のあった症例の割合 (%)",
                               margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, min(105, float(rate["hi"].max()) + 15)])
            st.plotly_chart(figb, use_container_width=True, config=PLOT_CONFIG)
            st.caption("棒の上 = 低下あり症例 / 症例数、ひげ = 95%信頼区間（Wilson）。判定基準はサイドバーの設定に従います。")

            # 指標の群間比較
            metrics = [c for c in L if c in s6.columns and s6[c].notna().sum() > 1]
            rows = []
            for c in metrics:
                row_ = {"指標": L[c]}
                vals = []
                for gname in g_order:
                    v = s6.loc[s6["群"] == gname, c].dropna()
                    vals.append(v)
                    row_[gname] = f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else "–"
                try:
                    from scipy import stats
                    use = [v for v in vals if len(v) >= 2]
                    if len(use) == 2:
                        row_["p値"] = stats.mannwhitneyu(use[0], use[1], alternative="two-sided").pvalue
                    elif len(use) > 2:
                        row_["p値"] = stats.kruskal(*use).pvalue
                except Exception:  # noqa: BLE001
                    pass
                rows.append(row_)
            st.dataframe(pd.DataFrame(rows).style.format({"p値": "{:.3g}"}, na_rep="–"), hide_index=True,
                         width="stretch")
            st.caption("平均 ± SD。p値は2群ならMann-Whitney、3群以上ならKruskal-Wallis（参考値。同一患者の反復測定は考慮していません）。")
            vsel = st.selectbox("箱ひげ図で見る指標", metrics, format_func=L.get,
                                index=metrics.index("min_dbv") if "min_dbv" in metrics else 0, key="gs_box")
            figx = px.box(s6.dropna(subset=[vsel]), x="群", y=vsel, color="群", points="all", labels={vsel: L[vsel]},
                          category_orders={"群": g_order}, color_discrete_sequence=palette)
            figx.update_layout(height=380, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figx, use_container_width=True, config=PLOT_CONFIG)
    with t7:
        st.caption("判定基準によって「血圧低下あり」と判定される割合がどれだけ変わるかの比較です（全症例が対象。上の絞り込みは反映しません）。")
        if st.toggle("比較表を作成する（全症例を基準ごとに再判定するため、初回は十数秒かかります）", False, key="pt_on"):
            st.dataframe(load_preset_table(REV, int(WINDOW)).style.format(precision=1, na_rep="–"),
                         hide_index=True, width="stretch")
            st.caption("研究で使う基準を決める参考にしてください。サイドバーの「判定基準」で選んだ基準が、画面全体の集計に使われます。")

# ----------------------------------------------------------------- ⑤ hypotension x ΔBV
@st.cache_data(show_spinner="低下直前のΔBV推移を集計中…")
def load_aligned(rev: int, crit: an.HypoCriteria, ids: tuple, pre: int):
    return an.aligned_traces(load_all_ts(rev), ids, crit, pre, "dbv")


@st.cache_data(show_spinner="95%区間を計算中…")
def load_auc_ci(rev: int, crit: an.HypoCriteria, window: int, ids: tuple, feature: str):
    f = load_features(rev, crit, window)
    return an.auc_cluster_ci(f[f.session_id.isin(ids)], feature)


def page_hypo():
    st.header("⑤ 血圧低下とΔBV")
    if sessions.empty:
        empty_state()
        return
    st.caption(f"血圧低下の判定: {CRIT.describe()}（サイドバーで変更）。血圧測定点（治療前値を除く）を単位に解析します。")
    dur_f, mode_f = group_filter("hb")
    sub = apply_filter(sessions, dur_f, mode_f)
    ids = tuple(int(i) for i in sub.session_id)
    feats = load_features(REV, CRIT, int(WINDOW))
    feats = feats[feats.session_id.isin(ids)].merge(
        sessions[["session_id", "patient_id", "dur_group", "mode_label"]], on="session_id", how="left")
    if feats.empty:
        st.info("この条件に合う血圧測定点がありません。")
        return
    n_h = int(feats.hypo.sum())
    k = st.columns(4)
    k[0].metric("症例数", feats.session_id.nunique())
    k[1].metric("血圧測定点", len(feats))
    k[2].metric("低下と判定", n_h)
    k[3].metric("測定点あたり低下率", f"{n_h / len(feats) * 100:.1f} %")
    if n_h < 5 or len(feats) - n_h < 5:
        st.warning("低下あり／なしの測定点が少なすぎて解析できません。絞り込みや判定基準を見直してください。")
        return

    W = int(WINDOW)
    FEATURES = {
        "dbv": "ΔBV（測定時点）(%)", "dbv_avg5": "ΔBV 5分平均(%)", "dbv_change": f"ΔBV変化/直前{W}分(%)",
        "dbv_slope": f"ΔBV傾き/直前{W}分(%/h)", "dbv_margin": "ΔBV − リファレンス下限(%)",
        "uf_vol": "累積除水量(L)", "ufr": "除水速度(L/h)", "treat_min": "経過時間(分)",
    }
    summ_h = load_summary(REV, CRIT, W)
    summ_h = summ_h[summ_h.session_id.isin(ids)].copy()
    summ_h["is5h"] = (summ_h["dur_group"] == "5時間群").astype(float)
    tab_d, tab_e, tab_f, tab_a, tab_b, tab_c, tab_g = st.tabs(
        ["4時間群 vs 5時間群（患者内）", "調整解析", "除水量の目安（概算）", "低下直前のΔBV推移", "予測性（ROC）",
         "ΔBV水準ごとの低下率", "低下後の除水停止"])

    with tab_a:
        c1, c2 = st.columns(2)
        pre = c1.selectbox("血圧測定の何分前から見るか", [30, 60, 90, 120], index=1, key="hb_pre")
        mode = c2.radio("表示", ["ΔBVの値", f"{pre}分前からの変化量"], horizontal=True, key="hb_disp")
        hyp, non, offs = load_aligned(REV, CRIT, ids, int(pre))
        if len(hyp) < 3 or len(non) < 3:
            st.info("該当する測定点が少なすぎます（測定が治療開始から十分に経っているものだけが対象です）。")
        else:
            if mode != "ΔBVの値":
                hyp, non = hyp - hyp[:, :1], non - non[:, :1]
            fig = go.Figure()
            for arr, name, color in ((non, "低下なし", "#1f5fbf"), (hyp, "低下あり", "#d62728")):
                m, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr))
                fig.add_trace(go.Scatter(x=np.r_[offs, offs[::-1]], y=np.r_[m + 1.96 * se, (m - 1.96 * se)[::-1]],
                                         fill="toself", fillcolor=color, opacity=0.18, line=dict(width=0),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=offs, y=m, name=f"{name}（n={len(arr)}）", line=dict(color=color, width=3)))
            fig.add_vline(x=0, line=dict(color="gray", dash="dot"), annotation_text="血圧測定", annotation_position="top")
            fig.update_layout(height=430, template="plotly_white", hovermode="x unified",
                              margin=dict(l=10, r=10, t=30, b=10), xaxis_title="血圧測定からの時間（分）",
                              yaxis_title="ΔBV (%)" if mode == "ΔBVの値" else "ΔBVの変化量 (%)")
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            from scipy import stats
            last = hyp[:, -1], non[:, -1]
            pv = stats.mannwhitneyu(*last, alternative="two-sided").pvalue
            st.caption(f"線 = 平均、帯 = 平均の95%区間。血圧測定時点の{'ΔBV' if mode == 'ΔBVの値' else '変化量'}: "
                       f"低下あり {last[0].mean():.1f}% / 低下なし {last[1].mean():.1f}%（Mann-Whitney p = {pv:.3g}）。"
                       "測定点を単位とするため、同一症例の複数の測定点は独立ではありません（参考値）。")

    with tab_b:
        use_ci = st.checkbox("AUCの95%区間を計算する（症例単位のブートストラップ。数十秒かかります）", False, key="hb_ci")
        rows = []
        for key, label in FEATURES.items():
            r = an.roc_analysis(feats["hypo"].to_numpy(), feats[key].to_numpy())
            if not r:
                continue
            row = {"指標": label, "AUC": r["auc"],
                   "低下しやすい方向": "高いほど" if r["direction"] == ">=" else "低いほど",
                   "しきい値": r["cutoff"], "感度": r["sens"], "特異度": r["spec"],
                   "低下あり(n)": r["n_pos"], "低下なし(n)": r["n_neg"], "_key": key}
            if use_ci:
                lo, hi = load_auc_ci(REV, CRIT, W, ids, key)
                row["AUC 95%区間"] = f"{lo:.2f}–{hi:.2f}" if not np.isnan(lo) else "–"
            rows.append(row)
        tbl = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        st.dataframe(tbl.drop(columns="_key").style.format({"AUC": "{:.3f}", "しきい値": "{:.2f}", "感度": "{:.0%}", "特異度": "{:.0%}"}),
                     hide_index=True, width="stretch")
        st.caption("AUC 0.5 = 予測できない、1.0 = 完全に予測。しきい値は感度+特異度が最大になる点（Youden指数）。"
                   "測定点を単位とした探索的な解析です。")
        labels = {r["_key"]: r["指標"] for r in rows}
        pick = st.selectbox("詳しく見る指標", list(labels), format_func=labels.get, key="hb_pick")
        r = an.roc_analysis(feats["hypo"].to_numpy(), feats[pick].to_numpy())
        cA, cB = st.columns(2)
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"AUC {r['auc']:.3f}", line=dict(color="#d62728", width=3)))
        figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dot"), showlegend=False))
        figr.update_layout(height=360, template="plotly_white", xaxis_title="1 − 特異度", yaxis_title="感度",
                           margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.6, y=0.1))
        cA.plotly_chart(figr, use_container_width=True, config=PLOT_CONFIG)
        d_ = feats.dropna(subset=[pick]).assign(判定=lambda x: x.hypo.map({True: "低下あり", False: "低下なし"}))
        figd = px.box(d_, x="判定", y=pick, color="判定", points=False, labels={pick: labels[pick]},
                      color_discrete_map={"低下あり": "#d62728", "低下なし": "#1f5fbf"})
        figd.add_hline(y=r["cutoff"], line=dict(color="gray", dash="dash"), annotation_text="しきい値")
        figd.update_layout(height=360, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        cB.plotly_chart(figd, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("**層別のAUC**（選択した指標）")
        by = st.radio("層別", ["治療時間群", "治療モード"], horizontal=True, key="hb_by")
        col_by = "dur_group" if by == "治療時間群" else "mode_label"
        srows = []
        for gname, g in feats.groupby(col_by):
            rr = an.roc_analysis(g["hypo"].to_numpy(), g[pick].to_numpy())
            srows.append({by: gname, "測定点": len(g), "低下あり": int(g.hypo.sum()),
                          "低下率(%)": g.hypo.mean() * 100,
                          "AUC": rr["auc"] if rr else np.nan,
                          "しきい値": rr["cutoff"] if rr else np.nan, "感度": rr["sens"] if rr else np.nan,
                          "特異度": rr["spec"] if rr else np.nan})
        st.dataframe(pd.DataFrame(srows).style.format({"低下率(%)": "{:.1f}", "AUC": "{:.3f}", "しきい値": "{:.2f}",
                                                       "感度": "{:.0%}", "特異度": "{:.0%}"}, na_rep="–"),
                     hide_index=True, width="stretch")

    # ------------------------------------------------------------ 患者内比較
    with tab_d:
        st.caption("同じ患者の中で4時間群と5時間群を比べます。患者ごとの血圧の下がりやすさの違いによる見かけの差を除くためです。"
                   "各群が一定数以上ある患者だけが対象です。")
        min_each = st.number_input("各群の最小症例数（患者ごと）", 1, 30, 3, key="pw_min")
        pp = sm_.paired_by_patient(summ_h, min_each=int(min_each))
        a4, b5 = summ_h[summ_h.dur_group == "4時間群"], summ_h[summ_h.dur_group == "5時間群"]
        if pp.empty or len(pp) < 2 or a4.empty or b5.empty:
            st.info("両方の群を十分に持つ患者が2名未満です。最小症例数を下げるか、上の絞り込みを見直してください。")
        else:
            from scipy import stats as sstats
            mh = sm_.mantel_haenszel(summ_h)
            diff = pp["差(b−a, %pt)"]
            fisher = sstats.fisher_exact([[int(a4.hypo_final.sum()), int((~a4.hypo_final.astype(bool)).sum())],
                                          [int(b5.hypo_final.sum()), int((~b5.hypo_final.astype(bool)).sum())]])[1]
            k = st.columns(4)
            k[0].metric("対象患者", f"{len(pp)} 名")
            k[1].metric("低下率の差（5h−4h）の平均", f"{diff.mean():+.1f} %pt")
            if mh:
                k[2].metric("患者で層別したオッズ比", f"{mh['or']:.2f}", f"95%区間 {mh['lo']:.2f}–{mh['hi']:.2f}", delta_color="off",
                            help="Mantel-Haenszel。1より小さければ5時間群で低下が少ない。")
                k[3].metric("p値", f"{mh['p']:.3g}")
            fig = go.Figure()
            for r in pp.itertuples(index=False):
                lab = pid_label(r.patient_id, patients)
                fig.add_trace(go.Scatter(x=["4時間群", "5時間群"], y=[r[3], r[6]], mode="lines+markers", name=lab,
                                         line=dict(width=2), marker=dict(size=9)))
            fig.update_layout(height=420, template="plotly_white", yaxis_title="血圧低下のあった症例の割合 (%)",
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption(f"1本 = 1患者。参考: 患者を区別しない単純な比較は 4時間群 {a4.hypo_final.mean() * 100:.1f}%（{len(a4)}症例） "
                       f"vs 5時間群 {b5.hypo_final.mean() * 100:.1f}%（{len(b5)}症例）、Fisher p = {fisher:.3g}。")
            show = pp.copy()
            show["patient_id"] = show["patient_id"].map(lambda x: pid_label(x, patients))
            st.dataframe(show.rename(columns={"patient_id": "患者"}).round(1), hide_index=True, width="stretch")
            st.markdown("**判定基準を変えた場合（感度分析）**")
            pf = load_preset_flags(REV)
            base = summ_h[["session_id", "patient_id", "dur_group"]]
            rows = []
            for name in an.PRESETS:
                d_ = base.assign(flag=base.session_id.map(pf[name]).astype(float))
                a_, b_ = d_[d_.dur_group == "4時間群"], d_[d_.dur_group == "5時間群"]
                m_ = sm_.mantel_haenszel(d_, flag_col="flag")
                rows.append({"判定基準": name, "4時間群 低下率(%)": a_.flag.mean() * 100, "5時間群 低下率(%)": b_.flag.mean() * 100,
                             "患者層別OR": m_["or"] if m_ else np.nan,
                             "95%区間": f"{m_['lo']:.2f}–{m_['hi']:.2f}" if m_ else "–", "p値": m_["p"] if m_ else np.nan})
            st.dataframe(pd.DataFrame(rows).style.format({"4時間群 低下率(%)": "{:.1f}", "5時間群 低下率(%)": "{:.1f}",
                                                         "患者層別OR": "{:.2f}", "p値": "{:.3g}"}, na_rep="–"),
                         hide_index=True, width="stretch")
            st.caption("どの判定基準でも結論が同じかを確認する表です。観察データの探索的な解析で、臨床上の指示を示すものではありません。")

    # ------------------------------------------------------------ 調整解析(GEE)
    with tab_e:
        st.caption("患者を単位とするGEEロジスティック回帰（同じ患者の症例は似るため、その相関を考慮）。"
                   "目的変数は「その症例で血圧低下があったか」です。")
        x = summ_h[summ_h.dur_group.isin(["4時間群", "5時間群"])]
        labels_p = {"is5h": "5時間群（4時間群と比較）", "uf_total": "除水量（1Lあたり）", "uf_rate_mean": "平均除水速度（1 L/hあたり）",
                    "sbp_base": "開始時SBP（1mmHgあたり）", "min_dbv": "最小ΔBV（1%あたり）", "treat_min": "治療時間（1分あたり）"}
        c1, c2 = st.columns([3, 1])
        sel = c1.multiselect("説明変数", list(labels_p), default=["is5h", "uf_total", "sbp_base"], format_func=labels_p.get, key="gee_sel")
        use_mode = c2.checkbox("治療モードで調整", True, key="gee_mode")
        if {"uf_total", "uf_rate_mean"} <= set(sel) or {"is5h", "treat_min"} <= set(sel):
            st.warning("除水量 = 除水速度 × 治療時間 の関係があるため、これらを同時に入れると係数が不安定になります。")
        if x.empty or not sel:
            st.info("説明変数を選んでください。")
        else:
            try:
                g = sm_.gee_logit(x, "hypo_final", sel, ["mode_label"] if use_mode else [])
            except Exception as e:  # noqa: BLE001
                g = None
                st.error(f"モデルを推定できませんでした: {e}")
            if g is not None:
                g["項目"] = g["項目"].map(lambda t: labels_p.get(t, t.replace("C(mode_label)[T.", "治療モード: ").rstrip("]") + "（HDと比較）"))
                st.caption(f"{g.attrs['n']} 症例 / {g.attrs['n_clusters']} 患者")
                fig = go.Figure(go.Scatter(x=g["OR"], y=g["項目"], mode="markers",
                                           error_x=dict(type="data", symmetric=False, array=g["上限"] - g["OR"],
                                                        arrayminus=g["OR"] - g["下限"]),
                                           marker=dict(size=9, color="#1f5fbf")))
                fig.add_vline(x=1, line=dict(color="gray", dash="dot"))
                fig.update_xaxes(type="log", title="オッズ比（対数軸、1より大きい = 低下しやすい）")
                fig.update_layout(height=60 + 48 * len(g), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                st.dataframe(g.style.format({"OR": "{:.2f}", "下限": "{:.2f}", "上限": "{:.2f}", "p値": "{:.3g}"}),
                             hide_index=True, width="stretch")
                st.caption("開始時SBPが高いほど『開始時から20mmHg以上低下』と判定されやすくなるため、この判定基準では開始時SBPの調整が重要です。"
                           "観察データの探索的解析で、因果を示すものではありません。")

    # ------------------------------------------------------------ 除水量の目安
    with tab_f:
        st.caption("過去の症例から、血圧低下の確率が目標以下に収まる除水量を概算します（体重・血圧薬・体液量などは考慮しない参考値）。")
        c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.4])
        scope = c1.selectbox("対象", ["全患者（まとめて）", *ACTIVE_PIDS], key="ug_scope",
                             format_func=lambda v: v if v.startswith("全患者") else pid_label(v, patients))
        dsel = c2.selectbox("治療時間群", ["4時間群", "5時間群", "両方"], key="ug_dur")
        xmode = c3.radio("指標", ["総除水量 (L)", "平均除水速度 (L/h)"], key="ug_x")
        target = c4.slider("許容する血圧低下の確率 (%)", 5, 60, 20, 5, key="ug_t") / 100
        xcol = "uf_total" if xmode.startswith("総") else "uf_rate_mean"
        d = summ_h.copy()
        if not scope.startswith("全患者"):
            d = d[d.patient_id == scope]
        if dsel != "両方":
            d = d[d.dur_group == dsel]
        d = d.dropna(subset=[xcol, "hypo_final"])
        d["y"] = d["hypo_final"].astype(float)
        fit = sm_.fit_logistic(d[xcol].to_numpy(float), d["y"].to_numpy(float))
        if fit is None:
            st.info("この条件では症例数（または低下あり/なしの数）が足りず、推定できません。対象を広げてください。")
        else:
            star = sm_.guideline(fit, target)
            cluster = "session_id" if not scope.startswith("全患者") else "patient_id"
            lo, hi, n_ok = sm_.guideline_ci(d, xcol, "y", target, cluster, n_boot=200)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
            ps = 1 / (1 + np.exp(-(fit["a"] + fit["b"] * xs)))
            unit = "L" if xcol == "uf_total" else "L/h"
            lo_x, hi_x = float(d[xcol].min()), float(d[xcol].max())
            ok_star = (not np.isnan(star)) and lo_x <= star <= hi_x
            k = st.columns(4)
            k[0].metric("症例数", f"{fit['n']}", f"低下あり {fit['n_pos']}（{fit['n_pos'] / fit['n'] * 100:.0f}%）", delta_color="off")
            k[1].metric(f"目安（低下確率 {target * 100:.0f}% 以下）", f"約 {star:.2f} {unit}" if ok_star else "決められません",
                        f"95%区間 {lo:.2f}–{hi:.2f}" if ok_star and not np.isnan(lo) else None, delta_color="off")
            nonh = d[d.y == 0][xcol]
            k[2].metric("低下なしの症例の中央値", f"{nonh.median():.2f} {unit}" if len(nonh) else "–",
                        f"75%点 {nonh.quantile(.75):.2f}" if len(nonh) else None, delta_color="off")
            k[3].metric("実績の平均", f"{d[xcol].mean():.2f} {unit}", f"傾きのp = {fit['p_slope']:.3g}", delta_color="off")
            fig = go.Figure()
            try:
                bins = pd.qcut(d[xcol], min(6, max(3, len(d) // 25)), duplicates="drop")
                bx = d.groupby(bins, observed=True).agg(x=(xcol, "mean"), p=("y", "mean"), n=("y", "size"))
                _t = []
                for iv, g_ in d.groupby(bins, observed=True):
                    p_, l_, h_ = wilson(int(g_.y.sum()), len(g_))
                    _t.append({f"{xmode} の区分": f"{iv.left:.2f} 〜 {iv.right:.2f}", "症例数": len(g_), "低下あり": int(g_.y.sum()),
                               "低下率(%)": p_, "95%区間": f"{l_:.0f}–{h_:.0f}%"})
                bin_table = pd.DataFrame(_t)
                fig.add_trace(go.Scatter(x=bx.x, y=bx.p * 100, mode="markers", name="実測（区分ごと）",
                                         marker=dict(size=np.sqrt(bx.n) * 3 + 6, color="rgba(31,95,191,.6)"),
                                         text=[f"n={n}" for n in bx.n], hovertemplate="%{x:.2f}: %{y:.0f}% (%{text})"))
            except ValueError:
                pass
            fig.add_trace(go.Scatter(x=xs, y=ps * 100, mode="lines", name="推定（ロジスティック）", line=dict(color="#d62728", width=3)))
            fig.add_hline(y=target * 100, line=dict(color="gray", dash="dash"), annotation_text=f"許容 {target * 100:.0f}%")
            if ok_star:
                fig.add_vline(x=star, line=dict(color="#2a9d8f", dash="dot"), annotation_text="目安")
            fig.update_layout(height=400, template="plotly_white", xaxis_title=xmode, yaxis_title="血圧低下の確率 (%)",
                              yaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            if "bin_table" in locals():
                st.markdown("**区分ごとの実測の低下率**（推定に頼らず、実際のデータをそのまま見る表）")
                st.dataframe(bin_table.style.format({"低下率(%)": "{:.0f}"}), hide_index=True, width="stretch")
            if fit["b"] <= 0 or fit["p_slope"] > 0.1:
                st.warning("この条件では、除水量（速度）と血圧低下の関連が明確ではありません。目安は不安定です。")
            elif not ok_star:
                st.warning(f"許容確率 {target * 100:.0f}% に対応する値が観測範囲（{lo_x:.2f}〜{hi_x:.2f} {unit}）の外になります。"
                           "この判定基準では、除水量が少なくても低下の確率が許容値を超えている可能性があります。"
                           "許容確率を上げるか、判定基準（サイドバー）を厳しくして（例: SBP<90のみ）ください。")
            if scope.startswith("全患者"):
                st.markdown("**患者ごとの目安**（各患者の症例だけで推定。症例が少ない患者は決められません）")
                rows = []
                for pid_, gp in d.groupby("patient_id"):
                    f_ = sm_.fit_logistic(gp[xcol].to_numpy(float), gp["y"].to_numpy(float))
                    st_ = sm_.guideline(f_, target) if f_ else np.nan
                    valid = f_ and not np.isnan(st_) and f_["p_slope"] < 0.1 and gp[xcol].min() <= st_ <= gp[xcol].max()
                    rows.append({"患者": pid_label(pid_, patients), "症例数": len(gp), "低下率(%)": gp.y.mean() * 100,
                                 f"実績の平均 ({unit})": gp[xcol].mean(), f"目安 ({unit})": st_ if valid else np.nan,
                                 "判定": "目安あり" if valid else "決められない"})
                st.dataframe(pd.DataFrame(rows).style.format({"低下率(%)": "{:.0f}", f"実績の平均 ({unit})": "{:.2f}",
                                                           f"目安 ({unit})": "{:.2f}"}, na_rep="–"),
                             hide_index=True, width="stretch")
            st.caption("過去の治療で実際に行われた除水量と血圧低下の関係から求めた統計的な概算です。除水量は医師が患者の状態に応じて決めているため"
                       "（耐えられる患者に多く除水している可能性）、因果的な安全量ではありません。臨床上の指示・処方には使用せず、研究上の参考値としてください。")

    with tab_c:
        key = st.selectbox("指標", list(FEATURES), format_func=FEATURES.get, key="hb_c_key")
        nb = st.slider("区分数（分位数で等分）", 3, 10, 5, key="hb_c_nb")
        d_ = feats.dropna(subset=[key]).copy()
        try:
            d_["区分"] = pd.qcut(d_[key], nb, duplicates="drop")
        except ValueError:
            st.info("この指標は値の種類が少なく、区分できません。")
            return
        rows = []
        for iv, g in d_.groupby("区分", observed=True):
            p_, lo_, hi_ = wilson(int(g.hypo.sum()), len(g))
            rows.append({"区分": f"{iv.left:.2f} 〜 {iv.right:.2f}", "測定点": len(g), "低下あり": int(g.hypo.sum()),
                         "低下率(%)": p_, "lo": lo_, "hi": hi_})
        rt = pd.DataFrame(rows)
        figc = go.Figure(go.Bar(x=rt["区分"], y=rt["低下率(%)"], marker_color="#d62728",
                                error_y=dict(type="data", symmetric=False, array=rt["hi"] - rt["低下率(%)"],
                                             arrayminus=rt["低下率(%)"] - rt["lo"]),
                                text=[f"{a}/{b}" for a, b in zip(rt["低下あり"], rt["測定点"])], textposition="outside"))
        figc.update_layout(height=380, template="plotly_white", xaxis_title=FEATURES[key], yaxis_title="血圧低下の割合 (%)",
                           margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, min(105, float(rt["hi"].max()) + 12)])
        st.plotly_chart(figc, use_container_width=True, config=PLOT_CONFIG)
        st.caption("横軸は測定点を指標の大きさで等分した区分、棒の上は 低下/測定点 の件数、ひげは95%信頼区間（Wilson）。")


    with tab_g:
        win = st.selectbox("血圧測定後、何分以内の除水停止を数えるか", [15, 30, 60], index=1, key="us_win")
        u = load_uf_stop(REV, CRIT, ids, int(win))
        if u.empty:
            st.info("該当する測定点がありません。")
        else:
            from scipy import stats as sstats
            t = u.groupby("hypo")["uf_stop"].agg(["sum", "size"]).reindex([False, True]).fillna(0)
            rates = [wilson(int(t.loc[h, "sum"]), int(t.loc[h, "size"])) for h in (False, True)]
            fig = go.Figure(go.Bar(x=["低下なし", "低下あり"], y=[r[0] for r in rates], marker_color=["#1f5fbf", "#d62728"],
                                   error_y=dict(type="data", symmetric=False, array=[r[2] - r[0] for r in rates],
                                                arrayminus=[r[0] - r[1] for r in rates]),
                                   text=[f"{int(t.loc[h, 'sum'])}/{int(t.loc[h, 'size'])}" for h in (False, True)],
                                   textposition="outside"))
            fig.update_layout(height=340, template="plotly_white", yaxis_title=f"測定後{win}分以内に除水停止があった割合 (%)",
                              margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            p = sstats.fisher_exact([[t.loc[True, "sum"], t.loc[True, "size"] - t.loc[True, "sum"]],
                                     [t.loc[False, "sum"], t.loc[False, "size"] - t.loc[False, "sum"]]])[1]
            st.caption(f"Fisher p = {p:.3g}。除水停止は血圧低下への対応以外（回路操作・測定など）でも発生するため、"
                       "臨床記録（対応）の入力と合わせて評価してください。")

# ----------------------------------------------------------------- ⑥ BV-UFC
@st.cache_resource(show_spinner="ΔBVの経過を準備中…", max_entries=1)
def load_trajs(rev: int) -> dict:
    return ufc.trajectories(load_all_ts(rev))


@st.cache_resource(show_spinner="全患者のモデルを作成中…", max_entries=1)
def load_pooled(rev: int):
    return ufc.fit_pooled(load_trajs(rev))


@st.cache_data(show_spinner=False)
def load_ufc_used(rev: int) -> pd.Series:
    return ufc.ufc_used(load_all_ts(rev))


def patient_sids(pid: str) -> list[int]:
    return [int(x) for x in sessions[sessions.patient_id == pid].sort_values("session_dt").session_id]


@st.cache_resource(show_spinner="この患者のモデルを作成中…", max_entries=64)
def load_patient_model(rev: int, pid: str):
    pooled = load_pooled(rev)
    return ufc.fit_response(load_trajs(rev), patient_sids(pid), pooled) if pooled else None


@st.cache_resource(show_spinner=False, max_entries=64)
def load_day_models(rev: int, pid: str):
    return ufc.day_models(load_trajs(rev), load_patient_model(rev, pid), patient_sids(pid))


@st.cache_data(show_spinner="過去の症例で予測を検証中…")
def load_backtest(rev: int, pid: str) -> pd.DataFrame:
    return ufc.backtest(load_trajs(rev), patient_sids(pid), load_pooled(rev), n_last=15)


@st.cache_data(show_spinner="血圧低下との関係を推定中…")
def load_risk(rev: int, crit: an.HypoCriteria, window: int, pid: str):
    return ufc.risk_fit(load_features(rev, crit, window), patient_sids(pid))


@st.cache_data(show_spinner="ステップ別に集計中…")
def load_step_table(rev: int, crit: an.HypoCriteria, ids: tuple) -> pd.DataFrame:
    return ufc.step_table(load_all_ts(rev), ids, crit)


@st.cache_data(show_spinner=False)
def load_idx_ach(rev: int, ids: tuple) -> pd.DataFrame:
    return ufc.index_achievement(load_all_ts(rev), ids)


@st.cache_data(show_spinner=False)
def load_shapes(rev: int) -> pd.DataFrame:
    return ufc.session_shapes(load_trajs(rev))


@st.cache_data(show_spinner=False)
def load_ufc_saved(rev: int, pid: str) -> pd.DataFrame:
    return db.ufc_settings_df(pid)


# 装置画面と同じ項目: 項目 -> (ラベル, 最小, 最大, 刻み, 表示形式)
UFC_FIELDS = {
    "total_min": ("治療時間 (分)", 60.0, 480.0, 5.0, "%.0f"),
    "target_uf": ("目標除水量 (L)", 0.0, 8.0, 0.1, "%.2f"),
    "start_min": ("時間 (分)", 0.0, 60.0, 1.0, "%.0f"),
    "start_mult": ("倍率 (×)", 0.0, 3.0, 0.05, "%.2f"),
    "const_min": ("時間 (分)", 0.0, 240.0, 5.0, "%.0f"),
    "const_mult": ("倍率 (×)", 0.0, 3.0, 0.05, "%.2f"),
    "end_min": ("時間 (分)", 0.0, 120.0, 5.0, "%.0f"),
    "ufr_max": ("上限 (L/h)", 0.0, 4.0, 0.05, "%.2f"),
    "ufr_min": ("下限 (L/h)", 0.0, 4.0, 0.05, "%.2f"),
    "start_dbv": ("開始時ΔBV (%)", -30.0, 10.0, 0.5, "%.1f"),
    "idx1": ("指数1 (%)", 0.0, 100.0, 1.0, "%.0f"),
    "idx2": ("指数2 (%)", 0.0, 100.0, 1.0, "%.0f"),
    "idx3": ("指数3 (%)", 0.0, 100.0, 1.0, "%.0f"),
    "end_dbv": ("終了時ΔBV (%)", -40.0, 0.0, 0.5, "%.1f"),
}
PHASE_COLORS = {"開始": "rgba(80,190,230,.16)", "定倍除水": "rgba(230,90,200,.12)",
                "UFC期間": "rgba(90,180,90,.12)", "終了前": "rgba(240,150,60,.16)"}


def _ufc_from_csv(pid: str) -> ufc.UFCSettings:
    """既定値に、その患者の最新症例のCSVの設定値(治療時間・目標除水量・開始/定倍/終了前)を上書き。"""
    sub = sessions[sessions.patient_id == pid].sort_values("session_dt")
    base = asdict(ufc.UFCSettings())
    if len(sub):
        base.update(ufc.planned_from_meta(sub.header_json.iloc[-1]))
    return ufc.UFCSettings.from_dict(base)


def _ufc_put(s: ufc.UFCSettings) -> None:
    for k, v in asdict(s).items():
        st.session_state[f"ufc_{k}"] = float(v) if k in UFC_FIELDS else v


def _ufc_get() -> ufc.UFCSettings:
    d = {k: float(st.session_state[f"ufc_{k}"]) for k in UFC_FIELDS}
    d["placement"] = st.session_state.get("ufc_placement", "whole")
    return ufc.UFCSettings.from_dict(d)


def _ufc_load_saved(pid: str) -> None:
    saved = load_ufc_saved(REV, pid)
    _ufc_put(ufc.UFCSettings.from_json(saved.settings_json.iloc[0]) if len(saved) else _ufc_from_csv(pid))


def _ufc_load_one(settings_json: str) -> None:
    _ufc_put(ufc.UFCSettings.from_json(settings_json))
    st.session_state["toast"] = "設定を読み込みました（「設定と予測」タブ）"


def _ufc_set_idx(vals: list[float]) -> None:
    for k, v in zip(("idx1", "idx2", "idx3"), vals):
        st.session_state[f"ufc_{k}"] = float(v)


def _ufc_set_end(v: float) -> None:
    st.session_state["ufc_end_dbv"] = float(v)


def _num(col, key: str) -> None:
    label, lo, hi, step, f = UFC_FIELDS[key]
    col.number_input(label, lo, hi, step=step, format=f, key=f"ufc_{key}")


def ufc_settings_form(pid: str) -> ufc.UFCSettings:
    """装置の『BV-UFC』設定画面と同じ並びの入力欄。"""
    if st.session_state.get("ufc_pid") != pid:  # 患者を切り替えたら、その患者の保存済み設定(無ければCSVの値)を読む
        _ufc_load_saved(pid)
        st.session_state["ufc_pid"] = pid
    elif any(f"ufc_{k}" not in st.session_state for k in [*UFC_FIELDS, "placement"]):
        # 他の画面へ移ると入力欄の値は消えるため、控えておいた編集中の設定を戻す
        _ufc_put(ufc.UFCSettings.from_json(st.session_state["ufc_shadow"]) if "ufc_shadow" in st.session_state
                 else _ufc_from_csv(pid))
    with st.container(border=True):
        c = st.columns([1, 1, 1.2, 3.2])
        _num(c[0], "total_min")
        _num(c[1], "target_uf")
        s0 = _ufc_get()
        c[2].metric("平均除水速度", f"{s0.mean_rate:.2f} L/h", help="目標除水量 ÷ 治療時間。開始・定倍除水の倍率はこの値に掛けます。")
        with c[3]:
            st.radio("目標点（指数）の時間配置", list(ufc.PLACEMENTS), format_func=ufc.PLACEMENTS.get,
                     key="ufc_placement", horizontal=True,
                     help="取り込んだBV-UFC症例では、治療時間を4等分した区間（CSVのステップ3〜6）の終わりに、"
                          "終了時ΔBVに対して指数1・2・3・100%のΔBVへ達するよう制御されていました。"
                          "装置画面のグラフどおりにUFC期間を4等分する配置も選べます。")
        p = st.columns(4)
        p[0].markdown(":blue-background[**開始**]")
        p[1].markdown(":violet-background[**定倍除水**]")
        p[2].markdown(f":green-background[**UFC期間**]　{s0.ufc_min:.0f} 分")
        p[3].markdown(":orange-background[**終了前**]")
        p = st.columns(8)
        _num(p[0], "start_min")
        _num(p[1], "start_mult")
        _num(p[2], "const_min")
        _num(p[3], "const_mult")
        _num(p[4], "ufr_max")
        _num(p[5], "ufr_min")
        _num(p[6], "end_min")
        p[7].caption(f"開始 {s0.mean_rate * s0.start_mult:.2f} L/h  \n定倍 {s0.mean_rate * s0.const_mult:.2f} L/h")
        q = st.columns(5)
        for col, key in zip(q, ["start_dbv", "idx1", "idx2", "idx3", "end_dbv"]):
            _num(col, key)
    s = _ufc_get()
    st.session_state["ufc_shadow"] = s.to_json()
    if s.ufr_min > s.ufr_max:
        st.warning("UFC期間の除水速度の下限が上限より大きくなっています。")
    if not (s.idx1 <= s.idx2 <= s.idx3 <= 100):
        st.warning("指数は 指数1 ≤ 指数2 ≤ 指数3 ≤ 100 にしてください（終了時ΔBVに対する%）。")
    if s.ufc_min <= 0:
        st.warning("開始＋定倍除水＋終了前 の時間が治療時間以上のため、UFC期間がありません。")
    return s


def ufc_figure(s: ufc.UFCSettings, sim: pd.DataFrame, const: pd.DataFrame, bd: pd.DataFrame,
               risk_dbv: float, past_hypo: pd.Series) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.22, 0.28])
    t = sim["min"]
    # 区間の色(装置画面の上部バーと同じ配色)
    edges = [(0, s.start_min, "開始"), (s.start_min, s.start_min + s.const_min, "定倍除水"),
             (s.start_min + s.const_min, s.total_min - s.end_min, "UFC期間"), (s.total_min - s.end_min, s.total_min, "終了前")]
    for a, b, name in edges:
        if b > a:
            fig.add_vrect(x0=a, x1=b, fillcolor=PHASE_COLORS[name], line_width=0, layer="below", row="all", col=1)
            fig.add_annotation(x=(a + b) / 2, y=1.0, xref="x", yref="y domain", yanchor="bottom", text=name,
                               showarrow=False, font=dict(size=11, color="#777"))
    if len(bd):
        fig.add_trace(go.Scatter(x=bd["min"], y=bd["dbv_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=bd["min"], y=bd["dbv_lo"], fill="tonexty", fillcolor="rgba(31,95,191,.13)",
                                 line=dict(width=0), name="日による幅(10〜90%)", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=const["min"], y=const["dbv"], name="参考: 一定速度で除水した場合",
                             line=dict(color="#999", width=1.6, dash="dot"), hovertemplate="一定速度 %{y:.1f}%"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=sim["target"], name="目標ΔBV", line=dict(color="#2a9d8f", width=2, dash="dash"),
                             hovertemplate="目標 %{y:.1f}%"), 1, 1)
    tp, vp = ufc.target_points(s)
    fig.add_trace(go.Scatter(x=tp, y=vp, mode="markers+text", showlegend=False,
                             marker=dict(color="#2a9d8f", size=9), text=["開始", "指数1", "指数2", "指数3", "終了"],
                             textposition="bottom center", textfont=dict(size=10, color="#2a9d8f"),
                             hovertemplate="%{text}: %{y:.1f}%"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=sim["dbv"], name="予測ΔBV（BV-UFC）", line=dict(color="#1f5fbf", width=2.6),
                             hovertemplate="予測 %{y:.1f}%"), 1, 1)
    if not np.isnan(risk_dbv):
        fig.add_hline(y=risk_dbv, line=dict(color="rgba(214,39,40,.6)", dash="dot", width=1.5), row=1, col=1,
                      annotation_text=f"血圧低下リスクの目安 {risk_dbv:.1f}%", annotation_font_size=10,
                      annotation_position="bottom left")
    if len(past_hypo):
        fig.add_trace(go.Scatter(x=past_hypo.index, y=past_hypo.values, mode="markers", name="過去の血圧低下時のΔBV",
                                 marker=dict(symbol="diamond", color="#d62728", size=8, opacity=.55,
                                             line=dict(color="white", width=1)),
                                 hovertemplate="%{x:.0f}分 ΔBV %{y:.1f}%"), 1, 1)
    # 除水速度とPRR
    fig.add_trace(go.Scatter(x=t, y=sim["ufr"], name="予測除水速度", line=dict(color="#e08a00", width=2, shape="hv"),
                             hovertemplate="%{y:.2f} L/h"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=sim["prr"], name="推定PRR", line=dict(color="#6a4c93", width=1.6, dash="dash"),
                             hovertemplate="PRR %{y:.2f} L/h"), 2, 1)
    fig.add_hline(y=s.ufr_max, line=dict(color="rgba(224,138,0,.45)", dash="dot"), row=2, col=1)
    # 除水量
    if len(bd):
        fig.add_trace(go.Scatter(x=bd["min"], y=bd["uf_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 3, 1)
        fig.add_trace(go.Scatter(x=bd["min"], y=bd["uf_lo"], fill="tonexty", fillcolor="rgba(108,117,125,.15)",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"), 3, 1)
    fig.add_trace(go.Scatter(x=const["min"], y=const["uf_vol"], showlegend=False,
                             line=dict(color="#999", width=1.6, dash="dot"), hovertemplate="一定速度 %{y:.2f} L"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=sim["uf_vol"], name="予測除水量", line=dict(color="#495057", width=2.4),
                             hovertemplate="%{y:.2f} L"), 3, 1)
    fig.add_hline(y=s.target_uf, line=dict(color="rgba(73,80,87,.5)", dash="dash"), row=3, col=1,
                  annotation_text=f"目標 {s.target_uf:.2f} L", annotation_font_size=10, annotation_position="top left")
    fig.update_yaxes(title_text="ΔBV (%)", row=1, col=1)
    fig.update_yaxes(title_text="L/h", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="除水量 (L)", row=3, col=1, rangemode="tozero")
    fig.update_xaxes(title_text="治療経過時間 (分)", row=3, col=1, dtick=30)
    fig.update_layout(height=760, template="plotly_white", hovermode="x unified",
                      margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _ufc_notes(s: ufc.UFCSettings, sim: pd.DataFrame, sm: dict, model, rk, risk_p: int, risk_dbv: float) -> None:
    """予測から読み取れる注意点を表示する。"""
    if sm["uf_total"] < s.target_uf - 0.2:
        need = ufc.solve_end_dbv(s, model)
        msg = f"目標除水量に **{s.target_uf - sm['uf_total']:.2f} L 届かない**見込みです。"
        if not np.isnan(need):
            nr = float(ufc.risk_at(rk, need)) if rk else np.nan
            msg += f" 目標に届く終了時ΔBVの目安は **{need:.1f}%**"
            msg += (f"（その時の血圧低下リスク {nr * 100:.0f}%）。" if not np.isnan(nr) else "。")
        else:
            msg += (" 終了時ΔBVを −30% まで深くしても届かない見込みです"
                    "（UFC期間の上限・定倍除水・治療時間・目標除水量の見直しが必要です）。")
        st.warning(msg)
    elif sm["uf_total"] > s.target_uf + 0.3:
        st.info(f"目標除水量を **{sm['uf_total'] - s.target_uf:.2f} L 上回る**見込みです。終了時ΔBVを浅くできる可能性があります。")
    if sm["min_dbv"] < s.end_dbv - 2 and sim.loc[sim["dbv"].idxmin(), "phase"] in ("開始", "定倍除水"):
        st.warning(f"定倍除水の間にΔBVが **{sm['min_dbv']:.1f}%** まで下がり、目標カーブより先に下がりすぎる見込みです"
                   "（その後UFCが除水を絞ります）。定倍除水の倍率または時間を下げることを検討してください。")
    if sm["at_min_pct"] > 30:
        st.info("UFC中、除水速度が下限に張り付く時間が長い見込みです（ΔBVが目標より下がっている）。")
    if sm["at_max_pct"] > 30:
        st.info("UFC中、除水速度が上限に張り付く時間が長い見込みです（PRRが多く、ΔBVが目標まで下がりにくい）。"
                "この間は目標カーブより上を推移します。")
    if not np.isnan(risk_dbv) and sm["min_dbv"] < risk_dbv:
        st.warning(f"予測ΔBVが、血圧低下リスク {risk_p}% の目安（{risk_dbv:.1f}%）を下回る時間帯があります。")


def ufc_index_section(s: ufc.UFCSettings, sim: pd.DataFrame, model, rk, pid: str) -> None:
    """指数ごとの目安: 各時点の目標・予測ΔBVとリスク、過去に血圧低下なく終えた症例の経過の形。"""
    st.markdown("**指数ごとの目安**")
    shapes = load_shapes(REV)
    summ = load_summary(REV, CRIT, int(WINDOW))
    own = summ[summ.patient_id == pid]
    guide = ufc.index_guide(shapes, own.session_id[~own.hypo_final.astype(bool)], own.session_id[own.hypo_final.astype(bool)])
    src = "この患者"
    if guide is None:  # 低下なく終えた症例が少ない患者は全患者で
        guide = ufc.index_guide(shapes, summ.session_id[~summ.hypo_final.astype(bool)],
                                summ.session_id[summ.hypo_final.astype(bool)])
        src = "全患者"
    tp, vp = ufc.target_points(s)
    names = ["開始", "指数1", "指数2", "指数3", "終了"]
    setv = [0.0, s.idx1, s.idx2, s.idx3, 100.0]
    rows = []
    for i, (name, t_, v_) in enumerate(zip(names, tp, vp)):
        pred = float(np.interp(t_, sim["min"], sim["dbv"]))
        key = f"idx{i}" if 1 <= i <= 3 else None
        rows.append({"時点": name, "時刻(分)": round(t_), "設定(%)": setv[i], "目標ΔBV(%)": v_, "予測ΔBV(%)": pred,
                     "血圧低下リスク(%)": float(ufc.risk_at(rk, pred)) * 100 if rk else np.nan,
                     "低下なく終えた症例の実績(%)": guide["ok"][key] if guide and key else (100.0 if i == 4 and guide else np.nan),
                     "低下があった症例の実績(%)": guide["bad"].get(key, np.nan) if guide and key else np.nan})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.style.format({c: "{:.1f}" for c in tbl.columns if c not in ("時点", "時刻(分)", "設定(%)")}
                                  | {"設定(%)": "{:.0f}"}, na_rep="–"), hide_index=True, width="stretch")
    st.caption("設定・実績の % は、終了時ΔBVに対する割合（指数）。血圧低下リスク = その時点の予測ΔBVでの、血圧測定1回あたりの低下の確率。"
               "実績 = 過去の症例で、治療を4等分した時点のΔBVが終了時ΔBVの何%だったか（中央値。BV-UFCを使っていない症例も含む）。")
    if not guide:
        st.info("血圧低下なく終えた症例が少ないため、指数の目安は出せません。")
        return
    rec = guide["rec"]
    x = ufc.UFCSettings.from_dict({**asdict(s), "idx1": rec[0], "idx2": rec[1], "idx3": rec[2]})
    xs = ufc.sim_summary(x, ufc.simulate(x, model))
    cur = ufc.sim_summary(s, sim)
    c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
    c1.markdown(f"目安: **指数1 {rec[0]:.0f}／指数2 {rec[1]:.0f}／指数3 {rec[2]:.0f}**　"
                f"（{src}で血圧低下なく終えた {guide['n_ok']} 症例の経過の形・中央値。その症例の終了時ΔBVの中央値は {guide['end_ok']:.1f}%）  \n"
                f"この指数にすると: 予測除水量 {xs['uf_total']:.2f} L（今の設定 {cur['uf_total']:.2f} L）")
    if [s.idx1, s.idx2, s.idx3] != rec:
        c2.button("この指数を設定する", on_click=_ufc_set_idx, args=(rec,), use_container_width=True)
    if s.placement == "ufc":
        st.caption("※ 実績の形は治療全体を4等分した時点で求めています。目標点の配置が「UFC期間を4等分」のときは時点がずれるため参考程度にしてください。")
    with st.expander("この目安の根拠と限界"):
        st.markdown(
            "- 取り込み済みの全症例で調べると、**血圧低下はその時点のΔBVの深さと関係**しており（深いほど起きやすい）、"
            "**ΔBVの下がる速さとは関係が見られません**でした。\n"
            "- 経過の形（前半に下げるか、後半に下げるか）と、その後の血圧低下との関係ははっきりしませんでした"
            "（前半に下げた症例で少ない傾向はあるものの、除水量・終了時ΔBVをそろえると明確ではない）。\n"
            "- そのため、統計的に最適な指数を求めるのではなく、**その患者で実際に血圧低下なく終えられた経過の形**を目安としています。\n"
            "- 血圧低下があった症例は、低下後の除水停止などで経過の形が変わっているため、比較は参考程度にしてください。")


def ufc_tab_sim(pid: str) -> None:
    s = ufc_settings_form(pid)
    b = st.columns([1.2, 1.4, 1, 2.4])
    b[0].button("保存済みの設定を読む", on_click=_ufc_load_saved, args=(pid,), use_container_width=True,
                help="この患者で最後に保存した設定。無ければ最新症例のCSVの設定値。")
    b[1].button("最新症例のCSVの設定値を読む", on_click=lambda: _ufc_put(_ufc_from_csv(pid)), use_container_width=True,
                help="CSVには治療時間・目標除水量・開始/定倍除水/終了前の値が記録されています。"
                     "指数・終了時ΔBV・上限/下限は記録されないため既定値になります。")
    b[2].button("既定値", on_click=lambda: _ufc_put(ufc.UFCSettings()), use_container_width=True)
    with b[3].popover("💾 この設定を保存", use_container_width=True):
        memo = st.text_input("メモ（任意）", key="ufc_memo", placeholder="例: 終了時ΔBVを−10%に変更")
        if st.button("保存", type="primary", key="ufc_save"):
            db.save_ufc_setting(pid, s.to_json(), memo)
            st.cache_data.clear()
            st.session_state["toast"] = "BV-UFCの設定を保存しました"
            st.rerun()

    if s.ufc_min <= 0 or s.total_min <= 0:
        return
    model = load_patient_model(REV, pid)
    if model is None:
        st.info("予測に必要なデータがまだありません。CSVを取り込んでください。")
        return
    sim = ufc.simulate(s, model)
    const = ufc.simulate(s, model, "const")
    bd = ufc.band(s, load_day_models(REV, pid))
    sm = ufc.sim_summary(s, sim)
    rk = load_risk(REV, CRIT, int(WINDOW), pid)

    st.subheader("予測")
    r1, r2 = st.columns([3, 1.3], vertical_alignment="bottom")
    basis = (f"この患者の症例が少ないため、全患者 {model.n_sessions} 症例から推定" if model.source == "全患者のデータ"
             else f"この患者の過去 {model.n_sessions} 症例から推定（{model.source}）")
    r1.caption(f"{basis}：循環血液量 **{model.bv:.1f} L**、"
               f"PRR **{float(model.prr(0, 0.5)):.2f} L/h**（開始直後）→ **{float(model.prr(-10, 3)):.2f} L/h**（ΔBV −10%・3時間）")
    risk_p = int(r2.number_input("血圧低下リスクの目安（測定1回あたり %）", 5, 60, 20, 5, key="ufc_risk_p",
                                 help="血圧測定1回あたりで血圧低下と判定される確率がこの値になるΔBVを、グラフの赤い点線で示します。"))
    risk_dbv = ufc.dbv_for_risk(rk, risk_p / 100)
    base_risk = float(ufc.risk_at(rk, 0.0)) if rk else np.nan
    if not np.isnan(risk_dbv) and risk_dbv > -0.5:  # ΔBVが下がる前からリスクが目安以上 → 目安線は意味を持たない
        risk_dbv = np.nan
        r1.caption(f"この患者はΔBVが0%でも血圧低下リスクが約 {base_risk * 100:.0f}% あり（目安 {risk_p}% 以上）、"
                   "ΔBVだけでは区切れないため、目安線は表示しません。")
    end_risk = float(ufc.risk_at(rk, sm["end_dbv"])) if rk else np.nan
    bd_end = bd.iloc[-1] if len(bd) else None

    k = st.columns(5)
    k[0].metric("予測除水量", f"{sm['uf_total']:.2f} L", f"{sm['uf_total'] - s.target_uf:+.2f} L（目標比）",
                delta_color="normal" if sm["uf_total"] >= s.target_uf - 0.1 else "inverse",
                help="日による幅（10〜90%）: " + (f"{bd_end.uf_lo:.2f}〜{bd_end.uf_hi:.2f} L" if bd_end is not None else "–"))
    k[1].metric("終了時ΔBV", f"{sm['end_dbv']:.1f} %", f"最低 {sm['min_dbv']:.1f} %", delta_color="off")
    k[2].metric("血圧低下リスク（終了時）", f"{end_risk * 100:.0f} %" if not np.isnan(end_risk) else "–",
                f"推定: {rk['source']}" if rk else "推定できません", delta_color="off",
                help="その時点のΔBVでの、血圧測定1回あたりの低下の確率（過去の測定点からのロジスティック回帰）。")
    k[3].metric("除水速度が上限の時間", f"{sm['at_max_pct']:.0f} %", "UFC期間・終了前のうち", delta_color="off")
    k[4].metric("除水速度が下限の時間", f"{sm['at_min_pct']:.0f} %", "UFC期間・終了前のうち", delta_color="off")
    _ufc_notes(s, sim, sm, model, rk, risk_p, risk_dbv)

    summ = load_patient_summary(REV, pid, CRIT, int(WINDOW))
    has = len(summ) and {"first_hypo_time", "dbv_at_first_hypo"} <= set(summ.columns)  # 低下が一度も無い患者は列が無い
    ph = summ.dropna(subset=["first_hypo_time", "dbv_at_first_hypo"]) if has else summ.iloc[:0]
    past = pd.Series(ph["dbv_at_first_hypo"].to_numpy(), index=ph["first_hypo_time"].to_numpy()) if len(ph) else pd.Series(dtype=float)
    st.plotly_chart(ufc_figure(s, sim, const, bd, risk_dbv, past), use_container_width=True, config=PLOT_CONFIG)
    st.caption("青 = BV-UFCでの予測ΔBV（帯は、この患者の過去の症例ごとのrefillingの違いを当てはめた幅）／緑の破線 = 目標ΔBV／"
               "灰の点線 = 目標除水量を一定速度で除水した場合／◆ = 過去の症例で初めて血圧低下と判定された時点のΔBV。")
    with st.container(border=True):
        ufc_index_section(s, sim, model, rk, pid)

    with st.expander("終了時ΔBVを変えた場合の比較"):
        rows = []
        for e in np.arange(-4.0, -22.1, -2.0):
            x = ufc.UFCSettings.from_dict({**asdict(s), "end_dbv": float(e)})
            r_ = ufc.sim_summary(x, ufc.simulate(x, model))
            rows.append({"終了時ΔBV(%)": e, "予測除水量(L)": r_["uf_total"], "目標比(L)": r_["uf_total"] - s.target_uf,
                         "血圧低下リスク(終了時,%)": float(ufc.risk_at(rk, r_["end_dbv"])) * 100 if rk else np.nan,
                         "下限の時間(%)": r_["at_min_pct"], "上限の時間(%)": r_["at_max_pct"]})
        cmp_ = pd.DataFrame(rows)
        st.dataframe(cmp_.style.format({c: "{:.1f}" for c in cmp_.columns} | {"予測除水量(L)": "{:.2f}", "目標比(L)": "{:+.2f}"},
                                       na_rep="–"), hide_index=True, width="stretch")
        st.caption("他の設定は今の値のまま、終了時ΔBVだけを変えて予測したものです。")
        need = ufc.solve_end_dbv(s, model) if sm["uf_total"] < s.target_uf - 0.05 else np.nan
        if not np.isnan(need) and abs(need - s.end_dbv) >= 0.5:
            st.button(f"目標除水量に届く終了時ΔBV（{need:.1f}%）を設定する", on_click=_ufc_set_end, args=(need,))

    with st.expander("予測のしくみと精度"):
        st.markdown(
            "- **ΔBVの変化 = −100 ÷ 循環血液量 ×（除水速度 − PRR）** で、治療を1分ずつ計算しています。\n"
            "- **循環血液量**は、装置が記録したPRR・除水速度・ΔBVの傾きから逆算した値（装置がPRRの計算に使っている値）です。\n"
            "- **PRR**は、過去の症例の除水量の収支（実際の除水量 − 循環血液量×ΔBVの低下）から水準を、"
            "装置が記録したPRRからΔBV・経過時間との関係を推定しています。\n"
            "- **UFC期間・終了前**は、2分ごとにΔBVの変化率を見て目標カーブに沿うよう除水速度を更新し、上限・下限で制限しています"
            "（装置の制御そのものではなく、それをまねた計算です）。\n"
            "- 症例が少ない患者は、全患者のデータで補っています。")
        bt = load_backtest(REV, pid)
        if bt.empty:
            st.info("この患者の検証に使える症例がまだありません。")
        else:
            k = st.columns(3)
            k[0].metric("検証した症例", f"{len(bt)} 件", "それより前の症例だけで予測", delta_color="off")
            k[1].metric("除水量の誤差（中央値）", f"±{bt.err.abs().median():.2f} L",
                        f"全患者モデルでは ±{bt.err_pooled.abs().median():.2f} L", delta_color="off")
            k[2].metric("平均のずれ", f"{bt.err.mean():+.2f} L", help="プラス = 多めに予測")
            figv = go.Figure()
            figv.add_trace(go.Scatter(x=bt.uf_actual, y=bt.uf_pred, mode="markers", name="この患者のモデル",
                                      marker=dict(color="#1f5fbf", size=9), text=bt.end_dbv.round(1),
                                      hovertemplate="実際 %{x:.2f} L / 予測 %{y:.2f} L（終了時ΔBV %{text}%）"))
            mx = float(max(bt.uf_actual.max(), bt.uf_pred.max())) + 0.3
            figv.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#aaa", dash="dot"),
                                      showlegend=False, hoverinfo="skip"))
            figv.update_layout(height=340, template="plotly_white", xaxis_title="実際の除水量 (L)",
                               yaxis_title="予測した除水量 (L)", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(figv, use_container_width=True, config=PLOT_CONFIG)
            st.caption("直近の症例を、それより前の症例だけで作ったモデルで予測した検証です（実際のΔBVの経過を与えて除水量を予測）。"
                       "点が点線に近いほど正確です。")
    st.caption("⚠ 観察データからの統計的な概算です。体重・血圧薬・体液量の変化は考慮していません。設定の最終判断は臨床で行ってください。")


def ufc_tab_results(pid: str) -> None:
    used = load_ufc_used(REV)
    scope = st.radio("対象", ["この患者", "全患者"], horizontal=True, key="ufc_scope")
    summ = load_summary(REV, CRIT, int(WINDOW)).copy()
    summ["ufc"] = summ.session_id.map(used).fillna(False).astype(bool)
    summ["UFC"] = np.where(summ.ufc, "BV-UFC使用", "不使用")
    if scope == "この患者":
        summ = summ[summ.patient_id == pid]
    if summ.empty:
        st.info("症例がありません。")
        return
    if not summ.ufc.any():
        st.info("BV-UFCを使用した症例がありません（CSVの BV-UFC step no が0のみ）。")
        return

    st.markdown("**BV-UFC使用 / 不使用の比較**")
    rows = []
    for g in ["BV-UFC使用", "不使用"]:
        d = summ[summ.UFC == g]
        p, lo, hi = wilson(int(d.hypo_final.sum()), len(d))
        rows.append({"群": g, "症例数": len(d), "血圧低下あり": int(d.hypo_final.sum()), "低下率(%)": p,
                     "95%区間": f"{lo:.0f}–{hi:.0f}" if len(d) else "–",
                     "除水量(L)": d.uf_total.mean(), "終了時ΔBV(%)": d.end_dbv.mean(), "最小ΔBV(%)": d.min_dbv.mean(),
                     "除水停止(回)": d.n_uf_stop.mean()})
    cmp_ = pd.DataFrame(rows)
    num = [c for c in cmp_.columns if c not in ("群", "95%区間", "症例数", "血圧低下あり")]
    st.dataframe(cmp_.style.format({c: "{:.1f}" for c in num} | {"除水量(L)": "{:.2f}"}, na_rep="–"),
                 hide_index=True, width="stretch")
    if scope == "全患者":
        mh = sm_.mantel_haenszel(summ, group_col="UFC", a="不使用", b="BV-UFC使用")
        if mh:
            st.caption(f"患者で層別したオッズ比（BV-UFC使用 vs 不使用）: **{mh['or']:.2f}**"
                       f"（95%区間 {mh['lo']:.2f}–{mh['hi']:.2f}、p = {mh['p']:.3g}、{mh['n_strata']} 名）。"
                       "1より小さければBV-UFC使用で血圧低下が少ない。")
        pp = sm_.paired_by_patient(summ, group_col="UFC", a="不使用", b="BV-UFC使用", min_each=3)
        if not pp.empty:
            show = pp.copy()
            show["patient_id"] = show["patient_id"].map(lambda x: pid_label(x, patients))
            st.dataframe(show.rename(columns={"patient_id": "患者"}).round(1), hide_index=True, width="stretch")
    st.caption("BV-UFCを使う症例は、もともと血圧が下がりやすい患者・日である可能性があり、単純な比較は因果を示しません。")

    ids_u = tuple(int(x) for x in summ[summ.ufc].session_id)
    ids_n = tuple(int(x) for x in summ[~summ.ufc].session_id)
    fig = go.Figure()
    for ids, name, color, fill in ((ids_u, "BV-UFC使用", "#2a9d8f", "rgba(42,157,143,.14)"),
                                   (ids_n, "不使用", "#888888", "rgba(136,136,136,.12)")):
        if not ids:
            continue
        cs, n = curve_stats(REV, ids, "dbv_avg5", False)
        cs = cs[cs.n >= max(3, n * 0.3)]
        fig.add_trace(go.Scatter(x=list(cs.x) + list(cs.x[::-1]), y=list(cs.q75) + list(cs.q25[::-1]), fill="toself",
                                 fillcolor=fill, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cs.x, y=cs["mean"], name=f"{name}（{n}症例）", line=dict(color=color, width=2.6),
                                 hovertemplate="%{y:.1f}%"))
    fig.update_layout(height=360, template="plotly_white", xaxis_title="治療経過時間 (分)", yaxis_title="ΔBV 5分平均 (%)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    st.caption("線 = 平均、帯 = 四分位範囲。")

    st.markdown("**ステップごとの集計（BV-UFC使用症例）**")
    stt = load_step_table(REV, CRIT, ids_u)
    if not stt.empty:
        num = [c for c in stt.columns if c not in ("ステップ", "症例数", "血圧測定数", "低下数")]
        st.dataframe(stt.style.format({c: "{:.1f}" for c in num} | {"平均除水速度(L/h)": "{:.2f}", "平均PRR(L/h)": "{:.2f}"},
                                      na_rep="–"), hide_index=True, width="stretch")
        st.caption("ステップはCSVの「BV-UFC step no」。区間1〜4は治療時間をほぼ4等分した区間で、各区間の終わりに指数1・2・3・"
                   "終了時ΔBVへ達するよう制御されていました（取り込み実績からの推定）。「停止/その他」はUFC使用症例の中で "
                   "step no が0の時間帯です。低下の頻度 = その区間の血圧測定のうち低下と判定された割合。")

    ia = load_idx_ach(REV, ids_u)
    if len(ia) >= 3:
        st.markdown("**指数の実績（区間の終わりのΔBVが、終了時ΔBVの何%だったか）**")
        figi = go.Figure()
        for k_ in ("idx1", "idx2", "idx3"):
            figi.add_trace(go.Box(y=ia[k_].clip(-20, 200), name=k_.replace("idx", "指数"), boxpoints="outliers",
                                  marker_color="#2a9d8f"))
        for ref in (50, 80, 95):
            figi.add_hline(y=ref, line=dict(color="rgba(0,0,0,.25)", dash="dot"))
        figi.update_layout(height=320, template="plotly_white", yaxis_title="終了時ΔBVに対する%", showlegend=False,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figi, use_container_width=True, config=PLOT_CONFIG)
        st.caption(f"{len(ia)} 症例。点線 = 装置画面の例の設定値（50・80・95）。症例ごとの実際の設定値はCSVに記録されないため、"
                   "目安として比べてください。中央値: " + "／".join(f"指数{i} {ia[f'idx{i}'].median():.0f}%" for i in (1, 2, 3)))


def ufc_tab_saved(pid: str) -> None:
    saved = load_ufc_saved(REV, pid)
    if saved.empty:
        st.info("この患者の保存済み設定はありません。「設定と予測」で「この設定を保存」を押すと、ここに記録されます。")
        return
    rows = []
    for r in saved.itertuples():
        s = ufc.UFCSettings.from_json(r.settings_json)
        rows.append({"id": r.setting_id, "保存日時": r.saved_at, "メモ": r.memo or "", "治療時間(分)": s.total_min,
                     "目標除水量(L)": s.target_uf, "開始": f"{s.start_min:.0f}分 ×{s.start_mult:.2f}",
                     "定倍除水": f"{s.const_min:.0f}分 ×{s.const_mult:.2f}",
                     "UFC上限/下限": f"{s.ufr_max:.2f} / {s.ufr_min:.2f}", "終了前": f"{s.end_min:.0f}分",
                     "指数": f"{s.idx1:.0f} / {s.idx2:.0f} / {s.idx3:.0f}",
                     "ΔBV 開始→終了": f"{s.start_dbv:.1f} → {s.end_dbv:.1f}"})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.drop(columns="id"), hide_index=True, width="stretch")
    lab = tbl.set_index("id")
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    pick = c1.selectbox("設定", tbl.id, format_func=lambda i: f"{lab.loc[i, '保存日時']}　{lab.loc[i, 'メモ']}")
    c2.button("この設定を読み込む", use_container_width=True, on_click=_ufc_load_one,
              args=(saved.set_index("setting_id").loc[pick, "settings_json"],))
    if c3.button("削除", use_container_width=True):
        db.delete_ufc_setting(int(pick))
        st.cache_data.clear()
        st.rerun()


def _on_pid_u() -> None:
    pid = st.session_state["w_pid_u"]
    st.session_state["sel_pid"], st.session_state["sel_sid"] = pid, _newest_sid(pid)


def page_ufc():
    st.header("⑥ BV-UFC")
    if sessions.empty:
        empty_state()
        return
    st.caption("BV-UFC（ΔBVを見て除水速度を自動で調節する機能）の設定を試し、その患者の過去データから ΔBV・除水量・血圧低下リスクを"
               "予測します。設定は装置には送られません。決めた値は装置で入力してください。")
    cur_pid, _, _ = current_selection()
    st.session_state["w_pid_u"] = cur_pid
    pid = st.selectbox("患者", ACTIVE_PIDS, key="w_pid_u", on_change=_on_pid_u, format_func=lambda x: pid_label(x, patients))
    t1, t2, t3 = st.tabs(["🎛 設定と予測", "📊 実績（BV-UFC使用症例）", "🗂 保存した設定"])
    with t1:
        ufc_tab_sim(pid)
    with t2:
        ufc_tab_results(pid)
    with t3:
        ufc_tab_saved(pid)


# ----------------------------------------------------------------- ⑦ manage

def similar_names(pts: pd.DataFrame, thr: float = 0.7) -> list[tuple[str, str]]:
    """表記違いの可能性がある、似た氏名の患者の組。"""
    import difflib
    from bv_parser import normalize_name
    d = pts[pts.n_sessions > 0].dropna(subset=["name"])
    items = [(r.patient_id, normalize_name(r.name)) for r in d.itertuples() if normalize_name(r.name)]
    return [(a, b) for i, (a, na) in enumerate(items) for b, nb in items[i + 1:]
            if difflib.SequenceMatcher(None, na, nb).ratio() >= thr]


def page_manage():
    st.header("⑦ データ管理")
    if sessions.empty:
        empty_state()
        return
    t1, t2, t3, t4 = st.tabs(["臨床情報の入力", "患者", "エクスポート", "削除"])
    with t1:
        st.caption("血圧低下の有無・症状・対応を入力します。「血圧低下(臨床)」を入力した症例は、集計で自動判定より優先されます。"
                   "除水量はCSVの値です。")
        pf = st.selectbox("患者で絞り込み", ["（全員）", *ACTIVE_PIDS], key="clin_f")
        d = sessions if pf == "（全員）" else sessions[sessions.patient_id == pf]
        d = d.merge(load_uf(REV), on="session_id", how="left")
        ed = d[["session_id", "patient_id", "session_dt", "mode_label", "dur_group", "uf_total",
                "hypo_event", "symptom", "intervention", "memo"]].copy()
        ed["hypo_event"] = ed["hypo_event"].map({0: "なし", 1: "あり"}).fillna("未記入")
        ed.insert(1, "patient", ed["patient_id"].map(lambda x: pid_label(x, patients)))
        out = st.data_editor(
            ed, hide_index=True, width="stretch", key="clin_editor",
            disabled=["session_id", "patient", "patient_id", "session_dt", "mode_label", "dur_group", "uf_total"],
            column_config={
                "session_id": "ID", "patient": "患者", "patient_id": "患者ID",
                "session_dt": st.column_config.DatetimeColumn("施行日時", format="YYYY-MM-DD HH:mm"),
                "mode_label": "治療モード", "dur_group": "時間群",
                "uf_total": st.column_config.NumberColumn("除水量(L)", format="%.2f"),
                "hypo_event": st.column_config.SelectboxColumn("血圧低下(臨床)", options=HYPO_OPTS, required=True),
                "symptom": "症状", "intervention": "対応（補液・除水停止など）", "memo": "メモ",
            })
        if st.button("臨床情報を保存", type="primary"):
            o = out.copy()
            o["hypo_event"] = o["hypo_event"].map({"なし": 0, "あり": 1})
            db.update_clinical(o[["session_id", *db.CLINICAL_COLS]])
            st.cache_data.clear()
            st.success("保存しました")
            st.rerun()
    with t2:
        cols = ["patient_id", *(["name"] if SHOW_NAMES else []), "alias", "note", "n_sessions", "first_dt", "last_dt",
                "consoles"]
        out = st.data_editor(patients[cols], hide_index=True, width="stretch", key=f"pat_editor_{SHOW_NAMES}",
                             disabled=["patient_id", "n_sessions", "first_dt", "last_dt", "consoles"],
                             column_config={"patient_id": "患者ID", "name": "氏名", "alias": "研究ID/表示名",
                                            "note": "メモ", "n_sessions": "症例数", "first_dt": "初回", "last_dt": "最終",
                                            "consoles": "使用した装置"})
        st.caption("患者IDはCSVの氏名から作っています（CSVの「ID」は装置の番号で、同じ装置を複数の患者が使うため）。")
        if not SHOW_NAMES:
            st.caption("氏名は非表示中です（サイドバーの「氏名を表示」をONにすると表示・編集できます）。")
        if st.button("患者情報を保存"):
            db.update_patients(out[[c for c in ("patient_id", "name", "alias", "note") if c in out.columns]])
            st.cache_data.clear()
            st.success("保存しました")
            st.rerun()
        st.caption(f"元CSVの保存先: `{db.PATIENT_DIR}\\<患者ID>\\`　／　DB: `{db.DB_PATH}`")
        with st.expander("患者の統合（同じ人が別の氏名で登録されている場合）"):
            st.caption("氏名の入力違い（漢字・字体の違いなど）で、同じ人が別の患者として登録された場合に使います。"
                       "左の患者の全症例・保存した設定を右の患者へ移し、左の患者を削除します。元に戻せません。")
            m1, m2 = st.columns(2)
            src = m1.selectbox("統合する患者（消える方）", ACTIVE_PIDS, key="merge_src",
                               format_func=lambda x: f"{pid_label(x, patients)}　{int(patients.set_index('patient_id').n_sessions.get(x, 0))}症例")
            dst = m2.selectbox("統合先の患者（残る方）", [p for p in ACTIVE_PIDS if p != src], key="merge_dst",
                               format_func=lambda x: f"{pid_label(x, patients)}　{int(patients.set_index('patient_id').n_sessions.get(x, 0))}症例")
            sims = similar_names(patients) if SHOW_NAMES else []
            if sims:
                st.caption("似た氏名の組: " + "、".join(f"{pid_label(a, patients)} と {pid_label(b, patients)}" for a, b in sims))
            if dst and st.checkbox("同じ人物であることを確認しました", key="merge_ok") and \
                    st.button("統合する", type="primary", key="merge_go"):
                n = db.merge_patients(src, dst)
                st.cache_data.clear()
                st.session_state["toast"] = f"{n} 症例を統合しました"
                st.rerun()
    with t3:
        if not st.toggle("エクスポートを準備する", False,
                         help="全症例を集計するため、症例数が多いと数秒〜十数秒かかります。"):
            st.caption("ONにすると、サマリー・時系列データ・DBバックアップのダウンロードボタンが表示されます。")
        else:
            summ = load_summary(REV, CRIT, int(WINDOW))
            L = summary_labels(int(WINDOW))
            show = summ.drop(columns=["hypo_auto", "treat_mode"], errors="ignore").rename(
                columns={**L, "dur_group": "時間群", "mode_label": "治療モード"})
            inc_name = st.checkbox("出力に氏名を含める", False,
                                   help="研究・共有用の出力では通常OFF（患者IDのみ）。")
            pat_out = patients if inc_name else patients.drop(columns=["name"])
            if inc_name:
                show.insert(1, "氏名", show["patient_id"].map(patients.set_index("patient_id")["name"]))
            c1, c2, c3 = st.columns(3)
            c1.download_button("症例サマリー (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                               "bv_summary.csv", "text/csv")
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as xw:
                show.to_excel(xw, sheet_name="症例サマリー", index=False)
                sessions.drop(columns=["header_json", "pre_weight", "post_weight", "dry_weight", "target_uf"]).to_excel(xw, sheet_name="症例情報", index=False)
                pat_out.to_excel(xw, sheet_name="患者", index=False)
            c2.download_button("サマリー＋症例情報 (Excel)", bio.getvalue(), "bv_summary.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            ts_all = load_all_ts(REV).merge(sessions[["session_id", "patient_id", "session_dt"]], on="session_id")
            c3.download_button("全時系列データ (CSV)", ts_all.to_csv(index=False).encode("utf-8-sig"),
                               "bv_timeseries.csv", "text/csv")
            if db.DB_PATH.exists():
                st.download_button("DBバックアップ (SQLite)", db.DB_PATH.read_bytes(), "bv_research.db",
                                   "application/octet-stream")
                st.caption("DBバックアップには氏名が含まれます。保管先にご注意ください（クラウド同期フォルダは避ける）。")
    with t4:
        st.warning("削除は元に戻せません。")
        kind = st.radio("対象", ["症例", "患者（全症例）"], horizontal=True)
        del_files = st.checkbox("患者フォルダ内の元CSVも削除する", False)
        if kind == "症例":
            sel = pick_session("del")
            if sel and st.checkbox("この症例を削除することを確認") and st.button("削除", type="primary"):
                db.delete_session(sel[1], del_files)
                st.cache_data.clear()
                st.rerun()
        else:
            pid = st.selectbox("患者", patients.patient_id.tolist(), key="del_pat")
            if st.checkbox(f"{pid} の全症例を削除することを確認") and st.button("削除", type="primary", key="delp"):
                db.delete_patient(pid, del_files)
                st.cache_data.clear()
                st.rerun()


# ----------------------------------------------------------------- ⑧ settings
def _open_folder(path: Path) -> None:
    import os
    try:
        os.startfile(str(path))  # Windows: エクスプローラで開く
    except Exception as e:  # noqa: BLE001
        st.toast(f"フォルダを開けませんでした: {e}", icon="⚠️")


def _dir_size_mb(p: Path) -> tuple[int, float]:
    n = tot = 0
    if p.exists():
        for f in p.rglob("*"):
            if f.is_file():
                n += 1
                tot += f.stat().st_size
    return n, tot / 1e6


def page_settings():
    st.header("⑧ 設定")
    cfg = bset.load()

    st.subheader("データの保存場所")
    n_csv, mb_csv = _dir_size_mb(db.PATIENT_DIR)
    db_mb = db.DB_PATH.stat().st_size / 1e6 if db.DB_PATH.exists() else 0
    items = [("データ保存先（すべてここに保存）", db.BASE_DIR, "フォルダごとコピーすれば、別のPCへ移行できます"),
             ("データベース（SQLite）", db.DB_PATH, f"{db_mb:.0f} MB"),
             ("元CSVの複製（患者別フォルダ）", db.PATIENT_DIR, f"{n_csv} ファイル / {mb_csv:.0f} MB"),
             ("設定ファイル", bset.SETTINGS_PATH, "")]
    for label, path, note in items:
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        c1.markdown(f"**{label}**" + (f"　<span style='opacity:.65'>{note}</span>" if note else ""), unsafe_allow_html=True)
        c1.code(str(path), language=None)
        if c2.button("📂 開く", key=f"open_{label}", use_container_width=True):
            _open_folder(path if path.is_dir() else path.parent)
    if "onedrive" in str(db.BASE_DIR).lower():
        st.warning("データ保存先がOneDrive配下です。患者情報がクラウドに同期されます。環境変数 BV_DATA_DIR で変更できます。")
    else:
        st.success("データ保存先はクラウド同期の対象外です。")
    st.caption("保存先を変更するには、環境変数 `BV_DATA_DIR` に新しいフォルダを指定してアプリを再起動します（既存データは手動でコピー）。")

    st.divider()
    st.subheader("バックアップ")
    dest = st.text_input("バックアップ先フォルダ", cfg["backup_dir"], placeholder=r"E:\BV_backup",
                         help="外付けディスクなど、このPCの外にあるフォルダを指定してください。")
    ok, msg = bset.check_dest(dest)
    if dest.strip() and not ok:
        st.error(msg)
    elif dest.strip():
        low = dest.lower()
        if any(w in low for w in ("onedrive", "dropbox", "google drive", "googledrive", "icloud")):
            st.warning("クラウド同期フォルダのようです。氏名を含むため、クラウドに同期されないフォルダを推奨します。")
        elif dest.strip()[:2].upper() == "C:":
            st.warning("PC本体（Cドライブ）内のため、PCの故障時には失われます。外付けディスクなどを推奨します。")
        else:
            st.success("使用できるフォルダです。")
    c1, c2, c3 = st.columns(3)
    auto = c1.toggle("画面を開いた日に1回、自動でバックアップ", cfg["auto_backup"])
    keep = c2.number_input("DBバックアップを残す世代数", 1, 365, int(cfg["keep"]))
    inc = c3.toggle("元CSVの複製も含める", cfg["include_csv"], help="未コピーのファイルだけを追加でコピーします。")
    b1, b2 = st.columns(2)
    if b1.button("設定を保存", type="primary", use_container_width=True):
        bset.save({"backup_dir": dest.strip().strip('"'), "auto_backup": auto, "keep": int(keep), "include_csv": inc})
        st.session_state["toast"] = "設定を保存しました"
        st.rerun()
    if b2.button("今すぐバックアップ", use_container_width=True, disabled=not ok):
        with st.spinner("バックアップ中…"):
            try:
                r = bset.backup_now(dest, int(keep), inc)
                st.success(f"作成しました: {r['file'].name}（{r['size_mb']:.0f} MB）／ 元CSVを {r['csv_copied']} 件追加コピー"
                           + (f" ／ 古い世代 {r['removed']} 件を削除" if r["removed"] else ""))
            except Exception as e:  # noqa: BLE001
                st.error(f"失敗しました: {e}")
    last = bset.last_backup_time(dest) if ok else None
    if last:
        days = (pd.Timestamp.now() - pd.Timestamp(last)).days
        st.caption(f"最新のバックアップ: {last:%Y-%m-%d %H:%M}（{days}日前）")
        files = bset.list_backups(dest)[::-1][:10]
        st.dataframe(pd.DataFrame({"ファイル": [f.name for f in files],
                                   "サイズ(MB)": [round(f.stat().st_size / 1e6, 1) for f in files],
                                   "作成日時": [pd.Timestamp(f.stat().st_mtime, unit="s").tz_localize("UTC")
                                            .tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M") for f in files]}),
                     hide_index=True, width="stretch")
    elif ok:
        st.caption("まだバックアップがありません。「今すぐバックアップ」を押してください。")
    with st.expander("バックアップから復元する方法"):
        st.markdown(
            "1. アプリを終了します（`run.bat` の黒い画面を閉じる）。\n"
            f"2. 現在のDB（`{db.DB_PATH}`）を念のため別名でコピーしておきます。\n"
            f"3. バックアップ先の `bv_research_日時.db` を `{db.BASE_DIR}` にコピーし、名前を `bv_research.db` に変更します。\n"
            "4. `run.bat` でアプリを起動します。元CSVの複製は、バックアップ先の `patients` フォルダを同じ場所へコピーしてください。")

dict(zip(PAGES, [page_import, page_session, page_patient, page_compare, page_hypo, page_ufc, page_manage,
                 page_settings]))[PAGE]()

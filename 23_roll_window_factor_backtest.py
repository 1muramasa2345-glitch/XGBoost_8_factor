"""
滚动窗口因子回测：44 个因子
窗口：半年（6M）和一年（12M），2026 年单独处理
指标：窗口内 RankIC 均值、RankICIR 均值、十分组多空累计收益
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 路径配置 ====================
VAL_DIR = "hs300_valuation_parquet"
PRICE_DIR = "cleaned_parquet"
OUTPUT_DIR = "rolling_backtest_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 工具函数 ====================
def extract_6digit(code_str: str) -> str:
    m = re.search(r"(\d{6})", str(code_str))
    return m.group(1) if m else None


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def load_and_merge_data():
    val_files = [f for f in os.listdir(VAL_DIR) if f.endswith(".parquet")]
    val_list = []
    for f in tqdm(val_files, desc="读取估值数据"):
        val_list.append(pd.read_parquet(os.path.join(VAL_DIR, f)))
    val = pd.concat(val_list, ignore_index=True)

    val["trade_date"] = normalize_date(val["trade_date"])
    val["code_6"] = val["code"].apply(extract_6digit)

    price_files = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
    price_list = []
    for f in tqdm(price_files, desc="读取行情数据"):
        price_list.append(pd.read_parquet(os.path.join(PRICE_DIR, f)))
    price = pd.concat(price_list, ignore_index=True)

    price["date"] = normalize_date(price["date"])
    price["code_6"] = price["code"].apply(extract_6digit)

    val_codes = set(val["code_6"].dropna())
    price_codes = set(price["code_6"].dropna())
    common_codes = val_codes & price_codes
    print(f"[对齐] 估值 {len(val_codes)} 只，行情 {len(price_codes)} 只，共同 {len(common_codes)} 只")

    val = val[val["code_6"].isin(common_codes)].copy()
    price = price[price["code_6"].isin(common_codes)].copy()

    merge_cols = ["date", "code_6", "open", "high", "low", "close",
                  "volume", "amount", "turn", "pctChg"]
    merged = pd.merge(
        price[merge_cols],
        val.drop(columns=["code"], errors="ignore"),
        left_on=["date", "code_6"],
        right_on=["trade_date", "code_6"],
        how="inner"
    )

    if "trade_date" in merged.columns:
        merged = merged.drop(columns=["trade_date"])

    merged = merged.sort_values(["code_6", "date"]).reset_index(drop=True)
    print(f"[合并] {len(merged)} 行，{merged['code_6'].nunique()} 只股票")
    return merged


# ==================== 因子计算 ====================
def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部 44 个因子。"""
    df = df.copy()
    df = df.sort_values(["code_6", "date"]).reset_index(drop=True)
    g = df.groupby("code_6", group_keys=False)

    # ========== 第一批 20 个 ==========
    df["EP_TTM"] = 1.0 / df["pe_ratio"].replace(0, np.nan)
    df["BP"] = 1.0 / df["pb_ratio"].replace(0, np.nan)
    df["SP_TTM"] = 1.0 / df["ps_ratio"].replace(0, np.nan)
    df["CFP_TTM"] = 1.0 / df["pcf_ratio"].replace(0, np.nan)
    df["PEG_proxy"] = df["pe_ratio_lyr"] / df["pe_ratio"].replace(0, np.nan) - 1.0

    df["LnMarketCap"] = np.log(df["market_cap"].clip(lower=1))
    df["LnFloatCap"] = np.log(df["circulating_market_cap"].clip(lower=1))
    df["Turnover_20D"] = g["turnover_ratio"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["Turnover_Vol"] = g["turnover_ratio"].transform(lambda x: x.rolling(20, min_periods=10).std())

    df["Momentum_20D"] = g["close"].transform(lambda x: x / x.shift(20) - 1.0)
    df["Momentum_60D"] = g["close"].transform(lambda x: x / x.shift(60) - 1.0)
    df["Reversal_5D"] = g["close"].transform(lambda x: x / x.shift(5) - 1.0)
    df["Volatility_20D"] = g["pctChg"].transform(lambda x: x.rolling(20, min_periods=10).std())

    df["PV_Corr_20D"] = g.apply(
        lambda x: x["close"].rolling(20, min_periods=10).corr(x["turnover_ratio"])
    ).reset_index(level=0, drop=True)

    df["Amount_20D_Ratio"] = g["turnover_ratio"].transform(
        lambda x: x.rolling(5, min_periods=3).mean() / x.rolling(20, min_periods=10).mean()
    )

    df["ILLIQ"] = g["turnover_ratio"].transform(
        lambda x: (1.0 / x.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    )

    df["EP_TTM_Change"] = g["EP_TTM"].transform(lambda x: x / x.shift(20) - 1.0)
    df["BP_Change"] = g["BP"].transform(lambda x: x / x.shift(20) - 1.0)

    df["PE_z"] = df.groupby("date")["pe_ratio"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df["PB_z"] = df.groupby("date")["pb_ratio"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df["PE_PB_Spread"] = df["PE_z"] - df["PB_z"]

    df["CFP_EP_Ratio"] = df["pcf_ratio"] / df["pe_ratio"].replace(0, np.nan)

    # ========== 第二批 24 个 ==========
    df["MOM_120"] = g["close"].transform(lambda x: x / x.shift(120) - 1.0)
    df["VOL_60"] = g["pctChg"].transform(lambda x: x.rolling(60, min_periods=30).std())
    df["AMOUNT_20"] = g["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    df["MAX_DROP_20"] = g["pctChg"].transform(lambda x: x.rolling(20, min_periods=10).min())

    df["F_PV_CORR"] = g.apply(
        lambda x: x["close"].rolling(20, min_periods=10).corr(x["volume"])
    ).reset_index(level=0, drop=True)

    df["F_PRICE_VOL_ELAS"] = g.apply(
        lambda x: x["close"].pct_change().rolling(20, min_periods=10).cov(
            x["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        ) / x["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        .rolling(20, min_periods=10).var().replace(0, np.nan)
    ).reset_index(level=0, drop=True)

    df["F_UP_VOL_RATIO"] = g.apply(
        lambda x: ((x["pctChg"] > 0) & (x["volume"] > x["volume"].shift(1)))
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_DOWN_SHRINK_RATIO"] = g.apply(
        lambda x: ((x["pctChg"] < 0) & (x["volume"] < x["volume"].shift(1)))
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_AMPLITUDE_20"] = g.apply(
        lambda x: ((x["high"] - x["low"]) / x["close"].shift(1))
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_UPPER_SHADOW"] = g.apply(
        lambda x: ((x["high"] - x[["open", "close"]].max(axis=1)) / x["close"])
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_LOWER_SHADOW"] = g.apply(
        lambda x: ((x[["open", "close"]].min(axis=1) - x["low"]) / x["close"])
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_GAP_FREQ"] = g.apply(
        lambda x: ((x["open"] / x["close"].shift(1) - 1).abs() > 0.01)
        .rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    df["F_SKEW_20"] = g["pctChg"].transform(lambda x: x.rolling(20, min_periods=10).skew())
    df["F_KURT_20"] = g["pctChg"].transform(lambda x: x.rolling(20, min_periods=10).kurt())

    df["F_UPSIDE_VOL"] = g["pctChg"].transform(
        lambda x: x[x > 0].rolling(20, min_periods=5).std()
    )
    df["F_DOWNSIDE_VOL"] = g["pctChg"].transform(
        lambda x: x[x < 0].rolling(20, min_periods=5).std()
    )

    mom20 = g["close"].transform(lambda x: x / x.shift(20) - 1.0)
    mom60 = g["close"].transform(lambda x: x / x.shift(60) - 1.0)
    df["F_MOM_ACCEL"] = mom20 - mom60
    df["F_MOM_STABILITY"] = g["close"].transform(
        lambda x: (x / x.shift(20) - 1.0).rolling(20, min_periods=10).std()
    )
    df["F_LONG_REVERSAL"] = -df["MOM_120"]
    df["F_SHORT_REVERSAL"] = -g["close"].transform(lambda x: x / x.shift(5) - 1.0)

    vol20 = g["pctChg"].transform(lambda x: x.rolling(20, min_periods=10).std())
    vol60 = g["pctChg"].transform(lambda x: x.rolling(60, min_periods=30).std())
    df["F_VOL_RATIO"] = vol20 / vol60.replace(0, np.nan)
    df["F_VOL_SKEW"] = df["F_SKEW_20"] * df["F_VOL_RATIO"]

    df["F_REL_STRENGTH"] = mom20 - df.groupby("date")["close"].transform(
        lambda x: x / x.shift(20) - 1.0
    ).groupby(df["date"]).transform("median")

    df["F_TREND_STRENGTH"] = g.apply(
        lambda x: x["close"].rolling(20, min_periods=10).apply(
            lambda y: np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2
            if len(y) > 1 else np.nan, raw=True
        )
    ).reset_index(level=0, drop=True)

    return df


FACTORS = [
    # 第一批 20 个
    "EP_TTM", "BP", "SP_TTM", "CFP_TTM", "PEG_proxy",
    "LnMarketCap", "LnFloatCap", "Turnover_20D", "Turnover_Vol",
    "Momentum_20D", "Momentum_60D", "Reversal_5D", "Volatility_20D",
    "PV_Corr_20D", "Amount_20D_Ratio", "ILLIQ",
    "EP_TTM_Change", "BP_Change", "PE_PB_Spread", "CFP_EP_Ratio",
    # 第二批 24 个
    "MOM_120", "VOL_60", "AMOUNT_20", "MAX_DROP_20",
    "F_PV_CORR", "F_PRICE_VOL_ELAS", "F_UP_VOL_RATIO", "F_DOWN_SHRINK_RATIO",
    "F_AMPLITUDE_20", "F_UPPER_SHADOW", "F_LOWER_SHADOW", "F_GAP_FREQ",
    "F_SKEW_20", "F_KURT_20", "F_UPSIDE_VOL", "F_DOWNSIDE_VOL",
    "F_MOM_ACCEL", "F_MOM_STABILITY", "F_LONG_REVERSAL", "F_SHORT_REVERSAL",
    "F_VOL_RATIO", "F_VOL_SKEW", "F_REL_STRENGTH", "F_TREND_STRENGTH",
]


# ==================== 未来收益 ====================
def compute_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["code_6", "date"]).reset_index(drop=True)
    df["open_next"] = df.groupby("code_6")["open"].shift(-1)
    df["open_next2"] = df.groupby("code_6")["open"].shift(-2)
    df["forward_ret"] = df["open_next2"] / df["open_next"] - 1.0
    return df


# ==================== 窗口划分 ====================
def build_windows(dates: list, window_months: int) -> list:
    """
    构建滚动窗口。
    返回 [(window_name, start_date, end_date), ...]
    start_date 和 end_date 均为 YYYY-MM-DD 字符串。
    """
    dates = pd.Series(sorted(dates))
    start = pd.Timestamp(dates.iloc[0])
    end = pd.Timestamp(dates.iloc[-1])

    windows = []
    current = start

    while current < end:
        next_boundary = current + pd.DateOffset(months=window_months)

        if next_boundary > end:
            next_boundary = end + pd.Timedelta(days=1)

        # 统一转为字符串比较，避免 str vs Timestamp 类型错误
        cur_str = current.strftime("%Y-%m-%d")
        next_str = next_boundary.strftime("%Y-%m-%d")

        win_dates = dates[(dates >= cur_str) & (dates < next_str)]
        if len(win_dates) > 0:
            win_start = win_dates.iloc[0]
            win_end = win_dates.iloc[-1]
            name = f"{pd.Timestamp(win_start).strftime('%Y-%m')}_{pd.Timestamp(win_end).strftime('%Y-%m')}"
            windows.append((name, win_start, win_end))

        current = next_boundary

    return windows


# ==================== 窗口内指标计算 ====================
def calc_window_metrics(df: pd.DataFrame, factor_col: str,
                        win_start, win_end, n_groups: int = 10):
    """
    计算单个窗口内的 RankIC 均值、RankICIR、十分组多空累计收益。
    """
    sub = df[(df["date"] >= win_start) & (df["date"] <= win_end)].copy()
    if len(sub) == 0:
        return None, None, None

    # ---- RankIC ----
    ic_list = []
    for d in sorted(sub["date"].unique()):
        day = sub[sub["date"] == d][[factor_col, "forward_ret"]].dropna()
        if len(day) < 10:
            continue
        ic, _ = spearmanr(day[factor_col], day["forward_ret"])
        if not np.isnan(ic):
            ic_list.append(ic)

    if len(ic_list) == 0:
        ic_mean, icir = np.nan, np.nan
    else:
        ic_arr = np.array(ic_list)
        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan

    # ---- 十分组多空（每日换仓） ----
    ls_returns = []
    for d in sorted(sub["date"].unique()):
        day = sub[sub["date"] == d][[factor_col, "forward_ret"]].dropna()
        if len(day) < n_groups * 2:
            continue
        try:
            day = day.copy()
            day["group"] = pd.qcut(day[factor_col], n_groups,
                                   labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        grp_ret = day.groupby("group")["forward_ret"].mean()
        r_short = grp_ret.get(1, np.nan)
        r_long = grp_ret.get(n_groups, np.nan)
        if pd.notna(r_long) and pd.notna(r_short):
            ls_returns.append(r_long - r_short)

    if len(ls_returns) == 0:
        ls_cum = np.nan
    else:
        ls_cum = np.prod(1 + np.array(ls_returns)) - 1.0

    return ic_mean, icir, ls_cum


def run_rolling_backtest(df: pd.DataFrame, window_months: int):
    """对全部因子执行滚动窗口回测。"""
    all_dates = sorted(df["date"].unique())
    windows = build_windows(all_dates, window_months)

    print(f"\n[窗口] {window_months} 个月，共 {len(windows)} 个窗口")
    for name, s, e in windows:
        print(f"  {name}: {s} ~ {e}")

    records = []

    for factor in tqdm(FACTORS, desc=f"因子滚动({window_months}M)"):
        if factor not in df.columns:
            continue

        for win_name, win_start, win_end in windows:
            ic_mean, icir, ls_cum = calc_window_metrics(
                df, factor, win_start, win_end
            )
            records.append({
                "factor": factor,
                "window": win_name,
                "window_start": win_start,
                "window_end": win_end,
                "rankic_mean": ic_mean,
                "rankicir": icir,
                "ls_cum_ret": ls_cum,
            })

    return pd.DataFrame(records)


# ==================== 可视化 ====================
def plot_factor_heatmap(result_df: pd.DataFrame, metric: str,
                        window_months: int, output_dir: str):
    """
    画因子 × 窗口热力图。
    行 = 因子，列 = 窗口，颜色 = 指标值。
    """
    pivot = result_df.pivot(index="factor", columns="window", values=metric)
    # 按均值排序
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.2),
                                     max(8, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(pivot.values)),
                   vmax=np.nanmax(np.abs(pivot.values)))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    ax.set_title(f"因子 × 窗口 {metric} 热力图（{window_months}M 窗口）")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    safe_metric = metric.replace("/", "_")
    path = os.path.join(output_dir, f"heatmap_{safe_metric}_{window_months}M.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [图] {path}")


def plot_ls_curve(result_df: pd.DataFrame, factor: str,
                  window_months: int, output_dir: str):
    """画单个因子在各窗口的多空累计收益柱状图。"""
    sub = result_df[result_df["factor"] == factor].sort_values("window_start")
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 1.2), 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sub["ls_cum_ret"]]
    ax.bar(range(len(sub)), sub["ls_cum_ret"].values, color=colors)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["window"].values, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{factor} 各窗口多空累计收益（{window_months}M）")
    ax.set_ylabel("累计收益")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    safe_name = factor.replace("/", "_")
    path = os.path.join(output_dir, f"ls_window_{safe_name}_{window_months}M.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("滚动窗口因子回测：44 个因子")
    print("=" * 60)

    df = load_and_merge_data()

    print("\n[计算] 44 个因子...")
    df = compute_all_factors(df)
    df = compute_forward_return(df)

    # ---- 半年窗口 ----
    print("\n" + "=" * 60)
    print("半年窗口（6M）")
    print("=" * 60)
    result_6m = run_rolling_backtest(df, window_months=6)
    result_6m.to_csv(os.path.join(OUTPUT_DIR, "rolling_6M.csv"),
                     index=False, encoding="utf-8-sig")

    # 汇总：各因子跨窗口均值
    summary_6m = result_6m.groupby("factor").agg(
        rankic_mean=("rankic_mean", "mean"),
        rankicir_mean=("rankicir", "mean"),
        ls_cum_mean=("ls_cum_ret", "mean"),
        rankic_std_across_windows=("rankic_mean", "std"),
        n_windows=("window", "count"),
    ).sort_values("rankicir_mean", ascending=False)
    summary_6m.to_csv(os.path.join(OUTPUT_DIR, "summary_6M.csv"),
                      encoding="utf-8-sig")
    print("\n[半年窗口汇总]")
    print(summary_6m.to_string())

    # ---- 一年窗口 ----
    print("\n" + "=" * 60)
    print("一年窗口（12M）")
    print("=" * 60)
    result_12m = run_rolling_backtest(df, window_months=12)
    result_12m.to_csv(os.path.join(OUTPUT_DIR, "rolling_12M.csv"),
                      index=False, encoding="utf-8-sig")

    summary_12m = result_12m.groupby("factor").agg(
        rankic_mean=("rankic_mean", "mean"),
        rankicir_mean=("rankicir", "mean"),
        ls_cum_mean=("ls_cum_ret", "mean"),
        rankic_std_across_windows=("rankic_mean", "std"),
        n_windows=("window", "count"),
    ).sort_values("rankicir_mean", ascending=False)
    summary_12m.to_csv(os.path.join(OUTPUT_DIR, "summary_12M.csv"),
                       encoding="utf-8-sig")
    print("\n[一年窗口汇总]")
    print(summary_12m.to_string())

    # ---- 热力图 ----
    print("\n[绘图] 生成热力图...")
    for metric in ["rankic_mean", "rankicir", "ls_cum_ret"]:
        plot_factor_heatmap(result_6m, metric, 6, OUTPUT_DIR)
        plot_factor_heatmap(result_12m, metric, 12, OUTPUT_DIR)

    # ---- 多空累计收益柱状图（对每个因子） ----
    print("[绘图] 生成各因子多空柱状图...")
    for factor in FACTORS:
        plot_ls_curve(result_6m, factor, 6, OUTPUT_DIR)
        plot_ls_curve(result_12m, factor, 12, OUTPUT_DIR)

    print(f"\n[输出] 全部结果已保存至 {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
"""
XGBoost 滚动回测：8 个精选因子（去掉 F_UPSIDE_VOL、AMOUNT_20）
初始训练窗口：12 个月
数据来源：cleaned_parquet（行情）+ hs300_valuation_parquet（估值）
按 6 位数字对齐两只文件夹的共同股票，合并后计算因子。
"""

import os
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ==================== 路径配置 ====================
VAL_DIR = "hs300_valuation_parquet"
PRICE_DIR = "cleaned_parquet"
OUTPUT_DIR = "backtest_output_xgb8_12m"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 配置区 ====================
FACTOR_COLS = [
    "EP_TTM",           # 估值
    "LnFloatCap",       # 规模
    "Turnover_20D",     # 流动性
    "Momentum_60D",     # 中期动量
    "Reversal_5D",      # 短期反转
    "Volatility_20D",   # 波动率
    "F_DOWNSIDE_VOL",   # 下行波动
    "F_AMPLITUDE_20",   # 价格形态
]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "gamma": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

LABEL_HORIZON = 20
TOP_QUANTILE = 0.3
BOTTOM_QUANTILE = 0.3

RETRAIN_MONTHS = 1
TOP_N = 10
BOTTOM_N = 10
TRADE_COST = 0.002
PERIODS_PER_YEAR = 12
MIN_HOLD_DAYS = 5

INITIAL_CAPITAL = 100.0

INITIAL_TRAIN_MONTHS = 12   # 改动：18 -> 12


# ==================== 工具函数 ====================
def extract_6digit(code_str: str) -> str:
    m = re.search(r"(\d{6})", str(code_str))
    return m.group(1) if m else None


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


# ================= 第一部分：数据加载与合并 =================

def load_and_merge_data():
    val_files = [f for f in os.listdir(VAL_DIR) if f.endswith(".parquet")]
    if not val_files:
        raise RuntimeError(f"{VAL_DIR} 中没有找到 parquet 文件")
    val = pd.concat(
        [pd.read_parquet(os.path.join(VAL_DIR, f)) for f in val_files],
        ignore_index=True
    )
    val["trade_date"] = normalize_date(val["trade_date"])
    val["code_6"] = val["code"].apply(extract_6digit)

    price_files = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
    if not price_files:
        raise RuntimeError(f"{PRICE_DIR} 中没有找到 parquet 文件")
    price = pd.concat(
        [pd.read_parquet(os.path.join(PRICE_DIR, f)) for f in price_files],
        ignore_index=True
    )
    price["date"] = normalize_date(price["date"])
    price["code_6"] = price["code"].apply(extract_6digit)

    val_codes = set(val["code_6"].dropna())
    price_codes = set(price["code_6"].dropna())
    common_codes = val_codes & price_codes
    print(f"[对齐] 估值 {len(val_codes)} 只，行情 {len(price_codes)} 只，共同 {len(common_codes)} 只")

    if len(common_codes) == 0:
        raise RuntimeError("两个文件夹没有共同股票，请检查代码格式")

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


# ================= 第二部分：因子计算 =================

def compute_factors(df):
    df = df.sort_values(["code_6", "date"]).copy()
    g = df.groupby("code_6")

    df["EP_TTM"] = 1.0 / df["pe_ratio"].replace(0, np.nan)
    df["LnFloatCap"] = np.log(df["circulating_market_cap"].clip(lower=1))

    df["Turnover_20D"] = g["turnover_ratio"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    df["Momentum_60D"] = g["close"].transform(lambda x: x / x.shift(60) - 1.0)
    df["Reversal_5D"] = g["close"].transform(lambda x: x / x.shift(5) - 1.0)

    df["Volatility_20D"] = g["pctChg"].transform(
        lambda x: x.rolling(20, min_periods=10).std() / 100
    )

    # 下行波动：用 where 保留索引
    df["F_DOWNSIDE_VOL"] = g["pctChg"].transform(
        lambda x: (x.where(x < 0) / 100).rolling(20, min_periods=5).std()
    )

    # 价格形态：先算振幅，再滚动平均，避免 groupby.apply 索引问题
    df["_amplitude"] = (df["high"] - df["low"]) / g["close"].shift(1)
    df["F_AMPLITUDE_20"] = g["_amplitude"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df = df.drop(columns=["_amplitude"])

    print("\n[因子缺失率]")
    for f in FACTOR_COLS:
        if f in df.columns:
            print(f"  {f}: {df[f].isna().mean():.2%}")
        else:
            print(f"  {f}: 不存在！")

    return df


def generate_labels(df):
    df = df.sort_values(["code_6", "date"]).copy()

    df["future_ret"] = df.groupby("code_6")["close"].transform(
        lambda x: x.shift(-LABEL_HORIZON) / x - 1
    )

    df["label"] = np.nan

    high_thresh = df.groupby("date")["future_ret"].transform(
        lambda s: s.quantile(1 - TOP_QUANTILE)
    )
    low_thresh = df.groupby("date")["future_ret"].transform(
        lambda s: s.quantile(BOTTOM_QUANTILE)
    )

    df.loc[df["future_ret"] >= high_thresh, "label"] = 1
    df.loc[df["future_ret"] <= low_thresh, "label"] = 0

    cross_size = df.groupby("date")["future_ret"].transform("size")
    df.loc[cross_size < 10, "label"] = np.nan

    return df


# ================= 第三部分：滚动训练与回测 =================

def get_rebalance_dates(all_dates, start_date, freq_months=1):
    dates = pd.Series(sorted(all_dates))
    rebalance = []
    current = pd.Timestamp(start_date)
    max_date = pd.Timestamp(dates.max())

    while current <= max_date:
        future = dates[dates >= current.strftime("%Y-%m-%d")]
        if len(future) > 0:
            rebalance.append(future.iloc[0])
        current = current + pd.DateOffset(months=freq_months)

    return sorted(set(pd.Timestamp(d) for d in rebalance))


def run_rolling_backtest(df):
    data = df.dropna(subset=FACTOR_COLS).copy()
    data = data.sort_values(["date", "code_6"]).reset_index(drop=True)

    if len(data) == 0:
        raise RuntimeError("因子全部为 NaN，请检查 compute_factors 和数据源字段")

    all_dates = sorted(data["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    min_date = pd.Timestamp(all_dates[0])
    max_date = pd.Timestamp(all_dates[-1])

    # 改动：初始训练窗口 18 -> 12 个月
    first_train_end = min_date + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
    rebalance_dates = get_rebalance_dates(all_dates, first_train_end, RETRAIN_MONTHS)

    print(f"\n数据范围: {min_date.date()} ~ {max_date.date()}")
    print(f"初始训练窗口: {INITIAL_TRAIN_MONTHS} 个月")
    print(f"调仓次数: {len(rebalance_dates)}")
    print(f"首次训练截止: {first_train_end.date()}")
    print(f"首次调仓: {rebalance_dates[0].date() if rebalance_dates else 'N/A'}")
    print(f"末次调仓: {rebalance_dates[-1].date() if rebalance_dates else 'N/A'}")

    period_records = []
    prev_holdings = set()
    feature_importance_records = []

    for i, reb_date in enumerate(rebalance_dates):
        reb_date = pd.Timestamp(reb_date)

        reb_idx = date_to_idx.get(reb_date.strftime("%Y-%m-%d"))
        if reb_idx is None or reb_idx < LABEL_HORIZON:
            continue
        train_cutoff = all_dates[reb_idx - LABEL_HORIZON]

        train_data = data[
            (data["date"] <= train_cutoff) & (data["label"].notna())
        ].copy()

        if len(train_data) < 5000:
            print(f"  跳过 {reb_date.date()}：训练样本不足 ({len(train_data)})")
            continue

        X_train = train_data[FACTOR_COLS].values
        y_train = train_data["label"].values.astype(int)

        if len(np.unique(y_train)) < 2:
            print(f"  跳过 {reb_date.date()}：训练集只有单一类别")
            continue

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = neg / pos if pos > 0 else 1.0

        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X_train, y_train, verbose=False)

        imp = model.feature_importances_
        feature_importance_records.append({
            "rebalance_date": reb_date,
            **{f: v for f, v in zip(FACTOR_COLS, imp)}
        })

        if i + 1 < len(rebalance_dates):
            next_date = pd.Timestamp(rebalance_dates[i + 1])
        else:
            next_date = max_date + pd.Timedelta(days=1)

        pred_slice = data[data["date"] == reb_date.strftime("%Y-%m-%d")].copy()
        if len(pred_slice) < TOP_N + BOTTOM_N:
            continue

        pred_slice["score"] = model.predict_proba(pred_slice[FACTOR_COLS].values)[:, 1]
        pred_slice = pred_slice.sort_values("score", ascending=False)

        long_stocks = pred_slice.head(TOP_N)["code_6"].tolist()
        short_stocks = pred_slice.tail(BOTTOM_N)["code_6"].tolist()

        hold_data = data[
            (data["date"] >= reb_date.strftime("%Y-%m-%d")) &
            (data["date"] < next_date.strftime("%Y-%m-%d"))
        ].copy()

        def calc_hold_return(g):
            g = g.sort_values("date")
            if len(g) < MIN_HOLD_DAYS:
                return np.nan
            return g["close"].iloc[-1] / g["close"].iloc[0] - 1

        hold_returns = hold_data.groupby("code_6").apply(calc_hold_return)

        long_rets = hold_returns.reindex(long_stocks).dropna()
        short_rets = hold_returns.reindex(short_stocks).dropna()

        if len(long_rets) == 0:
            continue

        long_ret = long_rets.mean()
        short_ret = short_rets.mean() if len(short_rets) > 0 else np.nan

        long_short_ret = long_ret - short_ret if not np.isnan(short_ret) else np.nan
        long_only_ret = long_ret

        current_holdings = set(long_stocks)
        if len(prev_holdings) == 0:
            turnover = 1.0
        else:
            changed = len(current_holdings.symmetric_difference(prev_holdings))
            turnover = changed / (2 * TOP_N)
        prev_holdings = current_holdings

        cost = turnover * TRADE_COST * 2
        long_only_ret_net = long_only_ret - cost
        long_short_ret_net = long_short_ret - cost if not np.isnan(long_short_ret) else np.nan

        period_records.append({
            "rebalance_date": reb_date,
            "next_date": next_date,
            "train_cutoff": train_cutoff,
            "n_train": len(train_data),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "long_only_ret": long_only_ret,
            "long_short_ret": long_short_ret,
            "long_only_ret_net": long_only_ret_net,
            "long_short_ret_net": long_short_ret_net,
            "turnover": turnover,
            "n_long": len(long_rets),
            "n_short": len(short_rets),
            "long_stocks": ",".join(long_stocks),
            "short_stocks": ",".join(short_stocks),
        })

        if (i + 1) % 6 == 0 or i == 0 or i == len(rebalance_dates) - 1:
            print(f"  [{i+1}/{len(rebalance_dates)}] {reb_date.date()} -> {next_date.date()}: "
                  f"多头 {long_ret:+.2%}, 空头 {short_ret:+.2%}, "
                  f"多空 {long_short_ret:+.2%}, 换手 {turnover:.2%}")

    if feature_importance_records:
        imp_df = pd.DataFrame(feature_importance_records)
        imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"),
                      index=False, encoding="utf-8-sig")
        print(f"\n[特征重要性] 已保存至 feature_importance.csv")

    return pd.DataFrame(period_records)


# ================= 第四部分：基准对比 =================

def compute_benchmark(df, period_df):
    """
    基准：每个调仓期内所有可用股票的等权平均收益。
    注意：这里用全样本（不 dropna FACTOR_COLS），确保基准不随模型因子集变化。
    """
    data = df.copy()
    data = data.sort_values(["date", "code_6"]).reset_index(drop=True)

    bench_returns = []
    for _, row in period_df.iterrows():
        reb_date = pd.Timestamp(row["rebalance_date"]).strftime("%Y-%m-%d")
        next_date = pd.Timestamp(row["next_date"]).strftime("%Y-%m-%d")

        hold_data = data[
            (data["date"] >= reb_date) & (data["date"] < next_date)
        ].copy()

        def calc_ret(g):
            g = g.sort_values("date")
            if len(g) < MIN_HOLD_DAYS:
                return np.nan
            return g["close"].iloc[-1] / g["close"].iloc[0] - 1

        rets = hold_data.groupby("code_6").apply(calc_ret).dropna()
        bench_returns.append(rets.mean() if len(rets) > 0 else np.nan)

    period_df = period_df.copy()
    period_df["benchmark_ret"] = bench_returns
    period_df["excess_ret"] = period_df["long_only_ret_net"] - period_df["benchmark_ret"]
    return period_df


# ================= 第五部分：绩效评估 =================

def evaluate_performance(period_df, ret_col, periods_per_year=12):
    rets = period_df[ret_col].dropna().values
    if len(rets) == 0:
        return {}

    nav = np.cumprod(1 + rets)
    total_return = nav[-1] - 1
    n_periods = len(rets)

    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    max_drawdown = drawdown.min()

    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else np.nan

    win_rate = (rets > 0).mean()
    avg_turnover = period_df["turnover"].mean()

    return {
        "总收益": total_return,
        "年化收益": annual_return,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "胜率": win_rate,
        "平均换手率": avg_turnover,
        "调仓次数": n_periods,
    }


def print_performance(name, perf):
    print(f"\n===== {name} =====")
    print(f"  总收益:     {perf['总收益']:.2%}")
    print(f"  年化收益:   {perf['年化收益']:.2%}")
    print(f"  最大回撤:   {perf['最大回撤']:.2%}")
    print(f"  夏普比率:   {perf['夏普比率']:.3f}")
    print(f"  胜率:       {perf['胜率']:.2%}")
    print(f"  平均换手率: {perf['平均换手率']:.2%}")
    print(f"  调仓次数:   {perf['调仓次数']}")


# ================= 第六部分：图表输出 =================

def plot_capital_curve(period_df, output_path):
    df = period_df.copy()

    df["capital_long"] = INITIAL_CAPITAL * (1 + df["long_only_ret_net"]).cumprod()
    df["capital_bench"] = INITIAL_CAPITAL * (1 + df["benchmark_ret"]).cumprod()
    df["capital_ls"] = INITIAL_CAPITAL * (1 + df["long_short_ret_net"].fillna(0)).cumprod()

    start_row = pd.DataFrame({
        "next_date": [df["rebalance_date"].iloc[0]],
        "capital_long": [INITIAL_CAPITAL],
        "capital_bench": [INITIAL_CAPITAL],
        "capital_ls": [INITIAL_CAPITAL],
    })
    plot_df = pd.concat([
        start_row,
        df[["next_date", "capital_long", "capital_bench", "capital_ls"]]
    ], ignore_index=True)
    plot_df = plot_df.rename(columns={"next_date": "date"})
    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(plot_df["date"], plot_df["capital_long"], marker="o",
            markersize=3, label="Long position", color="#1f77b4")
    ax.plot(plot_df["date"], plot_df["capital_bench"], marker="s",
            markersize=3, label="Market", color="#ff7f0e")
    ax.plot(plot_df["date"], plot_df["capital_ls"], marker="^",
            markersize=3, label="Long-short", color="#d62728")

    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8,
               label=f"Initial Capital {INITIAL_CAPITAL:.0f}")

    ax.set_title("Monthly Capital Curve", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"资金曲线图已保存至 {output_path}")

    csv_path = output_path.replace(".png", ".csv")
    plot_df.to_csv(csv_path, index=False)
    print(f"逐月资金量已保存至 {csv_path}")

    print(f"\n===== 每月资金变化（起点 {INITIAL_CAPITAL:.0f}） =====")
    print(f"{'日期':<12} {'多头资金':>10} {'基准资金':>10} {'多空资金':>10}")
    for _, row in plot_df.iterrows():
        print(f"{row['date'].date()!s:<12} "
              f"{row['capital_long']:>10.2f} "
              f"{row['capital_bench']:>10.2f} "
              f"{row['capital_ls']:>10.2f}")


def plot_feature_importance(output_dir):
    path = os.path.join(output_dir, "feature_importance.csv")
    if not os.path.exists(path):
        return
    imp_df = pd.read_csv(path, parse_dates=["rebalance_date"])
    factors = [c for c in imp_df.columns if c != "rebalance_date"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for f in factors:
        ax.plot(imp_df["rebalance_date"], imp_df[f], marker="o",
                markersize=2, label=f, alpha=0.7)
    ax.set_title("Feature Importance Over Time")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Importance")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"特征重要性时序图已保存至 {os.path.join(output_dir, 'feature_importance.png')}")


# ================= 主流程 =================

def main():
    print("=" * 60)
    print("XGBoost 滚动回测：8 个精选因子，12 个月初始训练")
    print("=" * 60)

    df = load_and_merge_data()

    print("\n计算因子...")
    df = compute_factors(df)

    print("\n生成标签...")
    df = generate_labels(df)

    print("\n===== 开始滚动回测 =====")
    period_df = run_rolling_backtest(df)

    if period_df.empty:
        print("回测没有产生任何持仓记录，退出")
        return

    print("\n计算基准收益...")
    period_df = compute_benchmark(df, period_df)

    period_df.to_csv(os.path.join(OUTPUT_DIR, "period_records.csv"), index=False)

    print("\n" + "=" * 50)
    print("绩效评估")
    print("=" * 50)

    perf_long = evaluate_performance(period_df, "long_only_ret_net", PERIODS_PER_YEAR)
    print_performance("纯多头（扣成本）", perf_long)

    perf_bench = evaluate_performance(period_df, "benchmark_ret", PERIODS_PER_YEAR)
    print_performance("基准（全市场等权）", perf_bench)

    perf_excess = evaluate_performance(period_df, "excess_ret", PERIODS_PER_YEAR)
    print_performance("超额收益（多头 - 基准）", perf_excess)

    perf_ls = evaluate_performance(period_df, "long_short_ret_net", PERIODS_PER_YEAR)
    if perf_ls:
        print_performance("多空组合（扣成本，理论）", perf_ls)

    print("\n" + "=" * 50)
    print("资金曲线")
    print("=" * 50)
    plot_capital_curve(period_df, os.path.join(OUTPUT_DIR, "capital_curve.png"))

    period_df["nav_long"] = (1 + period_df["long_only_ret_net"]).cumprod()
    period_df["nav_bench"] = (1 + period_df["benchmark_ret"]).cumprod()
    period_df["nav_excess"] = (1 + period_df["excess_ret"]).cumprod()
    period_df[["rebalance_date", "next_date", "nav_long", "nav_bench", "nav_excess",
               "long_only_ret_net", "benchmark_ret", "excess_ret", "turnover"]].to_csv(
        os.path.join(OUTPUT_DIR, "nav_curve.csv"), index=False
    )

    plot_feature_importance(OUTPUT_DIR)

    print(f"\n详细结果已保存至 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
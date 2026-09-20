import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")   # 无界面环境也能保存图片
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================= 配置 =================
INPUT_DIR = "./cleaned_parquet"
OUTPUT_DIR = "./backtest_output_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACTOR_COLS = ["MOM_20", "MOM_120", "VOL_20", "VOL_60",
               "TURN_20", "AMOUNT_20", "BP", "MAX_DROP_20"]

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

INITIAL_CAPITAL = 100.0   # 初始资金量


# ================= 第一部分：数据与因子 =================

def load_all_data():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"{INPUT_DIR} 中没有找到 parquet 文件")

    dfs = []
    for f in files:
        df = pd.read_parquet(os.path.join(INPUT_DIR, f))
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"加载完成: {len(files)} 只股票, {len(combined)} 条记录")
    return combined


def compute_factors(df):
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code")

    df["MOM_20"] = g["close"].transform(lambda x: x / x.shift(20) - 1)
    df["MOM_120"] = g["close"].transform(lambda x: x / x.shift(120) - 1)
    df["VOL_20"] = g["pctChg"].transform(lambda x: x.rolling(20).std() / 100)
    df["VOL_60"] = g["pctChg"].transform(lambda x: x.rolling(60).std() / 100)
    df["TURN_20"] = g["turn"].transform(lambda x: x.rolling(20).mean())
    df["AMOUNT_20"] = g["amount"].transform(lambda x: x.rolling(20).mean()).apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )
    df["BP"] = 1 / df["pbMRQ"].replace(0, np.nan)
    df["MAX_DROP_20"] = g["pctChg"].transform(lambda x: x.rolling(20).min() / 100)

    return df


def generate_labels(df):
    df = df.sort_values(["code", "date"]).copy()

    df["future_ret"] = df.groupby("code")["close"].transform(
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


# ================= 第二部分：滚动训练与回测 =================

def get_rebalance_dates(all_dates, start_date, freq_months=1):
    dates = pd.Series(sorted(all_dates))
    rebalance = []
    current = pd.Timestamp(start_date)
    max_date = dates.max()

    while current <= max_date:
        future = dates[dates >= current]
        if len(future) > 0:
            rebalance.append(future.iloc[0])
        current = current + pd.DateOffset(months=freq_months)

    return sorted(set(rebalance))


def run_rolling_backtest(df):
    data = df.dropna(subset=FACTOR_COLS).copy()
    data = data.sort_values(["date", "code"]).reset_index(drop=True)

    all_dates = sorted(data["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    min_date = pd.Timestamp(all_dates[0])
    max_date = pd.Timestamp(all_dates[-1])

    first_train_end = min_date + pd.DateOffset(months=18)
    rebalance_dates = get_rebalance_dates(all_dates, first_train_end, RETRAIN_MONTHS)

    print(f"\n调仓次数: {len(rebalance_dates)}")
    print(f"首次训练截止: {first_train_end.date()}")
    print(f"首次调仓: {rebalance_dates[0].date() if rebalance_dates else 'N/A'}")
    print(f"末次调仓: {rebalance_dates[-1].date() if rebalance_dates else 'N/A'}")

    period_records = []
    prev_holdings = set()

    for i, reb_date in enumerate(rebalance_dates):
        reb_date = pd.Timestamp(reb_date)

        reb_idx = date_to_idx.get(reb_date)
        if reb_idx is None or reb_idx < LABEL_HORIZON:
            print(f"  跳过 {reb_date.date()}：日期索引不足")
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

        if i + 1 < len(rebalance_dates):
            next_date = pd.Timestamp(rebalance_dates[i + 1])
        else:
            next_date = max_date + pd.Timedelta(days=1)

        pred_slice = data[data["date"] == reb_date].copy()
        if len(pred_slice) < TOP_N + BOTTOM_N:
            print(f"  跳过 {reb_date.date()}：预测截面样本不足")
            continue

        pred_slice["score"] = model.predict_proba(pred_slice[FACTOR_COLS].values)[:, 1]
        pred_slice = pred_slice.sort_values("score", ascending=False)

        long_stocks = pred_slice.head(TOP_N)["code"].tolist()
        short_stocks = pred_slice.tail(BOTTOM_N)["code"].tolist()

        hold_data = data[
            (data["date"] >= reb_date) & (data["date"] < next_date)
        ].copy()

        def calc_hold_return(g):
            g = g.sort_values("date")
            if len(g) < MIN_HOLD_DAYS:
                return np.nan
            return g["close"].iloc[-1] / g["close"].iloc[0] - 1

        hold_returns = hold_data.groupby("code").apply(calc_hold_return)

        long_rets = hold_returns.reindex(long_stocks).dropna()
        short_rets = hold_returns.reindex(short_stocks).dropna()

        if len(long_rets) == 0:
            print(f"  跳过 {reb_date.date()}：多头收益全为 NaN")
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

    return pd.DataFrame(period_records)


# ================= 第三部分：基准对比 =================

def compute_benchmark(df, period_df):
    data = df.dropna(subset=FACTOR_COLS).copy()
    data = data.sort_values(["date", "code"]).reset_index(drop=True)

    bench_returns = []
    for _, row in period_df.iterrows():
        reb_date = pd.Timestamp(row["rebalance_date"])
        next_date = pd.Timestamp(row["next_date"])

        hold_data = data[
            (data["date"] >= reb_date) & (data["date"] < next_date)
        ].copy()

        def calc_ret(g):
            g = g.sort_values("date")
            if len(g) < MIN_HOLD_DAYS:
                return np.nan
            return g["close"].iloc[-1] / g["close"].iloc[0] - 1

        rets = hold_data.groupby("code").apply(calc_ret).dropna()
        bench_returns.append(rets.mean() if len(rets) > 0 else np.nan)

    period_df = period_df.copy()
    period_df["benchmark_ret"] = bench_returns
    period_df["excess_ret"] = period_df["long_only_ret_net"] - period_df["benchmark_ret"]
    return period_df


# ================= 第四部分：绩效评估 =================

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


# ================= 第五部分：图表输出 =================

def plot_capital_curve(period_df, output_path):
    """
    画出以 INITIAL_CAPITAL 为起点的月度资金曲线。
    x轴为调仓日期，y轴为资金量。
    """
    df = period_df.copy()

    # 计算每期资金量
    df["capital_long"] = INITIAL_CAPITAL * (1 + df["long_only_ret_net"]).cumprod()
    df["capital_bench"] = INITIAL_CAPITAL * (1 + df["benchmark_ret"]).cumprod()
    df["capital_ls"] = INITIAL_CAPITAL * (1 + df["long_short_ret_net"].fillna(0)).cumprod()

    # 起点补一行（资金 = INITIAL_CAPITAL）
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

    # ---------- 画图 ----------
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

    # 同时把逐月资金量存成 CSV
    csv_path = output_path.replace(".png", ".csv")
    plot_df.to_csv(csv_path, index=False)
    print(f"逐月资金量已保存至 {csv_path}")

    # 打印每月资金变化
    print(f"\n===== 每月资金变化（起点 {INITIAL_CAPITAL:.0f}） =====")
    print(f"{'日期':<12} {'多头资金':>10} {'基准资金':>10} {'多空资金':>10}")
    for _, row in plot_df.iterrows():
        print(f"{row['date'].date()!s:<12} "
              f"{row['capital_long']:>10.2f} "
              f"{row['capital_bench']:>10.2f} "
              f"{row['capital_ls']:>10.2f}")


# ================= 主流程 =================

def main():
    df = load_all_data()
    print("\n计算因子...")
    df = compute_factors(df)
    print("生成标签...")
    df = generate_labels(df)

    print("\n===== 开始滚动回测（修复版，每月调仓） =====")
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

    # ---------- 资金曲线 ----------
    print("\n" + "=" * 50)
    print("资金曲线")
    print("=" * 50)
    plot_capital_curve(period_df, os.path.join(OUTPUT_DIR, "capital_curve.png"))

    # 保存详细净值数据
    period_df["nav_long"] = (1 + period_df["long_only_ret_net"]).cumprod()
    period_df["nav_bench"] = (1 + period_df["benchmark_ret"]).cumprod()
    period_df["nav_excess"] = (1 + period_df["excess_ret"]).cumprod()
    period_df[["rebalance_date", "next_date", "nav_long", "nav_bench", "nav_excess",
               "long_only_ret_net", "benchmark_ret", "excess_ret", "turnover"]].to_csv(
        os.path.join(OUTPUT_DIR, "nav_curve.csv"), index=False
    )

    print(f"\n详细结果已保存至 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
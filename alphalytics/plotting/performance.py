
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

from alphalytics.returns.relative import (capture_ratios, hit_rate, rolling_information_ratio,
                                          rolling_active_return,
                                          win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio)
from alphalytics.returns.metrics import annualized_rolling_return
from alphalytics.returns.capm import compute_capm
from alphalytics.utils import _infer_periods_per_year


####### PERFORMANCE VISUALS #######

def plot_growth(returns: pd.DataFrame, initial_value: int = 100,
    highlight: str = None, figsize: tuple = None,) -> tuple["plt.Figure", "plt.Axes"]:

    """Plot cumulative growth of one or more return series.

    Compounds periodic returns into a growth-of-investment chart,
    formatted with dollar values on the y-axis.

    Args:
        returns: Periodic returns with a DatetimeIndex. Each column is
            a separate series.
        initial_value: Starting investment amount in dollars.
        highlight: Column name to emphasize. All other series are dimmed
            to 30% opacity.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        The matplotlib Figure and Axes objects for further customization.
    """

    cumgrowth = (returns + 1).cumprod() * initial_value

    fig, ax = plt.subplots(figsize=figsize)

    for col in cumgrowth.columns:
        alpha = 0.3 if highlight and col != highlight else 1.0
        ax.plot(cumgrowth.index, cumgrowth[col], label=col, alpha=alpha)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="best")
    ax.set_title(f"Growth of ${initial_value:,}")
    ax.set_xlabel("")

    return fig, ax


def plot_cumulative_performance(returns: pd.DataFrame, title: str = None, periods_per_year: int = 252) -> None:

    from alphalytics.returns.aggregators import performance_table

    font_size, width = 10, 0.8

    # Calculate cumulative growth
    cumulative_growth = (returns + 1).cumprod()

    # Create subplots with adjusted height ratios
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10),
                                       gridspec_kw={'height_ratios': [1.5, 1, 1]})

    # --- Plot 1: Cumulative Growth ---
    cumulative_growth.plot(ax=ax1, xlabel="", title="Quantile Growth of $1", fontsize=font_size)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax1.legend(ncols=5, loc='upper left')

    # --- Plot 2: Cumulative Returns Bar Chart ---
    (cumulative_growth.iloc[-1] - 1).plot(kind="bar", ax=ax2, width=width,
                                        title="Quantile Cumulative Return", fontsize = font_size)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    for container in ax2.containers:
        ax2.bar_label(container, fmt='{:,.0%}', label_type='edge')
    plt.sca(ax2)  # Set current axis for xticks
    plt.xticks(rotation=0, ha='center')

    # --- Plot 3: Performance Table ---
    perf_table= performance_table(returns, periods_per_year)

    # Format table data as percentages
    table_data = perf_table.map(lambda x: f"{x:.2%}").reset_index()
    table_data.columns = ['Quantile'] + list(perf_table.columns)

    # Create table with proper formatting
    ax3.axis('off')
    table = ax3.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center'
    )

    # Format table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.1)  # Increase row height

    # Bold headers
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', size=10)
        cell.set_edgecolor('lightgray')

    # Add title and adjust layout
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()

    return fig, (ax1, ax2, ax3)


def plot_risk_return(returns: Union[pd.Series, pd.DataFrame],
                     periods_per_year=None, title="Risk-Return Analysis", fig_size=(3, 3), font_size=6,
                     legend_names=None, colors=None):
    """
    Plots a Risk-Return scatter chart for one or more return series.

    returns: pd.Series (single series) or pd.DataFrame (multiple series, one per column).
             Include the benchmark as a column in the DataFrame.
    periods_per_year: inferred from the index if None.
    legend_names: list of labels. Defaults to column/series names.
    colors: list of colors. If None, uses the active matplotlib style cycle.
    """
    # 1. Normalize into a DataFrame
    if isinstance(returns, pd.Series):
        df = returns.to_frame(name=returns.name or "Strategy")
    else:
        df = returns

    n = len(df.columns)

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df.index)

    # 2. Resolve legend names
    if legend_names is None:
        legend_names = list(df.columns)
    elif len(legend_names) < n:
        legend_names = list(legend_names) + [f"Series {i}" for i in range(len(legend_names), n)]

    # 3. Resolve colors — None means defer to the style cycle
    if colors is not None:
        colors = [colors[i % len(colors)] for i in range(n)]

    # 4. Helper function to calculate metrics
    def get_metrics(series, freq):
        ann_return = float(np.mean(series)) * freq
        ann_volatility = float(np.std(series)) * np.sqrt(freq)
        return ann_volatility, ann_return

    # 5. Calculate coordinates for all series
    all_risks, all_rets = [], []
    for col in df.columns:
        risk, ret = get_metrics(df[col].dropna(), periods_per_year)
        all_risks.append(risk)
        all_rets.append(ret)

    # 6. Create the Plot
    fig, ax = plt.subplots(figsize=fig_size)

    for i, (risk, ret) in enumerate(zip(all_risks, all_rets)):
        color = colors[i] if colors is not None else None
        ax.scatter(risk, ret, color=color, s=150, label=legend_names[i], zorder=5, edgecolors='black')

    # 7. Styling and formatting
    ax.set_title(title, fontsize=font_size+3, fontweight='bold', pad=5)
    ax.set_xlabel('Annualized Risk (Volatility)', fontsize=font_size)
    ax.set_ylabel('Annualized Return', fontsize=font_size)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    max_risk = max(all_risks) * 1.2
    max_ret = max(abs(r) for r in all_rets) * 1.2
    ax.set_xlim(0, max_risk)
    ax.set_ylim(min(-0.05, min(all_rets) * 1.2), max(0.05, max_ret))

    def percentage_formatter(x, _pos):
        return '{:,.1%}'.format(x)

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    ax.tick_params(axis='both', labelsize=font_size)

    plt.legend(loc='upper left', fontsize=font_size)
    plt.tight_layout()

    return fig, ax


def plot_xy_symmetric(data: pd.DataFrame, figsize=(3, 3), title=None, fontsize=6,
                      markers=None, markersize=150, colors=None,
                      center=1, min_distance=0.001):
    """Creates a symmetric scatter plot centered around a specified value with crosshairs.

    Designed for relative performance metrics (e.g., Up/Down Capture Ratios,
    Alpha vs. Beta) where axes share a common scale and a theoretical anchor
    point. Dynamically scales the axes symmetrically based on the maximum
    deviation from the center.

    Args:
        data: DataFrame containing the data to plot. The index labels the
            legend. First column maps to X-axis, second to Y-axis.
        figsize: The dimensions (width, height) of the figure in inches.
        title: The title of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.
        markersize: The size of the scatter plot markers.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        center: The anchor value where the crosshairs intersect. Pass a
            scalar to use the same center for both axes, or a tuple
            ``(center_x, center_y)`` for independent axis centers.
        min_distance: Minimum enforced distance from center to axis limits.

    Returns:
        The generated matplotlib Figure and Axes objects.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Support independent x/y centers
    if isinstance(center, (tuple, list)):
        center_x, center_y = center
    else:
        center_x = center_y = center

    # 1. Dynamically get column names by their integer index
    x_col = data.columns[0]
    y_col = data.columns[1]

    # --- NEW: Format markers securely ---
    if markers is None:
        markers = 'o' # Set a default uniform shape if nothing is passed

    if isinstance(markers, str):
        # Convert a single string (like 'D') into a dictionary for Seaborn's style parameter
        markers = {key: markers for key in data.index}
    # -----------------------------------

    # Seaborn handles the colors, shapes, and legend automatically
    sns.scatterplot(
        ax=ax, data=data, x=x_col, y=y_col,
        hue=data.index, style=data.index, markers=markers, # <-- Plural 'markers' used here
        s=markersize, zorder=5, palette=colors,
        edgecolors='black', linewidth=0.5
    )

    # Crosshairs & Styling
    ax.axhline(center_y, color='black', linestyle='--', alpha=0.3)
    ax.axvline(center_x, color='black', linestyle='--', alpha=0.3)

    ax.set_title(title, fontsize=fontsize+3, fontweight='bold', pad=8)
    ax.set_xlabel(x_col, fontsize=fontsize)
    ax.set_ylabel(y_col, fontsize=fontsize)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Find the absolute max distance from center for each axis
    max_dev_x = (data[x_col] - center_x).abs().max()
    max_dev_y = (data[y_col] - center_y).abs().max()

    # Add 20% padding and enforce minimum distance per axis
    dist_x = max(max_dev_x * 1.2, min_distance)
    dist_y = max(max_dev_y * 1.2, min_distance)

    # Apply the limits symmetrically around each center
    ax.set_xlim(center_x - dist_x, center_x + dist_x)
    ax.set_ylim(center_y - dist_y, center_y + dist_y)

    # Move legend to a consistent spot
    ax.legend(fontsize=fontsize, loc='upper left')

    plt.tight_layout()

    return fig, ax


def plot_capture_ratios(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
                        figsize=(3, 3), colors=None, title='Up vs. Down Market Capture',
                        fontsize=7, markers=['o']):
                        
    """High-level wrapper that calculates and plots Up/Down Capture ratios.

    Bridges ``capture_ratios()`` and ``plot_xy_symmetric()`` to generate a
    complete capture ratio scatter plot directly from raw periodic returns.

    Args:
        strategy_returns: Periodic returns of the strategies. Each column
            should represent a distinct strategy or asset.
        benchmark_returns: Periodic returns of the benchmark to calculate
            the capture against.
        figsize: The dimensions (width, height) of the figure in inches.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        title: The title displayed at the top of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.

    Returns:
        The generated matplotlib Figure and Axes objects.

    Example:
        >>> fig, ax = plot_capture_ratios(
        ...     strategy_returns=my_funds_df,
        ...     benchmark_returns=sp500_series,
        ...     title="Manager Capture Analysis"
        ... )
    """

    # Compute Capture Dataframe
    captures_df = capture_ratios(strategy_returns, benchmark_returns)[["Up Capture", "Down Capture"]]

    # Plot Capture DataFrame
    fig, ax = plot_xy_symmetric(data=captures_df,
                                figsize=figsize,
                                title=title,
                                fontsize=fontsize,
                                colors=colors,
                                markers=markers)


    return fig, ax


def plot_capture(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
                 figsize=(3, 3), colors=None, title='Down vs. Up Market Capture',
                 fontsize=7, markers=['o']):
    """High-level wrapper that plots Down Capture (x) vs Up Capture (y).

    Args:
        strategy_returns: Periodic returns of the strategies. Each column
            should represent a distinct strategy or asset.
        benchmark_returns: Periodic returns of the benchmark.
        figsize: The dimensions (width, height) of the figure in inches.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        title: The title displayed at the top of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.

    Returns:
        The generated matplotlib Figure and Axes objects.

    Example:
        >>> fig, ax = plot_capture(
        ...     strategy_returns=my_funds_df,
        ...     benchmark_returns=sp500_series,
        ... )
    """

    captures_df = capture_ratios(strategy_returns, benchmark_returns)[["Down Capture", "Up Capture"]]

    fig, ax = plot_xy_symmetric(data=captures_df,
                                figsize=figsize,
                                title=title,
                                fontsize=fontsize,
                                colors=colors,
                                markers=markers)

    return fig, ax


def plot_tail_capture(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
                      upper: float = 0.9, lower: float = 0.1,
                      figsize=(3, 3), colors=None, title='Down vs. Up Tail Capture',
                      fontsize=7, markers=['o']):
    """High-level wrapper that plots Tail Down Capture (x) vs Tail Up Capture (y).

    Tail captures are the strategy's conditional mean divided by the benchmark's
    conditional mean in the benchmark's upper and lower tail regimes.

    Args:
        strategy_returns: Periodic returns of the strategies. Each column
            should represent a distinct strategy or asset.
        benchmark_returns: Periodic returns of the benchmark.
        upper: Upper quantile threshold on the benchmark. Defaults to 0.9.
        lower: Lower quantile threshold on the benchmark. Defaults to 0.1.
        figsize: The dimensions (width, height) of the figure in inches.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        title: The title displayed at the top of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.

    Returns:
        The generated matplotlib Figure and Axes objects.

    Example:
        >>> fig, ax = plot_tail_capture(
        ...     strategy_returns=my_funds_df,
        ...     benchmark_returns=sp500_series,
        ... )
    """

    if isinstance(strategy_returns, pd.Series):
        strategy_returns = strategy_returns.to_frame()

    rows = {}
    for col in strategy_returns.columns:
        df = pd.concat([strategy_returns[col], benchmark_returns], axis=1, join='inner').dropna()
        if df.empty:
            rows[col] = {"Tail Down Capture": np.nan, "Tail Up Capture": np.nan}
            continue
        strat, bench = df.iloc[:, 0], df.iloc[:, 1]
        q_up = bench.quantile(upper)
        q_lo = bench.quantile(lower)
        up_mask = bench >= q_up
        lo_mask = bench <= q_lo
        bench_up_mean = bench[up_mask].mean() if up_mask.any() else np.nan
        bench_lo_mean = bench[lo_mask].mean() if lo_mask.any() else np.nan
        up_cap = strat[up_mask].mean() / bench_up_mean if bench_up_mean not in (0, np.nan) and not pd.isna(bench_up_mean) else np.nan
        lo_cap = strat[lo_mask].mean() / bench_lo_mean if bench_lo_mean not in (0, np.nan) and not pd.isna(bench_lo_mean) else np.nan
        rows[col] = {"Tail Down Capture": lo_cap, "Tail Up Capture": up_cap}

    tail_df = pd.DataFrame(rows).T[["Tail Down Capture", "Tail Up Capture"]]

    fig, ax = plot_xy_symmetric(data=tail_df,
                                figsize=figsize,
                                title=title,
                                fontsize=fontsize,
                                colors=colors,
                                markers=markers)

    return fig, ax


def plot_capm(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
              periods_per_year: int = 12,
              figsize=(3, 3), colors=None, title='Beta vs. Alpha',
              fontsize=7, markers=['o']):
    """High-level wrapper that calculates and plots CAPM Beta (x) vs Annualized Alpha (y).

    Args:
        strategy_returns: Periodic returns of the strategies. Each column
            should represent a distinct strategy or asset.
        benchmark_returns: Periodic returns of the benchmark.
        periods_per_year: Periods per year used to annualize alpha (e.g., 252
            for daily, 12 for monthly). Defaults to 12.
        figsize: The dimensions (width, height) of the figure in inches.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        title: The title displayed at the top of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.

    Returns:
        The generated matplotlib Figure and Axes objects.

    Example:
        >>> fig, ax = plot_capm(
        ...     strategy_returns=my_funds_df,
        ...     benchmark_returns=sp500_series,
        ... )
    """

    capm_df = compute_capm(strategy_returns, benchmark_returns, periods_per_year)[["Beta", "Annualized Alpha"]]

    fig, ax = plot_xy_symmetric(data=capm_df,
                                figsize=figsize,
                                title=title,
                                fontsize=fontsize,
                                colors=colors,
                                markers=markers,
                                center=(1, 0))

    return fig, ax


def plot_hit_rates(strategy_returns, benchmark_returns,
                   figsize=(3, 3),
                   title='Hit Rate',
                   font_size=7):
    """
    Calculates and plots the Batting Average (Win Rate) for Overall, Up, and Down markets.

    Parameters:
    - strategy_returns, benchmark_returns: pd.Series of periodic returns.
    - figsize: Tuple (width, height).
    - title: Chart title.
    - font_size: Base font size for labels.

    Returns:
    - fig, ax: Matplotlib objects.
    """

    # --- 1. Calculation Logic ---
    data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat = data.iloc[:, 0]
    bench = data.iloc[:, 1]

    # Calculate Win Rates
    overall_win = (strat > bench).mean()

    up_market_mask = bench > 0
    up_win = (strat[up_market_mask] > bench[up_market_mask]).mean() if up_market_mask.sum() > 0 else 0.0

    down_market_mask = bench < 0
    down_win = (strat[down_market_mask] > bench[down_market_mask]).mean() if down_market_mask.sum() > 0 else 0.0

    results = pd.Series(
        [overall_win, up_win, down_win],
        index=["Overall", "Up Market", "Down Market"]
    )

    # --- 2. Setup the Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Dynamically pull the active matplotlib color cycle
    mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Slice the color cycle to match the number of bars
    bars = ax.bar(results.index, results.values, color=mpl_colors[:len(results)], alpha=0.8, width=0.6)

    # --- 3. Styling ---
    ax.set_title(title, fontsize=font_size+3, fontweight='bold', pad=5)
    ax.set_ylim(0, 1.15) 
    ax.set_ylabel('Win Rate', fontsize=font_size)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.0%}',
                ha='center', va='bottom', fontsize=font_size+2, fontweight='bold')

    # --- 4. Add 50% "Coin Flip" Line ---
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    return fig, ax


def plot_win_loss(strategy_returns, benchmark_returns,
                  figsize=(3, 3),
                  title='Win/Loss Ratio',
                  font_size=7):
    """
    Calculates and plots the Win/Loss Ratio for Overall, Up, and Down markets.

    The Win/Loss Ratio is the average winning excess return divided by the
    average losing excess return (absolute value). A ratio > 1 means wins
    are larger than losses on average.

    Parameters:
    - strategy_returns, benchmark_returns: pd.Series of periodic returns.
    - figsize: Tuple (width, height).
    - title: Chart title.
    - font_size: Base font size for labels.

    Returns:
    - fig, ax: Matplotlib objects.
    """

    # --- 1. Calculation Logic ---
    results = pd.Series(
        [win_loss_ratio(strategy_returns, benchmark_returns),
         bull_win_loss_ratio(strategy_returns, benchmark_returns),
         bear_win_loss_ratio(strategy_returns, benchmark_returns)],
        index=["Overall", "Up Market", "Down Market"]
    )

    # --- 2. Setup the Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bars = ax.bar(results.index, results.values, color=mpl_colors[:len(results)], alpha=0.8, width=0.6)

    # --- 3. Styling ---
    ax.set_title(title, fontsize=font_size+3, fontweight='bold', pad=5)
    y_max = max(results.max() * 1.3, 1.5)
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Win/Loss Ratio', fontsize=font_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=font_size+2, fontweight='bold')

    # --- 4. Add 1.0 "Break Even" Line ---
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    return fig, ax


def plot_area(data: Union[pd.Series, pd.DataFrame], kind="area", fig_size=(8.25, 3),
              stacked=True, ylim_pad=(0.95, 1.02), tick_fontsize=7, y_fmt=None,
              colors: Union[List[str], str] = None, **kwargs):
    """
    Plot chart with area fill, optimized for both Series and DataFrames,
    with support for custom color palettes and keyword arguments.
    """

    # Calculate ymin and ymax safely, accounting for stacked DataFrames
    if isinstance(data, pd.DataFrame) and stacked:
        ymin = data.min().min() * (ylim_pad[1] if data.min().min() < 0 else ylim_pad[0])
        ymax = data.sum(axis=1).max() * ylim_pad[1]
    elif isinstance(data, pd.DataFrame):
        ymin = data.min().min() * (ylim_pad[1] if data.min().min() < 0 else ylim_pad[0])
        ymax = data.max().max() * ylim_pad[1]
    else:
        # Handle pd.Series (Added negative value logic here)
        ymin = data.min() * (ylim_pad[1] if data.min() < 0 else ylim_pad[0])
        ymax = data.max() * ylim_pad[1]

    fig, ax = plt.subplots(figsize=fig_size)

    # Passed **kwargs directly to the pandas plot method
    data.plot(kind=kind, ax=ax, xlabel="", ylim=(ymin, ymax),
              stacked=stacked, color=colors, **kwargs)

    # Remove chart junk
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    # Tick styling
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Format y-axis if a format string is provided (e.g., '${x:,.0f}')
    if y_fmt:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(y_fmt))

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    ax.grid(axis="x", visible=False)

    # Clean up the legend if dealing with a DataFrame
    if isinstance(data, pd.DataFrame):
        ax.legend(frameon=False, fontsize=tick_fontsize, loc="upper left")

    plt.tight_layout()
    return fig, ax



def plot_rolling_overunder(strategy_returns: pd.Series, benchmark_returns: pd.Series,
    window: int = 12, figsize: tuple = (10, 8)) -> tuple[plt.Figure, plt.Axes]:

    """Plot rolling over/under benchmark performance as a scatter chart.

    Computes rolling compounded returns for a strategy and benchmark,
    then plots each rolling window as a point. Points above the
    45-degree diagonal indicate strategy outperformance; points below
    indicate benchmark outperformance. Shaded regions and hit-rate
    statistics make relative performance easy to read at a glance.

    Args:
        strategy_returns: Periodic returns for the strategy in decimal form.
            The Series name is used as the display label.
        benchmark_returns: Periodic returns for the benchmark in decimal form.
            The Series name is used as the display label.
        window: Rolling window size in periods.
        figsize: Figure dimensions (width, height) in inches. A square shape
            (e.g. 8, 8) pairs best with the equal aspect ratio.

    Returns:
        The matplotlib Figure and Axes objects for further customization.

    Note:
        Rolling returns are computed using log-sum-exp for numerical
        stability: ``np.exp(np.log1p(r).rolling(w).sum()) - 1``.

        Colors for shaded regions, reference lines, and the legend frame
        are pulled dynamically from the active matplotlib style, so the
        chart adapts to any theme.
    """

    # Use the Series name if it exists; otherwise, fall back to the default argument
    strategy_name = strategy_returns.name or "Strategy"
    benchmark_name = benchmark_returns.name or "Benchmark"

    # 1. Fast rolling calculation
    strat_rolling = np.exp(np.log1p(strategy_returns).rolling(window=window).sum()) - 1
    bench_rolling = np.exp(np.log1p(benchmark_returns).rolling(window=window).sum()) - 1

    df = pd.concat([strat_rolling, bench_rolling], axis=1).dropna()
    df.columns = [strategy_name, benchmark_name]

    fig, ax = plt.subplots(figsize=figsize)

    # 2. Plot scatter
    ax.scatter(df[benchmark_name], df[strategy_name], alpha=0.8, s=40, zorder=3)

    # 3. Set dynamic axis limits for a perfect square grid
    min_val = min(df[strategy_name].min(), df[benchmark_name].min())
    max_val = max(df[strategy_name].max(), df[benchmark_name].max())
    padding = (max_val - min_val) * 0.05
    lower_lim = min_val - padding
    upper_lim = max_val + padding

    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_aspect("equal")

    # 4. Pull contextual colors dynamically directly from the active plot
    bg_color = ax.get_facecolor()
    text_color = plt.rcParams['text.color']
    grid_color = plt.rcParams['grid.color']

    # 5. Draw Reference Lines
    ax.axhline(0, color=grid_color, linewidth=1.5, zorder=2)
    ax.axvline(0, color=grid_color, linewidth=1.5, zorder=2)
    ax.plot([lower_lim, upper_lim], [lower_lim, upper_lim],
            color=text_color, linestyle='-', linewidth=2, zorder=2)

    # 6. Create Shaded Regions using your brand's color cycle
    x_fill = np.linspace(lower_lim, upper_lim, 100)
    ax.fill_between(x_fill, x_fill, upper_lim, color='C3', alpha=0.3, zorder=1)
    ax.fill_between(x_fill, lower_lim, x_fill, color='C4', alpha=0.3, zorder=1)

    # 7. Calculate Hit Rate Statistics
    total_periods = len(df)
    strat_wins = (df[strategy_name] > df[benchmark_name]).sum()
    bench_wins = total_periods - strat_wins

    strat_win_pct = strat_wins / total_periods
    bench_win_pct = bench_wins / total_periods

    # 8. Build an Inside Framed Legend
    strat_patch = mpatches.Patch(color='C3', alpha=0.3, label=f'{strategy_name} outperforms {strat_wins} times ({strat_win_pct:.2%})')
    bench_patch = mpatches.Patch(color='C4', alpha=0.3, label=f'{benchmark_name} outperforms {bench_wins} times ({bench_win_pct:.2%})')
    line_handle = mlines.Line2D([], [], color=text_color, linestyle='-', linewidth=2, label='Zero Excess Return')

    ax.legend(handles=[strat_patch, bench_patch, line_handle],
              loc='upper left', frameon=True, facecolor=bg_color,
              edgecolor=grid_color, framealpha=0.9, borderpad=1,
              fontsize=14)

    # 9. Formatting
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(f'{benchmark_name} Return')
    ax.set_ylabel(f'{strategy_name} Return')

    ax.set_title("Over/Under Benchmark Performance", loc='left', fontweight='bold', fontsize=17, pad=25)

    start_date = df.index.min().strftime('%m/%d/%Y')
    end_date = df.index.max().strftime('%m/%d/%Y')
    subtitle = f"Time Period: {start_date} to {end_date}    Rolling Window: {window} Periods"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=12, color=text_color)

    return fig, ax





def plot_rolling_metrics(data: Union[pd.Series, pd.DataFrame], figsize: tuple = (10, 6),
    is_percentage: bool = True, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a DataFrame or Series of rolling metrics using pure Matplotlib.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series data to plot. Each DataFrame column is a separate line.
    figsize : tuple, default (10, 6)
        Figure dimensions (width, height).
    is_percentage : bool, default True
        If True, formats the Y-axis as percentages (e.g., 5.0%).
    title : str, optional
        Title for the chart.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and Axes for further customization.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.Series):
        label = data.name if data.name is not None else "Series"
        ax.plot(data.index, data.values, label=label, linewidth=1.5)
    else:
        for column in data.columns:
            ax.plot(data.index, data[column].values, label=str(column), linewidth=1.5)

    if is_percentage:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.1%}"))

    if title:
        ax.set_title(title)

    if isinstance(data.index, pd.DatetimeIndex):
        fig.autofmt_xdate()

    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    return fig, ax


def plot_rolling_information_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, periods_per_year: int = None, figsize: tuple = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling information ratio for one or more strategies vs a benchmark.

    Computes the rolling annualized information ratio and plots it as a
    time series. For a single strategy, positive and negative regions are
    shaded for quick visual assessment.

    Args:
        strategy_returns: Periodic returns in decimal form. Series name or
            DataFrame column names are used as display labels.
        benchmark_returns: Periodic returns for the benchmark in decimal
            form. The Series name is used as the display label.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from
            the index frequency.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        The matplotlib Figure and Axes objects for further customization.
    """
    benchmark_name = benchmark_returns.name or "Benchmark"

    # Compute rolling IR via the standalone function
    rolling_ir = rolling_information_ratio(
        strategy_returns, benchmark_returns, window, periods_per_year
    )
    if isinstance(rolling_ir, pd.Series):
        rolling_ir = rolling_ir.to_frame(name=rolling_ir.name or "Strategy")

    # Base chart via the generic plotter (IR is not a percentage)
    fig, ax = plot_rolling_metrics(
        data=rolling_ir,
        figsize=figsize,
        is_percentage=False,
        title=None,
    )

    # --- IR-specific customizations layered on top ---

    text_color = plt.rcParams["text.color"]
    grid_color = plt.rcParams["grid.color"]
    bg_color = ax.get_facecolor()

    # Shade positive/negative regions for single-strategy plots
    if len(rolling_ir.columns) == 1:
        col = rolling_ir.columns[0]
        ax.fill_between(
            rolling_ir.index, rolling_ir[col], 0,
            where=(rolling_ir[col] >= 0), color="C0", alpha=0.3, zorder=1, interpolate=True,
        )
        ax.fill_between(
            rolling_ir.index, rolling_ir[col], 0,
            where=(rolling_ir[col] < 0), color="C1", alpha=0.3, zorder=1, interpolate=True,
        )

    # Neutral reference line
    ax.axhline(0, color=text_color, linewidth=1.5, linestyle="--", zorder=2, alpha=0.7)

    # Override formatting from base plotter
    ax.set_ylabel("Information Ratio")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.2f}"))

    ax.legend(
        loc="best", frameon=True, facecolor=bg_color,
        edgecolor=grid_color, framealpha=0.9, borderpad=1,
    )

    # Title and subtitle
    if isinstance(strategy_returns, pd.Series):
        title = strategy_returns.name or "Strategy"
    else:
        title = f"Rolling {window}-Period Information Ratio"
    ax.set_title(
        title,
        loc="left", fontweight="bold", fontsize=16, pad=25,
    )

    start_date = rolling_ir.index.min().strftime("%m/%d/%Y")
    end_date = rolling_ir.index.max().strftime("%m/%d/%Y")
    subtitle = f"Rolling {window}-Period Information Ratio | Benchmark: {benchmark_name} | Time Period: {start_date} to {end_date}"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=text_color)

    return fig, ax


def plot_rolling_active_return(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, periods_per_year: int = None, figsize: tuple = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling annualized active return for one or more strategies vs a benchmark.

    Computes the rolling annualized active return (strategy minus benchmark)
    and plots it as a time series. For a single strategy, positive and
    negative regions are shaded for quick visual assessment.

    Args:
        strategy_returns: Periodic returns in decimal form. Series name or
            DataFrame column names are used as display labels.
        benchmark_returns: Periodic returns for the benchmark in decimal
            form. The Series name is used as the display label.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from
            the index frequency.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        The matplotlib Figure and Axes objects for further customization.
    """
    benchmark_name = benchmark_returns.name or "Benchmark"

    # Compute rolling active return via the standalone function
    rolling_ar = rolling_active_return(
        strategy_returns, benchmark_returns, window, periods_per_year
    )
    if isinstance(rolling_ar, pd.Series):
        rolling_ar = rolling_ar.to_frame(name=rolling_ar.name or "Strategy")

    # Base chart via the generic plotter (active return is a percentage)
    fig, ax = plot_rolling_metrics(
        data=rolling_ar,
        figsize=figsize,
        is_percentage=True,
        title=None,
    )

    # --- Active-return-specific customizations layered on top ---

    text_color = plt.rcParams["text.color"]
    grid_color = plt.rcParams["grid.color"]
    bg_color = ax.get_facecolor()

    # Shade positive/negative regions for single-strategy plots
    if len(rolling_ar.columns) == 1:
        col = rolling_ar.columns[0]
        ax.fill_between(
            rolling_ar.index, rolling_ar[col], 0,
            where=(rolling_ar[col] >= 0), color="C0", alpha=0.3, zorder=1, interpolate=True,
        )
        ax.fill_between(
            rolling_ar.index, rolling_ar[col], 0,
            where=(rolling_ar[col] < 0), color="C1", alpha=0.3, zorder=1, interpolate=True,
        )

    # Neutral reference line
    ax.axhline(0, color=text_color, linewidth=1.5, linestyle="--", zorder=2, alpha=0.7)

    ax.set_ylabel("Active Return")

    ax.legend(
        loc="best", frameon=True, facecolor=bg_color,
        edgecolor=grid_color, framealpha=0.9, borderpad=1,
    )

    # Title and subtitle
    if isinstance(strategy_returns, pd.Series):
        title = strategy_returns.name or "Strategy"
    else:
        title = f"Rolling {window}-Period Active Return"
    ax.set_title(
        title,
        loc="left", fontweight="bold", fontsize=16, pad=25,
    )

    start_date = rolling_ar.index.min().strftime("%m/%d/%Y")
    end_date = rolling_ar.index.max().strftime("%m/%d/%Y")
    subtitle = f"Rolling {window}-Period Active Return | Benchmark: {benchmark_name} | Time Period: {start_date} to {end_date}"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=text_color)

    return fig, ax


def plot_rolling_return(strategy_returns: Union[pd.Series, pd.DataFrame],
    window: int, periods_per_year: int = None, figsize: tuple = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling annualized return for one or more strategies.

    Computes the annualized rolling return and plots it as a time series.
    For a single strategy, positive and negative regions are shaded for
    quick visual assessment.

    Args:
        strategy_returns: Periodic returns in decimal form. Series name or
            DataFrame column names are used as display labels.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from
            the index frequency.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        The matplotlib Figure and Axes objects for further customization.
    """
    rolling_ret = annualized_rolling_return(strategy_returns, window, periods_per_year)
    if isinstance(rolling_ret, pd.Series):
        rolling_ret = rolling_ret.to_frame(name=rolling_ret.name or "Strategy")

    fig, ax = plot_rolling_metrics(
        data=rolling_ret,
        figsize=figsize,
        is_percentage=True,
        title=None,
    )

    text_color = plt.rcParams["text.color"]
    grid_color = plt.rcParams["grid.color"]
    bg_color = ax.get_facecolor()

    if len(rolling_ret.columns) == 1:
        col = rolling_ret.columns[0]
        ax.fill_between(
            rolling_ret.index, rolling_ret[col], 0,
            where=(rolling_ret[col] >= 0), color="C0", alpha=0.3, zorder=1, interpolate=True,
        )
        ax.fill_between(
            rolling_ret.index, rolling_ret[col], 0,
            where=(rolling_ret[col] < 0), color="C1", alpha=0.3, zorder=1, interpolate=True,
        )

    ax.axhline(0, color=text_color, linewidth=1.5, linestyle="--", zorder=2, alpha=0.7)

    ax.set_ylabel("Annualized Return")

    ax.legend(
        loc="best", frameon=True, facecolor=bg_color,
        edgecolor=grid_color, framealpha=0.9, borderpad=1,
    )

    if isinstance(strategy_returns, pd.Series):
        title = strategy_returns.name or "Strategy"
    else:
        title = f"Rolling {window}-Period Annualized Return"
    ax.set_title(
        title,
        loc="left", fontweight="bold", fontsize=16, pad=25,
    )

    start_date = rolling_ret.index.min().strftime("%m/%d/%Y")
    end_date = rolling_ret.index.max().strftime("%m/%d/%Y")
    subtitle = f"Rolling {window}-Period Annualized Return | Time Period: {start_date} to {end_date}"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=text_color)

    return fig, ax


def plot_trailing_performance(returns: pd.DataFrame, periods_per_year: int = 12,
    figsize: tuple = (8, 4), title: str = 'Trailing Performance',
    fontsize: int = 9) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart of trailing returns (YTD, 1Y, 3Y, 5Y, CI) per strategy.

    Args:
        returns: Periodic returns of the strategies (DataFrame, one column per strategy).
        periods_per_year: Annualization factor (12 for monthly, 252 for daily).
        figsize: Figure dimensions (width, height) in inches.
        title: Chart title.
        fontsize: Base font size for labels and annotations.

    Returns:
        The matplotlib Figure and Axes objects for further customization.
    """
    from alphalytics.returns.aggregators import performance_table

    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "Strategy")

    perf = performance_table(returns, periods_per_year=periods_per_year)

    period_map = {"YTD": "YTD", "1 Year": "1-Year", "3 Year": "3-Year",
                  "5 Year": "5-Year", "SI": "CI"}
    cols = [c for c in period_map if c in perf.columns]
    data = perf[cols].rename(columns=period_map)

    fig, ax = plt.subplots(figsize=figsize)
    mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    strategies = list(data.index)
    periods = list(data.columns)
    n_strat = len(strategies)
    n_period = len(periods)

    x = np.arange(n_period)
    total_width = 0.8
    bar_width = total_width / max(n_strat, 1)

    for i, strat in enumerate(strategies):
        offsets = x - total_width / 2 + bar_width * (i + 0.5)
        values = data.loc[strat].values
        bars = ax.bar(offsets, values, bar_width,
                      label=strat, color=mpl_colors[i % len(mpl_colors)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if pd.isna(h):
                continue
            va = 'bottom' if h >= 0 else 'top'
            offset = 0.003 if h >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f'{h:.1%}', ha='center', va=va, fontsize=fontsize - 2)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=fontsize)
    ax.set_ylabel('Return', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 3, fontweight='bold', pad=8, loc='left')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=fontsize)
    if n_strat > 1:
        ax.legend(fontsize=fontsize, frameon=False)

    plt.tight_layout()
    return fig, ax


def plot_compare_drawdowns(drawdown_dict: dict, figsize: tuple = None,
    fontsize: int = 8, title: str = None) -> tuple:
    """
    Vertical bar chart of strategy drawdowns during benchmark stress periods.

    Takes the output of ``compare_drawdowns()`` and plots one subplot per
    benchmark drawdown period, showing each strategy's depth as a vertical bar.
    Depths are displayed as positive values (e.g. -0.15 shown as 15%).

    Args:
        drawdown_dict: Dict returned by ``compare_drawdowns()``.
            Keys are peak-date strings; values are DataFrames with a 'Depth' column.
        figsize: Figure dimensions (width, height). Defaults based on number of periods.
        fontsize: Base font size for labels and annotations.
        title: Overall figure title. Defaults to "Strategy Drawdowns During Benchmark Stress".

    Returns:
        The matplotlib Figure and array of Axes objects.
    """
    n = len(drawdown_dict)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No drawdown periods to display")
        return fig, ax

    if figsize is None:
        figsize = (max(4 * n, 6), 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (peak_date, df) in enumerate(drawdown_dict.items()):
        ax = axes[i]
        depths = df["Depth"].astype(float).abs()
        trough_date = df["Trough"].iloc[0] if "Trough" in df.columns else ""

        colors = [prop_cycle[j % len(prop_cycle)] for j in range(len(depths))]

        bars = ax.bar(
            x=depths.index,
            height=depths.values,
            color=colors,
            edgecolor="none",
        )

        for bar, val in zip(bars, depths.values):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.002,
                    f"{val:.1%}",
                    va="bottom", ha="center",
                    fontsize=fontsize - 1,
                    fontweight="bold",
                )

        ax.set_ylim(0, depths.max() * 1.15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{peak_date}  →  {trough_date}", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 1, rotation=45)
        ax.tick_params(axis="y", labelsize=fontsize)

    fig.suptitle(
        title or "Strategy Drawdowns During Benchmark Stress",
        fontsize=fontsize + 2, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    return fig, axes


def plot_compare_drawdown_volatility(drawdown_dict: dict, figsize: tuple = None,
    fontsize: int = 8, title: str = None) -> tuple:

    """
    Vertical bar chart of strategy volatility during benchmark stress periods.

    Takes the output of ``compare_drawdowns()`` and plots one subplot per
    benchmark drawdown period, showing each strategy's annualised volatility.

    Args:
        drawdown_dict: Dict returned by ``compare_drawdowns()``.
            Keys are peak-date strings; values are DataFrames with a
            'Volatility (Ann)' column.
        figsize: Figure dimensions (width, height). Defaults based on number of periods.
        fontsize: Base font size for labels and annotations.
        title: Overall figure title. Defaults to "Annualised Volatility During Benchmark Stress".

    Returns:
        The matplotlib Figure and array of Axes objects.
    """
    
    n = len(drawdown_dict)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No drawdown periods to display")
        return fig, ax

    if figsize is None:
        figsize = (max(4 * n, 6), 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (peak_date, df) in enumerate(drawdown_dict.items()):
        ax = axes[i]
        vols = df["Volatility (Ann)"].astype(float)
        trough_date = df["Trough"].iloc[0] if "Trough" in df.columns else ""

        colors = [prop_cycle[j % len(prop_cycle)] for j in range(len(vols))]

        bars = ax.bar(
            x=vols.index,
            height=vols.values,
            color=colors,
            edgecolor="none",
        )

        for bar, val in zip(bars, vols.values):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.002,
                    f"{val:.1%}",
                    va="bottom", ha="center",
                    fontsize=fontsize - 1,
                    fontweight="bold",
                )

        ax.set_ylim(0, vols.max() * 1.15 if vols.notna().any() else 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{peak_date}  →  {trough_date}", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 1, rotation=45)
        ax.tick_params(axis="y", labelsize=fontsize)

    fig.suptitle(
        title or "Annualised Volatility During Benchmark Stress",
        fontsize=fontsize + 2, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    return fig, axes


def plot_capture_hit_rate(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
                         figsize=(3, 3), colors=None, title='Overall Capture vs Hit Rate',
                         fontsize=7, markers=['o']):
    """High-level wrapper that calculates and plots Overall Capture vs Hit Rate.

    Bridges ``capture_ratios()``, ``hit_rate()``, and ``plot_xy_symmetric()``
    to generate a complete scatter plot directly from raw periodic returns.

    Args:
        strategy_returns: Periodic returns of the strategies. Each column
            should represent a distinct strategy or asset.
        benchmark_returns: Periodic returns of the benchmark to calculate
            the capture and hit rate against.
        figsize: The dimensions (width, height) of the figure in inches.
        colors: Color palette or list of colors. If None, Seaborn's default
            palette is used.
        title: The title displayed at the top of the plot.
        fontsize: Base font size for title, axis labels, ticks, and legend.
        markers: Marker style for scatter points. Pass a single string for
            uniform shapes, or a dict mapping index names to shapes.

    Returns:
        The generated matplotlib Figure and Axes objects.

    Example:
        >>> fig, ax = plot_capture_hit_rate(
        ...     strategy_returns=my_funds_df,
        ...     benchmark_returns=sp500_series,
        ...     title="Capture vs Batting Average"
        ... )
    """

    # Compute metrics
    captures = capture_ratios(strategy_returns, benchmark_returns)
    hit_rates = hit_rate(strategy_returns, benchmark_returns)

    # Build the two-column DataFrame expected by plot_xy_symmetric
    plot_data = pd.DataFrame({
        "Overall Capture": captures["Overall Capture"],
        "Hit Rate": hit_rates,
    })

    # Plot via plot_xy_symmetric (x centered at 1.0, y centered at 0.5)
    fig, ax = plot_xy_symmetric(data=plot_data,
                                figsize=figsize,
                                title=title,
                                fontsize=fontsize,
                                colors=colors,
                                markers=markers,
                                center=(1, 0.5))

    return fig, ax

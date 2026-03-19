
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

from alphalytics.returns.relative import capture_ratios, rolling_information_ratio
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


def plot_risk_return(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year=252,
                     title="Risk-Return Analysis", fig_size=(3, 3), font_size=6,
                     legend_names=["Strategy", "Benchmark"], colors=["orange", "blue"]):
    """
    Plots a Risk-Return scatter chart comparing a strategy to a benchmark.
    (Fixed to avoid UserWarning about FixedLocator)
    """

    # 1. Helper function to calculate metrics
    def get_metrics(returns, freq):
        ann_return = float(np.mean(returns)) * freq
        ann_volatility = float(np.std(returns)) * np.sqrt(freq)
        return ann_volatility, ann_return

    # 2. Calculate coordinates
    strat_risk, strat_ret = get_metrics(strategy_returns, periods_per_year)
    bench_risk, bench_ret = get_metrics(benchmark_returns, periods_per_year)

    # 3. Create the Plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot Strategy
    ax.scatter(strat_risk, strat_ret, color=colors[0], s=150, label=legend_names[0], zorder=5, edgecolors='black')

    # Plot Benchmark
    ax.scatter(bench_risk, bench_ret, color=colors[1], s=150, label=legend_names[1], zorder=5, edgecolors='black', alpha=0.7)

    # 4. Styling and formatting
    ax.set_title(title, fontsize=font_size+3, fontweight='bold', pad=5)
    ax.set_xlabel('Annualized Risk (Volatility)', fontsize=font_size)
    ax.set_ylabel('Annualized Return', fontsize=font_size)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add zero lines for reference
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Adjust axis limits to give some breathing room
    max_risk = max(strat_risk, bench_risk) * 1.2
    max_ret = max(abs(strat_ret), abs(bench_ret)) * 1.2
    ax.set_xlim(0, max_risk)
    ax.set_ylim(min(-0.05, min(strat_ret, bench_ret)*1.2), max(0.05, max_ret))

    # Function to format ticks as percentages
    def percentage_formatter(x, pos):
        return '{:,.1%}'.format(x)

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))

    # Apply font size to the ticks
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
        center: The anchor value where the X and Y crosshairs intersect.
        min_distance: Minimum enforced distance from center to axis limits.

    Returns:
        The generated matplotlib Figure and Axes objects.
    """

    fig, ax = plt.subplots(figsize=figsize)

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
    ax.axhline(center, color='black', linestyle='--', alpha=0.3)
    ax.axvline(center, color='black', linestyle='--', alpha=0.3)

    ax.set_title(title, fontsize=fontsize+3, fontweight='bold', pad=8)
    ax.set_xlabel(x_col, fontsize=fontsize)
    ax.set_ylabel(y_col, fontsize=fontsize)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Find the absolute max distance from center for both X and Y
    max_dev_x = (data[x_col] - center).abs().max()
    max_dev_y = (data[y_col] - center).abs().max()

    # Get the largest deviation, add 20% padding, and enforce minimum distance
    max_dist = max(max_dev_x, max_dev_y) * 1.2
    max_dist = max(max_dist, min_distance)

    # Apply the limits symmetrically
    ax.set_xlim(center - max_dist, center + max_dist)
    ax.set_ylim(center - max_dist, center + max_dist)

    # Move legend to a consistent spot
    ax.legend(fontsize=fontsize, loc='upper left')

    plt.tight_layout()

    return fig, ax


def plot_capture_ratios(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
                        figsize=(3, 3),
                        colors=None,
                        title='Up vs. Down Market Capture',
                        fontsize=7,
                        markers=['o']):
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


def plot_batting_averages(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                          figsize=(3, 3),
                          colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                          title='Batting Average',
                          font_size=7):
    """
    Calculates and plots the Batting Average (Win Rate) for Overall, Up, and Down markets.

    Parameters:
    - strategy_returns, benchmark_returns: pd.Series of periodic returns.
    - figsize: Tuple (width, height).
    - colors: List of colors for the bars.
    - title: Chart title.
    - font_size: Base font size for labels.

    Returns:
    - fig, ax: Matplotlib objects.
    """

    # --- 1. Calculation Logic ---
    # Align data
    data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat = data.iloc[:, 0]
    bench = data.iloc[:, 1]

    # Calculate Win Rates
    # Overall: % of periods where Strat > Bench
    overall_win = (strat > bench).mean()

    # Up Market: % of periods where Strat > Bench (given Bench > 0)
    up_market_mask = bench > 0
    if up_market_mask.sum() > 0:
        up_win = (strat[up_market_mask] > bench[up_market_mask]).mean()
    else:
        up_win = 0.0

    # Down Market: % of periods where Strat > Bench (given Bench < 0)
    down_market_mask = bench < 0
    if down_market_mask.sum() > 0:
        down_win = (strat[down_market_mask] > bench[down_market_mask]).mean()
    else:
        down_win = 0.0

    # Create the Results Series
    results = pd.Series(
        [overall_win, up_win, down_win],
        index=["Overall", "Up Market", "Down Market"]
    )

    # --- 2. Setup the Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Use the provided colors, repeating if necessary
    if len(colors) < len(results):
        colors = colors * len(results)

    bars = ax.bar(results.index, results.values, color=colors[:len(results)], alpha=0.8, width=0.6)

    # --- 3. Styling ---
    ax.set_title(title, fontsize=font_size+3, fontweight='bold', pad=5)
    ax.set_ylim(0, 1.15) # Give headroom for labels
    ax.set_ylabel('Win Rate', fontsize=font_size)

    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust Font Sizes
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Add Labels on Top of Bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.0%}',
                ha='center', va='bottom', fontsize=font_size+2, fontweight='bold')

    # --- 4. Add 50% "Coin Flip" Line ---
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

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
              fontsize=12)

    # 9. Formatting
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(f'{benchmark_name} Return')
    ax.set_ylabel(f'{strategy_name} Return')

    ax.set_title("Over/Under Benchmark Performance", loc='left', fontweight='bold', fontsize=16, pad=25)

    start_date = df.index.min().strftime('%m/%d/%Y')
    end_date = df.index.max().strftime('%m/%d/%Y')
    subtitle = f"Time Period: {start_date} to {end_date}    Rolling Window: {window} Periods"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=text_color)

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
    ax.set_title(
        f"Rolling {window}-Period Information Ratio",
        loc="left", fontweight="bold", fontsize=16, pad=25,
    )

    start_date = rolling_ir.index.min().strftime("%m/%d/%Y")
    end_date = rolling_ir.index.max().strftime("%m/%d/%Y")
    subtitle = f"Benchmark: {benchmark_name} | Time Period: {start_date} to {end_date}"

    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=text_color)

    return fig, ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

import seaborn as sns

import quantstats as qs
from scipy.stats import norm, probplot

from .performance_metrics import compute_capm, performance_table, capture_ratios
from .ic_analysis import cs_spearmanr, compute_ic_stats, factor_decay
from .quantile_analysis import fwd_quantile_stats
from .turnover_analysis import compute_quantiles_turnover

# =============== RAW FACTOR DATA ANALYSIS ============== #

def plot_factor_data(factor_data: pd.DataFrame):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    factor_data.mean(axis=1).plot(title="Mean Factor Value & Distribution", ax=ax[0], xlabel="", color="black")
    factor_data.mean(axis=1).hist(ax=ax[1], bins=30, color="skyblue")

    return ax
    


def plot_cumulative_performance(returns: pd.DataFrame, title: str = None, periods_per_year: int = 252) -> None:
    
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
    performance_table= compute_performance_table(returns, periods_per_year)
    
    # Format table data as percentages
    table_data = performance_table.map(lambda x: f"{x:.2%}").reset_index()
    table_data.columns = ['Quantile'] + list(performance_table.columns)
    
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
    plt.show()



def plot_quantiles_risk_metrics(quantile_returns: pd.DataFrame, benchmark: pd.Series = None, periods_per_year: int = 52) -> None:
    """
    Plot various risk metrics for quantile returns.

    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        DataFrame containing returns for different quantiles (columns) with datetime index.
    benchmark : pd.Series, optional
        Benchmark returns series with matching datetime index. If None, defaults to equal-weighted universe.
    periods : int, default 52
        Number of periods per year (e.g., 52 for weekly data, 252 for daily data).

    Returns:
    --------
    None
        This function does not return any value. It displays the plots.

    Notes:
    ------
    - The function creates a figure with multiple subplots to display various risk metrics.
    - Metrics include Annualized Return, Volatility, Sharpe Ratio, Beta, Alpha, Tail Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, and Drawdowns.
    - The CAPM metrics (Beta and Alpha) are computed using the `compute_capm` function.
    """
    font_size, width = 10, 0.8
    
    # Create a figure and define the gridspec layout
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.75, 0.75, 0.75, 2])  # 2 rows for 2x3 + 1 row for full plot

    ax1 = fig.add_subplot(gs[0, 0])
    qs.stats.cagr(quantile_returns).plot.bar(ax=ax1, title="Annualized Return", width=0.8, fontsize=font_size)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    ax2 = fig.add_subplot(gs[0, 1])
    qs.stats.volatility(quantile_returns, periods=periods_per_year).plot.bar(ax=ax2, title="Volatility", width=0.8, fontsize=font_size)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    ax3 = fig.add_subplot(gs[0, 2])
    qs.stats.sharpe(quantile_returns).plot.bar(ax=ax3, title="Sharpe Ratio", width=0.8, fontsize=font_size)
    ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}"))
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

    capm_table = compute_capm(quantile_returns, benchmark=benchmark)
    ax4 = fig.add_subplot(gs[1, 0])
    capm_table["Beta"].plot.bar(ax=ax4, title="Beta", width=0.8, fontsize=font_size, xlabel="")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

    ax5 = fig.add_subplot(gs[1, 1])
    capm_table["Alpha"].plot.bar(ax=ax5, title="Alpha", width=0.8, fontsize=font_size, xlabel="")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)

    ax6 = fig.add_subplot(gs[1, 2])
    qs.stats.tail_ratio(quantile_returns).plot.bar(ax=ax6, title="Tail Ratio", width=0.8, fontsize=font_size)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)

    ax7 = fig.add_subplot(gs[2, 0])
    qs.stats.sortino(quantile_returns).plot.bar(ax=ax7, title="Sortino Ratio", width=0.8, fontsize=font_size)
    ax7.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}"))
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0)

    ax8 = fig.add_subplot(gs[2, 1])
    qs.stats.calmar(quantile_returns).plot.bar(ax=ax8, title="Calmar Ratio", width=0.8, fontsize=font_size)
    ax8.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}"))
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=0)

    ax9 = fig.add_subplot(gs[2, 2])
    qs.stats.max_drawdown(quantile_returns).abs().plot.bar(ax=ax9, title="Max Drawdown", width=0.8, fontsize=font_size)
    ax9.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax9.set_xticklabels(ax9.get_xticklabels(), rotation=0)

    # Create the full-width plot below the grid
    ax_full = fig.add_subplot(gs[3, :])
    qs.stats.to_drawdown_series(quantile_returns).plot(ax=ax_full, title="Drawdowns", legend=False, xlabel="", fontsize=font_size)
    ax_full.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax_full.legend(ncols=5, loc='best', fontsize=font_size)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_quantile_correlations(returns: pd.DataFrame, title: str = None) -> None:

    """
    Plot quantile correlations and returns analysis.

    The demeaning process helps analyze relative performance by removing the overall trend across all quantiles for each time period.
    
    Args:
        returns (pd.DataFrame): DataFrame with quantile returns 
        title (str, optional): Custom title for the plot
    """

    font_size, width = 10, 0.8

    # Calculate demeaned returns
    demeaned_rets = returns.sub(returns.mean(axis=1), axis=0)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot 1: Demeaned Q1 and Q5 returns
    demeaned_rets.iloc[:, [0, -1]].plot(
        ax=ax1,
        title="Demeaned Long/Short Quantile Returns",
        xlabel="", 
        fontsize=font_size
    )
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # xmax=1 for decimal input
    ax1.legend(ncols=5, loc='best', fontsize=font_size)

    # Plot 2: Demeaned correlations bar plot
    demeaned_rets.corr().iloc[0].plot(
        kind="bar",
        ax=ax2,
        width=width,
        title="Demeaned Correlations to Q1",
        xlabel="",
        fontsize=font_size
    )
    ax2.set_xticklabels([])  # Hide x-axis labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='{:,.2f}', fontsize=font_size)  # Fixed format string (f not comma)

    # Plot 3: Original correlations bar plot
    returns.corr().iloc[0].plot(
        kind="bar",
        ax=ax3,
        width=width,
        title="Correlations to Q1",
        fontsize=font_size
    )
    for container in ax3.containers:
        ax3.bar_label(container, fmt='{:,.2f}', label_type='edge', fontsize=font_size)

    # Adjust layout
    plt.setp(ax3.get_xticklabels(), rotation=0, ha='center')  # Changed ha='right' to 'center'
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_spearman_rank(spearmanr_ts: pd.Series, window: int, ax = None):

    data = pd.concat([spearmanr_ts, spearmanr_ts.rolling(window).mean()], axis=1).dropna()
    data.columns = ["Spearman Correlation", "Rolling Mean"]

    # Use provided axis or create new one
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    # Plot the first series on the left y-axis
    ax1.plot(data["Spearman Correlation"], color='skyblue', alpha = 0.7)
    ax1.set_ylabel('IC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Center the left y-axis around 0
    data1_max = data["Spearman Correlation"].max()
    data1_min = data["Spearman Correlation"].min()
    data1_abs_max = max(abs(data1_max), abs(data1_min))
    ax1.set_ylim(-data1_abs_max, data1_abs_max)
    ax1.legend(["Spearman Correlation (IC)"], loc='upper left')
    ax1.hlines(0, data.index[0], data.index[-1], color='gray', linestyle='--', alpha=0.5)

    # Create a secondary y-axis for the second series
    ax2 = ax1.twinx()
    ax2.plot(data["Rolling Mean"], color='black')
    ax2.set_ylabel('Rolling IC', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(["Rolling Mean"], loc='upper right')

    # Center the right y-axis around 0
    data2_max = data["Rolling Mean"].max()
    data2_min = data["Rolling Mean"].min()
    data2_abs_max = max(abs(data2_max), abs(data2_min))
    ax2.set_ylim(-data2_abs_max, data2_abs_max)

    return ax1


def plot_ic_hist(corr_ts:pd.Series, ax = None):

    # Calculate mean and standard deviation of the IC values
    mean_ic = corr_ts.mean()
    std_ic = corr_ts.std()

    # Create the histogram

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax

    corr_ts.hist(bins=30, density=True, alpha=0.6, color='skyblue', label='IC Histogram', ax=ax)

    # Fit a normal distribution to the data and plot the curve
    norm_mean = 0 # Mean of the normal distribution
    x = np.linspace(min(corr_ts), max(corr_ts), corr_ts.size)
    ax.plot(x, norm.pdf(x, norm_mean, std_ic), 'b-', lw=2, label='Normal Fit')

    # Add vertical line at x=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

    # Customize the plot
    ax.set_title('IC Distribution', fontsize=14)
    ax.set_xlabel('IC', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend([f'Mean {mean_ic:.3f}\nStd {std_ic:.3f}', 'Normal Fit'], loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    return ax


def qqplot_ic(corr_ts: pd.Series, ax=None):
    
    # Set up the axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # Create the Q-Q plot directly on the specified axis
    probplot(corr_ts, dist="norm", plot=ax)
    
    # Customize the plot
    ax.set_title('IC Normal Dist. Q-Q')
    ax.set_xlabel('Normal Distribution Quantile')
    ax.set_ylabel('Observed Quantile')
    
    # Customize points and reference line
    ax.get_lines()[0].set_color('skyblue')  # Points
    ax.get_lines()[0].set_marker('o')    # Ensure circular markers
    ax.get_lines()[1].set_color('red')   # Reference line
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_ic_summary(factors: pd.DataFrame, returns: pd.DataFrame, window: int, 
                    periods_label: str = "Days") -> tuple[plt.Figure, np.ndarray]:
    # Calculate Spearman rank correlations
    spearmanr_ts = cs_spearmanr(factors, returns)
    
    # Compute statistics (placeholder function)
    corr_stats = compute_ic_stats(factors, returns)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    ax3 = fig.add_subplot(gs[1, :])  # Bottom, full width
    
    # Top subplots: Plot mean factors and returns
    plot_ic_hist(spearmanr_ts, ax = ax1)
    qqplot_ic(spearmanr_ts, ax  = ax2)
    
    # Bottom subplot: Spearman rank plot
    plot_spearman_rank(spearmanr_ts, window=window, ax=ax3)
    
    # Format and add table
    table_data = corr_stats.apply(lambda x: x.map(lambda y: f"{y:.4f}") if x.dtype.kind in 'if' else x).reset_index()
    plot_table = ax3.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="bottom",
        cellLoc="center",
        bbox=[0, -0.40, 1, 0.15]
    )
    
    # Style table
    plot_table.auto_set_font_size(False)
    plot_table.set_fontsize(10)
    plot_table.scale(1, 1)
    for i, key in enumerate(table_data.columns):
        cell = plot_table[0, i]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e6f2ff")
    
    fig.suptitle("Information Coefficient (Spearman Rank) Analysis", y=1.02, fontsize = 16)
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()
    
    
    # Return figure and axes array
    return fig, np.array([ax1, ax2, ax3])


def plot_factor_decay(factor_data:pd.DataFrame, returns:pd.DataFrame, max_horizon: int, periods_label:str = "Days") -> None:
        
    decay = factor_decay(factor_data, returns, max_horizon)
    # Create subplots
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    # Plot IC on the first subplot (ax[0])
    decay['IC'].plot(ax=ax[0], marker='o', linestyle='-', color='skyblue')
    ax[0].set_title('Factor IC Decay', fontsize=14)  # Larger title
    ax[0].set_ylabel('Average IC')
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # Plot p-values on the second subplot (ax[1])
    decay['T_PValue'].plot(ax=ax[1], marker='o', linestyle='-', color='black')
    ax[1].set_title('P-Value of IC', fontsize=14)  # Larger title
    ax[1].set_xlabel(f'Horizon({periods_label})')
    ax[1].set_ylabel('P-Value')
    ax[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # Add a horizontal line at y=0 for IC
    ax[0].axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Axial 0')

    # Add a horizontal line at p=0.05 for significance
    ax[1].axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='p=0.05')
    ax[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_forward_returns(returns: pd.DataFrame, quantiles: pd.DataFrame, periods: list, 
                        periods_label: str = "Days", fig_size: tuple = (10, 6)):
    """
    Plot forward returns and risk-adjusted returns across multiple periods.

    This function generates a grid of bar plots, with the top row showing cumulative
    forward returns and the bottom row showing risk-adjusted returns (mean returns
    divided by standard deviation) for specified periods.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame containing returns data, typically with dates as index and assets/quantiles as columns.
    quantiles : pd.DataFrame
        DataFrame defining quantile groups for the returns, aligned with `returns`.
    periods : list
        List of integers representing the forward periods to analyze (e.g., [17, 26, 52]).
    periods_label : str, optional
        Label for the periods in the subplot titles (e.g., "Days", "Weeks"). Default is "Days".
    fig_size : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 6).

    Notes
    -----
    - The function assumes `compute_forward_quantile_returns` is defined elsewhere and
      returns a DataFrame of forward returns.
    - Subplot titles are formatted as "{period}-{periods_label}" (e.g., "17-Days").
    - Y-axes are formatted as percentages, and horizontal grids are added for readability.
    - The main figure title is "Forward Returns Analysis".
    """
    n_rows = 2
    n_columns = len(periods)
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_rows, n_columns, figsize=fig_size)
    fig.text(-0.02, 0.6, 'Cumulative Return', rotation=90, fontsize=12)
    fig.text(-0.02, 0.1, 'Risk-Adjusted Return', rotation=90, fontsize=12)
    
    quantile_list = range(1, quantiles.nunique().max()+1, 1)
    for idx, period in enumerate(periods):
        #print(f"Calculating forward returns for {period} periods...")
        # Assuming compute_forward_quantile_returns is defined elsewhere
        forward_stats_df = fwd_quantile_stats(returns, quantiles, period)
        mean_rets = forward_stats_df["Return"]
        mean_rarets = forward_stats_df["Risk-Adjusted Return"]

        # Plot bars
        mean_rets.plot(kind="bar", ax=ax[0, idx], title=f"{period}-{periods_label}", 
                      xlabel="", width=0.9, legend=False, color='black')
        mean_rarets.plot(kind="bar", ax=ax[1, idx], title=f"{period}-{periods_label}", 
                        xlabel="", width=0.9, legend=False, color='skyblue')

        # Adjust tick parameters
        for row in range(n_rows):
            ax[row, idx].tick_params(axis='both', labelsize=8)
            ax[row, idx].set_xticklabels(ax[row, idx].get_xticklabels(), rotation=0)
            ax[row, idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2}'))
            ax[row, idx].grid(axis='y', linestyle='--', alpha=0.7)
            ax[row, idx].title.set_fontsize(10)  # Set title font size

    # Add main figure title
    plt.suptitle("Quantile Forward Returns Analysis", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.show()


def plot_quantiles_annual_turnover(quantiles:pd.DataFrame, periods_per_year:int, fig_size = (10,4)):

    ax = (compute_quantiles_turnover(quantiles).mean() * periods_per_year).plot(kind="bar", title= "Quantiles Turnover", figsize=fig_size, color='black')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()


def plot_risk_return(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year=252, 
                     title="Risk-Return Analysis", fig_size=(3, 3), font_size=6, 
                     legend_names=["Strategy", "Benchmark"], colors=["orange", "blue"]):
    """
    Plots a Risk-Return scatter chart comparing a strategy to a benchmark.
    (Fixed to avoid UserWarning about FixedLocator)
    """
    
    # 1. Helper function to calculate metrics
    def get_metrics(returns, freq):
        ann_return = np.mean(returns) * freq
        ann_volatility = np.std(returns) * np.sqrt(freq)
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

    # --- FIX STARTS HERE ---
    
    # Function to format ticks as percentages
    def percentage_formatter(x, pos):
        return '{:,.1%}'.format(x)

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    
    # Apply font size to the ticks (since we can't pass it to set_xticklabels anymore)
    ax.tick_params(axis='both', labelsize=font_size)

    # --- FIX ENDS HERE ---
    
    plt.legend(loc='upper left', fontsize=font_size)
    plt.tight_layout()
    
    return fig, ax


def plot_xy_symmetric(data: pd.DataFrame, figsize=(3, 3), title=None, fontsize=6, 
                      markers=None, markersize=150, colors=None, 
                      center=1, min_distance=0.001):    
    """
    Creates a symmetric scatter plot centered around a specified value with crosshairs.
    
    This visualization is designed for relative performance metrics (e.g., Up/Down 
    Capture Ratios, Alpha vs. Beta) where axes share a common scale and a theoretical 
    anchor point. It dynamically scales the axes symmetrically based on the maximum 
    deviation from the center.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot. The index is used to label the legend.
        The first column maps to the X-axis, and the second column maps to the Y-axis.
    figsize : tuple, default (3, 3)
        The dimensions (width, height) of the figure in inches.
    title : str, optional
        The title of the plot.
    fontsize : int or float, default 6
        The base font size applied to the title, axis labels, tick marks, and legend.
    markers : str, list, or dict, optional
        The marker style for the scatter points. Pass a single string for uniform 
        shapes, or a dictionary mapping index names to distinct shapes. Default is 'o'.
    markersize : int or float, default 200
        The size of the scatter plot markers.
    colors : list or str, optional
        Color palette or list of colors to apply to the plotted points. If None, 
        Seaborn's default palette is used.
    center : int or float, default 1
        The anchor value where the X and Y crosshairs intersect. 
    min_distance : float, default 0.001
        The minimum enforced distance from the center to the axis limits. 

    Returns
    -------
    fig, ax : The generated matplotlib figure and axes objects.
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
        s=markersize, zorder=5, palette=colors, edgecolors='black'
    )
    
    # Crosshairs & Styling
    ax.axhline(center, color='black', linestyle='--', alpha=0.3)
    ax.axvline(center, color='black', linestyle='--', alpha=0.3)
    
    ax.set_title(title, fontsize=fontsize+3, fontweight='bold', pad=5)
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
    ax.legend(fontsize=fontsize, loc='lower right')

    return fig, ax


def plot_capture_ratios(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series, 
                        figsize=(3, 3), 
                        colors=None, 
                        title='Up vs. Down Market Capture',
                        fontsize=6,
                        markers=['o']):
    """
    High-level wrapper that calculates and plots Up/Down Capture ratios.
    
    This function acts as an orchestrator, bridging the mathematical calculation 
    (`capture_ratios`) and the visualization (`plot_xy_symmetric`). It allows 
    users to generate a complete capture ratio scatter plot directly from raw 
    periodic return series.

    Parameters
    ----------
    strategy_returns : pd.DataFrame
        Periodic returns of the strategies. Each column should represent a 
        distinct strategy or asset, and the index should be datetime.
    benchmark_returns : pd.Series
        Periodic returns of the benchmark to calculate the capture against. 
        Must be aligned or alignable with `strategy_returns`.
    figsize : tuple, default (3, 3)
        The dimensions (width, height) of the figure in inches.
    colors : list or str, optional
        Color palette or list of colors to apply to the plotted strategies. 
        If None, Seaborn's default palette is used.
    title : str, default 'Up vs. Down Market Capture'
        The title displayed at the top of the plot.
    fontsize : int or float, default 6
        The base font size applied to the title, axis labels, tick marks, 
        and legend.
    markers : str, list, or dict, optional
        The marker style for the scatter points. Pass a single string for uniform 
        shapes, or a dictionary mapping index names to distinct shapes. Default is 'o'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    ax : matplotlib.axes.Axes
        The axes containing the plotted data.
        
    Example
    -------
    >>> fig, ax = plot_capture_ratios(
    ...     strategy_returns=my_funds_df, 
    ...     benchmark_returns=sp500_series,
    ...     title="Manager Capture Analysis"
    ... )
    """

    # Compute Capture Dataframe
    captures_df = capture_ratios(strategy_returns, benchmark_returns)

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
                          font_size=6):
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
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Dynamic positioning for the text (placed near the left edge)
    ax.text(x=ax.get_xlim()[0] + 0.1, y=0.52, s="50% Threshold", 
            color='gray', fontsize=font_size, ha='left')

    plt.tight_layout()
    
    return fig, ax

 # ============== THE END ============== #     
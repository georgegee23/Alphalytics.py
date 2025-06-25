
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import quantstats as qs
from scipy.stats import norm, probplot


from .performance_metrics import compute_capm, compute_performance_table
from .ic_analysis import cross_sectional_spearmanr, compute_spearman_stats, factor_decay
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
    data.columns = ["Spearman Correlation", f"{window}-Window Rolling Mean"]

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
    ax2.plot(data[f"{window}-Window Rolling Mean"], color='black')
    ax2.set_ylabel('Rolling IC', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend([f"{window}-Window Rolling Mean"], loc='upper right')

    # Center the right y-axis around 0
    data2_max = data[f"{window}-Window Rolling Mean"].max()
    data2_min = data[f"{window}-Window Rolling Mean"].min()
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
                    factor_lag: int = 1, periods_label: str = "Days") -> tuple[plt.Figure, np.ndarray]:
    # Calculate Spearman rank correlations
    spearmanr_ts = cross_sectional_spearmanr(factors, returns, factor_lag=factor_lag)["SpearmanR"]
    
    # Compute statistics (placeholder function)
    corr_stats = compute_spearman_stats(factors, returns, factor_lag=factor_lag)
    
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
    decay['p_value'].plot(ax=ax[1], marker='o', linestyle='-', color='black')
    ax[1].set_title('P-Value of IC', fontsize=14)  # Larger title
    ax[1].set_xlabel(f'Horizon({periods_label})')
    ax[1].set_ylabel('P-Value')
    ax[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax[1].grid(True, linestyle='--', alpha=0.7)

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



 # ============== THE END ============== #     
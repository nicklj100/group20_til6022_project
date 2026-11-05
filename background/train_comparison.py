# Data imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""This script is used to compare the train data from one scenario with the train data of another scenario. 

In this case the purpose is to compare the baseline with a selected disruption period.

Lots of the code is recycled and/or changed from train_analysis.py"""

# hours of operation defined
HOURS = list(range(5,23))

# Indicators in the comparison table
# key, label, higher_better?, relative_change percentage?
KPI_INFO = [
    ('days', 'Days analysed (n)', None, None),
    ('stops _per_day', 'Stops per day', None, True),
    ('delay_minutes per vehicle', 'Delay minutes per vehicle', False, True),
    ('cancelled_pct', 'Cancelled trains (%)', False, True),
    ('arr_punctuality_pct', 'Arrival punctuality (%)', True, True),
    ('dep_punctuality_pct', 'Departure punctuality (%)', True, True),
    ('absorption_pct', 'Delay absorption (%)', True, True),
    ('net_delay_change', 'Net delay change (minutes)', False, True),
    ('reduced_delay_pct', 'Reduced delay (%)', True, True)
]

# KPI creation in short
def calculate_kpis(train):
    """Caclulate the KPIs for the given train data and return a dictionary with the results"""
    n = len(train)
    days = train['Date'].dt.normalize().nunique()

    # Delay-minutes per vehicle (from train_analysis.py)
    total_delay = train['arr_delay'].clip(lower=0).fillna(0).sum() + train['dep_delay'].clip(lower=0).fillna(0).sum()

    # Non-cancelled trains only, as in punctuality and delay absorption
    running = train.loc[~train['cancelled'].fillna(False)]
    arr_punctual = running['arr_delay'] < 3
    dep_punctual = running['dep_delay'] < 3

    # Absorption needs both delay values (in and output for late arrivals
    absorption = running.dropna(subset = ['arr_delay', 'dep_delay'])
    delay_change = absorption['arr_delay'] - absorption['dep_delay']

    # Select late arriving trains only for absorption
    late = absorption.loc[absorption['arr_delay'] > 0]
    late_arr_sum = late['arr_delay'].sum()
    late_dep_sum = late['dep_delay'].sum()

    # Trains whose delay changed between arrival and departure
    changed = absorption.loc[absorption['arr_delay'] != absorption['dep_delay']]

    # Return KPI's
    return{
        'days': days,
        'stops _per_day': n / days,
        'delay_minutes per vehicle': total_delay / n,
        'cancelled_pct': train['cancelled'].mean() * 100,
        'arr_punctuality_pct': arr_punctual.mean() * 100,
        'dep_punctuality_pct': dep_punctual.mean() * 100,
        'absorption_pct': (late_arr_sum - late_dep_sum) / late_arr_sum * 100 if late_arr_sum > 0 else np.nan,
        'net_delay_change': delay_change.mean(),
        'reduced_delay_pct': (changed['arr_delay'] - changed['dep_delay']).mean() * 100
    }

# Comparison table
def compare_kpis(baseline_kpis, scenario_kpis, baseline_name, scenario_name):
    """Compare the baseline (baseline_kpis) with the selected train period (scenario_kpis) and return results"""
    rows = []

    for key, label, higher_better, relative_change in KPI_INFO:
        base = baseline_kpis[key]
        scn = scenario_kpis[key]

        if key == 'days':
            difference = ''
            change = ''
        else:
            difference = scn - base
            # Percentage change relative to baseline (baseline above zero enforces)
            change = (difference / base * 100 if relative_change and pd.notna(base) and base != 0 else np.nan)

        # Effect depending on the direction of the indicator
        if higher_better is None or pd.isna(difference):
            effect = ''
        elif abs(difference) < 0.05: # Arbitrary threshold for "no effect"
            effect = 'No effect'
        elif (difference > 0) == higher_better:
            effect = 'Positive'
        else:
            effect = 'Negative'

        rows.append({
            'Indicator': label, baseline_name: base, scenario_name: scn,
            'Difference': difference, 'Change (%)': change, 'Effect': effect
        })

    return pd.DataFrame(rows).set_index('Indicator')


# --- Plots --- #
def plot_hourly_comparison(scenarios):
    fig, axes = plt.subplots(1,3, figsize=(18,5))

    axes[2].sharey(axes[1])  # Both punctuality panels share the y-axis, so arrival and departure can be compared directly

    for train, name, color in scenarios:
        # Cancellation route per hour for all, punctuality only for non-cancelled
        cancelled = train.groupby('arr_hour')['cancelled'].mean().reindex(HOURS) * 100
        axes[0].plot(
            HOURS, cancelled.values,
            marker = 'o', color = color, label = name
        )

        running = train.loc[~train['cancelled'].fillna(False)] # recycle from above
        for ax, (hour_col, delay_col) in zip(axes[1:], [('arr_hour', 'arr_delay'), ('dep_hour', 'dep_delay')]):
            # Punctual = delay below 3 minutes, shown as a percentage per hour
            by_hour = (running[delay_col] < 3).groupby(running[hour_col]).mean().reindex(HOURS) * 100
            ax.plot(
                HOURS, by_hour.values,
                marker = 'o', color = color, label = name
            )

    for ax, title, xlabel, ylabel in zip(axes, [
        'Cancellation rate', 'Arrival punctuality (< 3 min)', 'Departure punctuality (< 3 min)'], 
        ['Arrival hour', 'Arrival hour', 'Departure hour'],
        ['Cancelled stops (%)', 'Punctual trains (%)', 'Punctual trains (%)']):
        ax.set_xticks(HOURS)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left')
        ax.grid(axis = 'y', alpha = 0.25)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend()
    fig.suptitle('Figure 3.4.1: Cancellation and Punctuality per Hour Comparison')
    fig.tight_layout()
    plt.show()

def plot_weekday_comparison(scenarios):
    """Compare cancellation rates and punctuality per weekday."""

    weekdays = range(7)
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[2].sharey(axes[1])

    bar_width = 0.35
    x = np.arange(7)

    for i, (train, name, color) in enumerate(scenarios):

        train = train.copy()
        train['weekday'] = pd.to_datetime(train['Date']).dt.dayofweek

        offset = (i - (len(scenarios) - 1) / 2) * bar_width

        # Cancellation rate per weekday
        cancelled = (
            train.groupby('weekday')['cancelled']
            .mean()
            .reindex(weekdays) * 100
        )

        bars = axes[0].bar(
            x + offset, cancelled.values,
            width=bar_width, color=color, label=name
        )

        axes[0].bar_label(
            bars, fmt='%.0f%%', padding=3, fontsize=8
        )

        # Punctuality for non-cancelled trains
        running = train.loc[~train['cancelled'].fillna(False)]

        for ax, delay_col in zip(
            axes[1:], ['arr_delay', 'dep_delay']
        ):

            # Punctual = delay below 3 minutes
            punctual = (
                (running[delay_col] < 3)
                .groupby(running['weekday'])
                .mean()
                .reindex(weekdays) * 100
            )

            bars = ax.bar(
                x + offset, punctual.values,
                width=bar_width, color=color, label=name
            )

            ax.bar_label(
                bars, fmt='%.0f%%', padding=3, fontsize=8
            )

    # Format plots
    for ax, title, ylabel in zip(axes, [
        'Cancellation rate',
        'Arrival punctuality (< 3 min)',
        'Departure punctuality (< 3 min)'
    ], [
        'Cancelled trains (%)',
        'Punctual trains (%)',
        'Punctual trains (%)'
    ]):

        ax.set_xticks(x)
        ax.set_xticklabels(weekday_names)
        ax.set_xlabel('Weekday')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left')
        ax.grid(axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)

    # Consistent punctuality scale
    axes[1].set_ylim(0, 110)

    axes[0].legend()

    fig.suptitle(
        'Figure 3.4.2: Cancellation and Punctuality per Weekday Comparison'
    )

    fig.tight_layout()
    plt.show()

# -------------- #
DWELL_BINS = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
DWELL_LABELS = [
    '0-1', '2', '3', '4', '5',
    '6-7', '8-9', '10-14', '15+'
]

def delay_exceedance(train, max_delay=5):
    # Share of running trains with an arrival delay larger than each threshold (in minutes)
    running = train.loc[~train['cancelled'].fillna(False)]
    delays = running['arr_delay'].dropna()
    thresholds = np.arange(0, max_delay + 1)
    shares = np.array([(delays > t).mean() * 100 for t in thresholds])
    return thresholds, shares

def absorption_by_dwell_time(train, min_trains=15, min_arr_delay=60):
    # Delay absorption of late-arriving trains per scheduled dwell time, same formula and masking as the heatmap in analyse_correlations
    # Non-cancelled trains with both delay values known, as in analyse_delay_absorption
    comp = train.loc[~train['cancelled'].fillna(False)].dropna(
        subset=['arr_delay', 'dep_delay']
    ).copy()

    # Scheduled dwell time in minutes, same calculation as in analyse_delay_absorption
    sched_arr = pd.to_datetime(
        comp['scheduled_arrival'], format='%H:%M', errors='coerce'
    )
    sched_dep = pd.to_datetime(
        comp['scheduled_departure'], format='%H:%M', errors='coerce'
    )

    comp['dwell_time'] = (
        (sched_dep - sched_arr).dt.total_seconds() / 60
    ) % 1440
    comp['dwell_bin'] = pd.cut(
        comp['dwell_time'],
        bins=DWELL_BINS,
        labels=DWELL_LABELS,
        right=False
    )

    late = comp.loc[comp['arr_delay'] > 0]
    grouped = late.groupby('dwell_bin', observed=False).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    # Dwell times with too little data are left empty, as their percentages are unreliable
    absorption = (
        (grouped['arr_sum'] - grouped['dep_sum']) / grouped['arr_sum'] * 100
    ).where(
        (grouped['arr_sum'] >= min_arr_delay)
        & (grouped['n_trains'] >= min_trains)
    )

    return absorption.reindex(DWELL_LABELS).astype(float)

def plot_delay_comparison(situations, min_trains=15, min_arr_delay=10, max_delay=5):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: how large the arrival delays are (the curve at 0 min is the share of delayed arrivals)
    for train, name, colour in situations:
        thresholds, shares = delay_exceedance(train, max_delay)
        axes[0].plot(thresholds, shares, linewidth=2, color=colour, label=name)

    axes[0].set_xlabel('Arrival delay threshold (min)')
    axes[0].set_ylabel(f'Trains delayed by more than threshold ({max_delay} min)')
    axes[0].set_title(f'Trains with arrival delay > threshold ({max_delay} min)', loc='left', fontweight='bold')
    axes[0].legend(frameon=False)

    # Right plot: delay absorption of late-arriving trains per scheduled dwell time, baseline and selected period side by side
    width = 0.4
    x = np.arange(len(DWELL_LABELS))
    for i, (train, name, colour) in enumerate(situations):
        values = absorption_by_dwell_time(train, min_trains, min_arr_delay)
        axes[1].bar(
            x + (i - 0.5) * width, values.values, width,
            color=colour, label=name
        )

    axes[1].axhline(0, color='black', linewidth=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DWELL_LABELS)
    axes[1].set_xlabel('Scheduled dwell time (minutes)')
    axes[1].set_ylabel('Delay absorption (%)')
    axes[1].set_title(
        'Delay absorption of late-arriving trains', loc='left', fontweight='bold'
    )

    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        'Figure 3.4.2: Delay Size and Delay Absorption, Baseline vs Selected Period',
    )
    fig.text(
        0.5, -0.02,
        f'Dwell times with fewer than {min_trains} late-arriving trains '
        f'or less than {min_arr_delay} arrival delay-minutes are not shown.',
        ha='center', fontsize=9
    )
    fig.tight_layout()
    plt.legend()
    plt.show()
# --------- #


# --- Comparison --- #
def compare_operations(train, train_selected, baseline_name='Baseline',
                       scenario_name='Selected period', plots=True):
    """Compare the baseline with the selected train period and return a formatted table."""

    if baseline_name == scenario_name:
        raise ValueError("Baseline and scenario names must be different.")

    table = compare_kpis(
        calculate_kpis(train),
        calculate_kpis(train_selected),
        baseline_name, scenario_name
    )

    formatted_table = table.style.format(
        lambda x: f'{x:.1f}' if isinstance(x, (int, float, np.number)) else x,
        na_rep=''
    )

    # print('Difference of the indicators is in percentage points, Change is relative change compared to baseline.')

    if plots:
        scenarios = [
            (train, baseline_name, 'C0'),
            (train_selected, scenario_name, 'C3')
        ]
        plot_hourly_comparison(scenarios)
        plot_weekday_comparison(scenarios)
        plot_delay_comparison(scenarios)

    return table, formatted_table
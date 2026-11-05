# Data imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""This script is used to compare the train data from one scenario with the train data of another scenario.

In this case the purpose is to compare the baseline with a selected disruption period.

Lots of the code is recycled and/or changed from train_analysis.py"""

# Hours of operation defined (05:00-22:00)
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
    """Calculate the KPIs for the given train data and return a dictionary with the results."""

    # Total number of train stops and unique operating days
    n = len(train)
    days = train['Date'].dt.normalize().nunique()

    # Delay-minutes per vehicle (from train_analysis.py)
    # Negative delays are set to zero because early trains do not contribute to delay minutes
    # Missing delay values are also treated as zero
    total_delay = train['arr_delay'].clip(lower=0).fillna(0).sum() + train['dep_delay'].clip(lower=0).fillna(0).sum()

    # Non-cancelled trains only, as in punctuality and delay absorption
    # A train is considered punctual when its delay is below 3 minutes
    running = train.loc[~train['cancelled'].fillna(False)]
    arr_punctual = running['arr_delay'] < 3
    dep_punctual = running['dep_delay'] < 3

    # Absorption needs both arrival and departure delay values
    # Calculate the difference between arrival and departure delay for all running trains
    absorption = running.dropna(subset = ['arr_delay', 'dep_delay'])
    delay_change = absorption['arr_delay'] - absorption['dep_delay']

    # Select late arriving trains only for absorption
    # Only trains arriving with a positive delay are included
    late = absorption.loc[absorption['arr_delay'] > 0]
    late_arr_sum = late['arr_delay'].sum()
    late_dep_sum = late['dep_delay'].sum()

    # Trains whose delay changed between arrival and departure
    # This selects trains where the arrival and departure delay are not equal
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
    """Compare the baseline (baseline_kpis) with the selected train period (scenario_kpis) and return results."""

    rows = []

    # Loop through all indicators defined in KPI_INFO
    for key, label, higher_better, relative_change in KPI_INFO:
        base = baseline_kpis[key]
        scn = scenario_kpis[key]

        # Calculate the absolute and relative difference between the baseline and selected period
        if key == 'days':
            difference = ''
            change = ''
        else:
            # Positive difference means the indicator increased during the selected period
            difference = scn - base

            # Percentage change relative to baseline (baseline above zero enforces)
            change = (difference / base * 100 if relative_change and pd.notna(base) and base != 0 else np.nan)

        # Effect depending on the direction of the indicator
        # The higher_better variable determines whether an increase is positive or negative
        if higher_better is None or pd.isna(difference):
            effect = ''
        elif abs(difference) < 0.05: # Arbitrary threshold for "no effect"
            effect = 'No effect'
        elif (difference > 0) == higher_better:
            effect = 'Positive'
        else:
            effect = 'Negative'

        # Add the results for this indicator to the comparison table
        rows.append({
            'Indicator': label, baseline_name: base, scenario_name: scn,
            'Difference': difference, 'Change (%)': change, 'Effect': effect
        })

    # Convert the list of results into a DataFrame and use the indicator names as the index
    return pd.DataFrame(rows).set_index('Indicator')


# --- Plots --- #
def plot_hourly_comparison(scenarios):
    # Create three plots next to each other
    fig, axes = plt.subplots(1,3, figsize=(18,5))

    # Both punctuality panels share the y-axis, so arrival and departure can be compared directly
    axes[2].sharey(axes[1])

    # Loop through the different scenarios (baseline and selected period)
    for train, name, color in scenarios:

        # Cancellation rate per arrival hour, including all trains
        # Punctuality is calculated separately using non-cancelled trains only
        cancelled = train.groupby('arr_hour')['cancelled'].mean().reindex(HOURS) * 100
        axes[0].plot(
            HOURS, cancelled.values,
            marker = 'o', color = color, label = name
        )

        # Select non-cancelled trains for the punctuality calculations
        running = train.loc[~train['cancelled'].fillna(False)]

        # Calculate arrival and departure punctuality per hour
        for ax, (hour_col, delay_col) in zip(axes[1:], [('arr_hour', 'arr_delay'), ('dep_hour', 'dep_delay')]):

            # Punctual = delay below 3 minutes, shown as a percentage per hour
            # The mean of the True/False values gives the share of punctual trains
            by_hour = (running[delay_col] < 3).groupby(running[hour_col]).mean().reindex(HOURS) * 100
            ax.plot(
                HOURS, by_hour.values,
                marker = 'o', color = color, label = name
            )

    # Add titles, axis labels and formatting to all three plots
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

    # Add a legend and overall figure title
    axes[0].legend()
    fig.suptitle('Figure 3.4.1: Cancellation and Punctuality per Hour Comparison')
    fig.tight_layout()
    plt.show()


def plot_weekday_comparison(scenarios):
    """Compare cancellation rates and punctuality per weekday."""

    # Define weekdays from Monday (0) to Sunday (6)
    weekdays = range(7)
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create three plots next to each other
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Both punctuality panels share the y-axis for direct comparison
    axes[2].sharey(axes[1])

    # Define the width of each bar and the x-axis positions
    bar_width = 0.35
    x = np.arange(7)

    # Loop through the different scenarios
    for i, (train, name, color) in enumerate(scenarios):

        # Create a copy and determine the weekday for each train operation
        train = train.copy()
        train['weekday'] = pd.to_datetime(train['Date']).dt.dayofweek

        # Offset the bars so that the scenarios appear next to each other
        offset = (i - (len(scenarios) - 1) / 2) * bar_width

        # Cancellation rate per weekday, including all trains
        cancelled = (
            train.groupby('weekday')['cancelled']
            .mean()
            .reindex(weekdays) * 100
        )

        # Plot cancellation rates as grouped bars
        bars = axes[0].bar(
            x + offset, cancelled.values,
            width=bar_width, color=color, label=name
        )

        # Add percentage labels above each bar
        axes[0].bar_label(
            bars, fmt='%.0f%%', padding=3, fontsize=8
        )

        # Punctuality for non-cancelled trains
        running = train.loc[~train['cancelled'].fillna(False)]

        # Calculate arrival and departure punctuality per weekday
        for ax, delay_col in zip(
            axes[1:], ['arr_delay', 'dep_delay']
        ):

            # Punctual = delay below 3 minutes
            # Calculate the percentage of punctual trains for each weekday
            punctual = (
                (running[delay_col] < 3)
                .groupby(running['weekday'])
                .mean()
                .reindex(weekdays) * 100
            )

            # Plot punctuality as grouped bars
            bars = ax.bar(
                x + offset, punctual.values,
                width=bar_width, color=color, label=name
            )

            # Add percentage labels above each bar
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

    # Add legend and overall figure title
    axes[0].legend()

    fig.suptitle(
        'Figure 3.4.2: Cancellation and Punctuality per Weekday Comparison'
    )

    fig.tight_layout()
    plt.show()


# -------------- #

# Define the scheduled dwell time categories in minutes
DWELL_BINS = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
DWELL_LABELS = [
    '0-1', '2', '3', '4', '5',
    '6-7', '8-9', '10-14', '15+'
]


def delay_exceedance(train, max_delay=5):
    # Share of running trains with an arrival delay larger than each threshold (in minutes)

    # Exclude cancelled trains and missing arrival delay values
    running = train.loc[~train['cancelled'].fillna(False)]
    delays = running['arr_delay'].dropna()

    # Create thresholds from 0 up to max_delay (inclusive)
    thresholds = np.arange(0, max_delay + 1)

    # Calculate the percentage of trains with an arrival delay greater than each threshold
    shares = np.array([(delays > t).mean() * 100 for t in thresholds])

    return thresholds, shares


def absorption_by_dwell_time(train, min_trains=15, min_arr_delay=60):
    # Delay absorption of late-arriving trains per scheduled dwell time, same formula and masking as the heatmap in analyse_correlations

    # Non-cancelled trains with both delay values known, as in analyse_delay_absorption
    comp = train.loc[~train['cancelled'].fillna(False)].dropna(
        subset=['arr_delay', 'dep_delay']
    ).copy()

    # Convert scheduled arrival and departure times into datetime values
    # This allows the difference between the two times to be calculated
    sched_arr = pd.to_datetime(
        comp['scheduled_arrival'], format='%H:%M', errors='coerce'
    )
    sched_dep = pd.to_datetime(
        comp['scheduled_departure'], format='%H:%M', errors='coerce'
    )

    # Calculate scheduled dwell time in minutes
    # Modulo 1440 corrects for trains arriving before midnight and departing after midnight (for reproducability in other scenarios)
    comp['dwell_time'] = (
        (sched_dep - sched_arr).dt.total_seconds() / 60
    ) % 1440

    # Assign each train to a scheduled dwell time category
    # right=False means the lower boundary is included and the upper boundary is excluded
    comp['dwell_bin'] = pd.cut(
        comp['dwell_time'],
        bins=DWELL_BINS,
        labels=DWELL_LABELS,
        right=False
    )

    # Select late arriving trains only for the delay absorption calculation
    late = comp.loc[comp['arr_delay'] > 0]

    # Group trains by their scheduled dwell time category
    # Sum arrival and departure delays and count the number of trains in each category
    grouped = late.groupby('dwell_bin', observed=False).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    # Calculate delay absorption as the percentage of arrival delay minutes recovered by departure
    # Dwell times with too little data are left empty, as their percentages are unreliable
    absorption = (
        (grouped['arr_sum'] - grouped['dep_sum']) / grouped['arr_sum'] * 100
    ).where(
        (grouped['arr_sum'] >= min_arr_delay)
        & (grouped['n_trains'] >= min_trains)
    )

    # Return the absorption values in the predefined dwell time order
    return absorption.reindex(DWELL_LABELS).astype(float)


def plot_delay_comparison(situations, min_trains=15, min_arr_delay=10, max_delay=5):
    # Create two plots next to each other
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: show the percentage of trains with an arrival delay exceeding each threshold
    # The point at 0 minutes represents all trains with a positive arrival delay
    for train, name, colour in situations:
        thresholds, shares = delay_exceedance(train, max_delay)
        axes[0].plot(thresholds, shares, linewidth=2, color=colour, label=name)

    # Format the arrival delay exceedance plot
    axes[0].set_xlabel('Arrival delay threshold (min)')
    axes[0].set_ylabel(f'Trains delayed (%)')
    axes[0].set_title(f'Trains with arrival delay > threshold', loc='left')
    axes[0].legend(frameon=False)

    # Right plot: delay absorption of late-arriving trains per scheduled dwell time
    # Baseline and selected period are shown side by side
    width = 0.4
    x = np.arange(len(DWELL_LABELS))

    # Loop through the different scenarios and calculate absorption per dwell time category
    for i, (train, name, colour) in enumerate(situations):
        values = absorption_by_dwell_time(train, min_trains, min_arr_delay)

        # Plot the absorption values as grouped bars
        axes[1].bar(
            x + (i - 0.5) * width, values.values, width,
            color=colour, label=name
        )

    # Add a horizontal reference line at zero absorption
    axes[1].axhline(0, color='black', linewidth=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DWELL_LABELS)
    axes[1].set_xlabel('Scheduled dwell time (minutes)')
    axes[1].set_ylabel('Delay absorption (%)')
    axes[1].set_title('Delay absorption of late-arriving trains', loc='left')

    # Apply consistent formatting to both plots
    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    # Add the overall figure title and explain which dwell time categories are not displayed
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

    # Check that the baseline and scenario have different names
    if baseline_name == scenario_name:
        raise ValueError("Baseline and scenario names must be different.")

    # Calculate the KPIs for both datasets and create the comparison table
    table = compare_kpis(
        calculate_kpis(train),
        calculate_kpis(train_selected),
        baseline_name, scenario_name
    )

    # Format all numerical values to one decimal place for display
    # The original table remains unchanged
    formatted_table = table.style.format(
        lambda x: f'{x:.1f}' if isinstance(x, (int, float, np.number)) else x,
        na_rep=''
    )

    # Optional: generate comparison plots
    if plots:
        # Define the datasets, scenario names and colours used in the plots
        scenarios = [
            (train, baseline_name, 'C0'),
            (train_selected, scenario_name, 'C3')
        ]

        # Generate hourly, weekday and delay comparisons
        plot_hourly_comparison(scenarios)
        plot_weekday_comparison(scenarios)
        plot_delay_comparison(scenarios)

    # Return both the original numerical table and its formatted display version
    return table, formatted_table
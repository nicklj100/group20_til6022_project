import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import BoundaryNorm


# ---------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------
def load_train_data(data='data/traindata_total.csv',
                    start_period='2025-12-15', end_period='2026-04-30'):
    """Load, clean and prepare the train dataset; does not create plots."""
    train = pd.read_csv(data)
    train['Service:Date'] = pd.to_datetime(train['Service:Date'])
    train = train[train['Service:Date'].between(start_period, end_period)].copy()

    train = train.drop(columns=['index', 'Service:RDT-ID'])
    train = train.rename(columns={
        'Service:Type': 'Type', 'Service:Date': 'Date',
        'Stop:Arrival time': 'scheduled_arrival',
        'Stop:Departure time': 'scheduled_departure',
        'Stop:Arrival cancelled': 'arr_cancelled',
        'Stop:Departure cancelled': 'dep_cancelled',
        'Stop:Arrival delay': 'arr_delay',
        'Stop:Departure delay': 'dep_delay'
    })
    train.columns = (train.columns.str.replace('Service:', '', regex=False)
                     .str.replace('Stop:', '', regex=False)
                     .str.replace('RDT-ID', 'train_id', regex=False))

    drop_cols = [
        'Completely cancelled', 'Partly cancelled', 'Maximum delay',
        'Train number', 'Station name', 'Station code', 'Platform change',
        'Planned platform', 'Actual platform', 'Company'
    ]
    train = train.drop(columns=drop_cols)
    train = train[train['Type'].isin(['Intercity', 'Sprinter'])].copy()

    for col in ['scheduled_arrival', 'scheduled_departure']:
        train[col] = pd.to_datetime(train[col], utc=True).dt.strftime('%H:%M')

    for label, delay_col in [('arrival', 'arr_delay'), ('departure', 'dep_delay')]:
        sched_col = f'scheduled_{label}'
        actual_col = f'actual_{label}'
        scheduled = pd.to_datetime(train[sched_col], format='%H:%M')
        actual = scheduled + pd.to_timedelta(train[delay_col], unit='m')
        train[sched_col] = scheduled.dt.strftime('%H:%M')
        train[actual_col] = actual.dt.strftime('%H:%M')

    train['arr_hour'] = pd.to_datetime(train['actual_arrival'], format='%H:%M', errors='coerce').dt.hour
    train['dep_hour'] = pd.to_datetime(train['actual_departure'], format='%H:%M', errors='coerce').dt.hour
    train = train[train['dep_hour'].between(5, 22) & train['arr_hour'].between(5, 22)].copy()
    train = train.sort_values(['Date', 'actual_arrival']).reset_index(drop=True)

    train['arrival_delayed'] = train['arr_delay'] > 0
    train['departure_delayed'] = train['dep_delay'] > 0
    train['arrival_delayed_3'] = train['arr_delay'] >= 3
    train['departure_delayed_3'] = train['dep_delay'] >= 3
    train['cancelled'] = train['arr_cancelled'].fillna(False).astype(bool) | train['dep_cancelled'].fillna(False).astype(bool)
    train['week'] = train['Date'].dt.to_period('W').astype(str)
    return train


# ---------------------------------------------------------------------
# Filter days with unusually high cancellation counts
# ---------------------------------------------------------------------
def filter_cancellation_days(train, cancellation_percentile=75, verbose=True):
    """Exclude days whose cancellation count is above the chosen percentile.

    Set cancellation_percentile=None to retain all days.
    The percentile threshold is calculated from daily cancellation counts
    in the loaded dataset.
    """
    if cancellation_percentile is None:
        return train.copy()

    daily_cancellations = train.groupby(
        train['Date'].dt.normalize()
    )['cancelled'].sum()

    if daily_cancellations.empty:
        return train.copy()

    threshold = np.percentile(daily_cancellations, cancellation_percentile)
    excluded_days = daily_cancellations[
        daily_cancellations > threshold
    ].index

    filtered_train = train[
        ~train['Date'].dt.normalize().isin(excluded_days)
    ].copy()

    # if verbose:
    #     print('\nCANCELLATION-DAY FILTER')
    #     print(f'Percentile used: {cancellation_percentile}th')
    #     print(f'Cancellation threshold: {threshold:.1f} per day')
    #     print(f'Days excluded: {len(excluded_days)} of {len(daily_cancellations)}')
    #     print(f'Trains removed: {len(train) - len(filtered_train)}')
    #     print(f'Trains remaining: {len(filtered_train)}')
    #     print("")

    return filtered_train

# ---------------------------------------------------------------------
# 4.1 - Delays and cancellations in numbers
# ---------------------------------------------------------------------
def analyse_delays(train, start_period=None, end_period=None, verbose=True):
    n = len(train)
    days = train['Date'].dt.normalize().nunique()
    delayed_vehicles = ((train['arrival_delayed']) | (train['departure_delayed'])).mean() * 100
    delayed_operations = ((train['arrival_delayed_3']) | (train['departure_delayed_3'])).mean() * 100
    cancellations = int(train['cancelled'].sum())
    cancellations_pct = cancellations / n * 100 if n else np.nan
    total_delay = train['arr_delay'].clip(lower=0).fillna(0).sum() + train['dep_delay'].clip(lower=0).fillna(0).sum()
    delay_per_vehicle = total_delay / n if n else np.nan
    delay_per_day = total_delay / days if days else np.nan
    cancel_per_100 = cancellations / n * 100 if n else np.nan
    cancel_per_day = cancellations / days if days else np.nan

    if verbose:
        period = f' BETWEEN: {start_period} AND {end_period}' if start_period is not None and end_period is not None else ''
        print(f'DELAYS AND CANCELLATIONS{period}')
        print(f'Number of complete cancellations: {cancellations}')
        print(f'    {delayed_vehicles:.1f}% of trains were delayed.')
        print(f'    {delayed_operations:.1f}% of train operations were delayed by at least 3 minutes.')
        print(f'    {cancellations_pct:.1f}% of trains were cancelled.')
        print('\nRELATIVE PERFORMANCE')
        print(f'    Delay-minutes per vehicle: {delay_per_vehicle:.2f} minutes')
        print(f'    Total delay-minutes per day: {delay_per_day:.1f} minutes')
        print(f'    Cancellations per 100 vehicles: {cancel_per_100:.2f}')
        print(f'    Cancellations per day: {cancel_per_day:.2f}')

    daily = train.groupby(train['Date'].dt.normalize()).agg(
        arrival_delayed=('arrival_delayed', 'mean'),
        departure_delayed=('departure_delayed', 'mean'),
        arrival_delayed_3=('arrival_delayed_3', 'mean'),
        departure_delayed_3=('departure_delayed_3', 'mean')
    ) * 100

    # Smooth daily variation using a 7-day rolling average
    rolling = daily.rolling(window=7, min_periods=4).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    plots = [
        ('arrival_delayed', 'departure_delayed',
        'Percentage of trains delayed (> 0 minutes)'),
        ('arrival_delayed_3', 'departure_delayed_3',
        'Percentage of trains delayed (≥ 3 minutes)')
    ]

    for ax, (arrival, departure, title) in zip(axes, plots):
        # Daily observations
        ax.plot(
            daily.index, daily[arrival],
            color='C0', alpha=0.2, linewidth=0.8
        )
        ax.plot(
            daily.index, daily[departure],
            color='C1', alpha=0.2, linewidth=0.8
        )

        # Seven-day rolling averages
        ax.plot(
            rolling.index, rolling[arrival],
            color='C0', linewidth=2, label='Arrival'
        )
        ax.plot(
            rolling.index, rolling[departure],
            color='C1', linewidth=2, label='Departure'
        )

        # Overall average
        ax.axhline(
            daily[arrival].mean(),
            color='C0', linestyle='--', alpha=0.8,
            label=f'Arrival mean ({daily[arrival].mean():.1f}%)'
        )
        ax.axhline(
            daily[departure].mean(),
            color='C1', linestyle='--', alpha=0.8,
            label=f'Departure mean ({daily[departure].mean():.1f}%)'
        )

        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delayed trains (%)')
        ax.set_ylim(bottom=0, top=50)
        ax.grid(axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(ncol=2, frameon=False, loc='upper right')

    axes[-1].set_xlabel('Date')
    fig.suptitle(
        'Daily Train Delays Over Time',
        fontsize=15, fontweight='bold'
    )
    fig.tight_layout()
    plt.show()

    return {
        'Vehicles': n, 'Days': days, 'Delayed vehicles (%)': delayed_vehicles,
        'Delayed operations (%)': delayed_operations,
        'Delay per vehicle (min)': delay_per_vehicle, 'Delay per day (min)': delay_per_day,
        'Cancellations per 100 vehicles': cancel_per_100, 'Cancellations per day': cancel_per_day
    }

# ---------------------------------------------------------------------
# 4.2 - Delays per hour
# ---------------------------------------------------------------------
def analyse_delays_per_hour(train):
    hourly = train.groupby('arr_hour').agg(
        arrival_mean=('arr_delay', 'mean'),
        departure_mean=('dep_delay', 'mean'),
        arrival_median=('arr_delay', 'median'),
        departure_median=('dep_delay', 'median'),
        arrival_delayed=('arrival_delayed', 'mean'),
        departure_delayed=('departure_delayed', 'mean'),
        arrival_delayed_3=('arrival_delayed_3', 'mean'),
        departure_delayed_3=('departure_delayed_3', 'mean'),
        cancelled=('cancelled', 'mean'),
        n_trains=('train_id', 'count')
    )

    hours = hourly.index

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 12), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.2, 0.8]}
    )

    # 1. Average delay
    axes[0].plot(
        hours, hourly['arrival_mean'],
        marker='o', linewidth=2, label='Arrival'
    )
    axes[0].plot(
        hours, hourly['departure_mean'],
        marker='o', linewidth=2, label='Departure'
    )
    axes[0].set_ylabel('Mean delay (min)')
    axes[0].set_title('Average delay by hour', loc='left')
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis='y', alpha=0.25)

    # 2. Percentage delayed
    for col, label, linestyle in [
        ('arrival_delayed', 'Arrival > 0 min', '-'),
        ('departure_delayed', 'Departure > 0 min', '-'),
        ('arrival_delayed_3', 'Arrival ≥ 3 min', '--'),
        ('departure_delayed_3', 'Departure ≥ 3 min', '--')
    ]:
        axes[1].plot(
            hours, hourly[col] * 100,
            marker='o', linewidth=1.8,
            linestyle=linestyle, label=label
        )

    axes[1].set_ylabel('Trains delayed (%)')
    axes[1].set_title('Frequency of delays', loc='left')
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(axis='y', alpha=0.25)

    # 3. Cancellation rate
    axes[2].bar(
        hours, hourly['cancelled'] * 100,
        width=0.7, label='Cancellation rate'
    )
    axes[2].set_ylabel('Cancelled (%)')
    axes[2].set_title('Cancellation rate by hour', loc='left')
    axes[2].grid(axis='y', alpha=0.25)

    # Shared formatting
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(4.5, 22.5)
        ax.set_xticks(range(5, 23))

    axes[2].set_xlabel('Scheduled arrival hour')
    fig.suptitle(
        'Train Performance by Hour of Day',
        fontsize=15, fontweight='bold'
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 4.3 - Punctuality
# ---------------------------------------------------------------------
def analyse_punctuality(train, train_punctual=None, verbose=True):
    punctual = train.loc[~train['cancelled'].fillna(False)].copy()
    punctual['arr_punctual'] = punctual['arr_delay'] < 3
    punctual['dep_punctual'] = punctual['dep_delay'] < 3
    punctual['both_punctual'] = punctual['arr_punctual'] & punctual['dep_punctual']
    punctual['delay_status'] = np.select(
        [punctual['arr_delay'] <= 0, punctual['arr_delay'] <= 3],
        ['On time', 'Delayed'], default='Significantly Delayed')
    punctual['dep_delay_status'] = np.select(
        [punctual['dep_delay'] <= 0, punctual['dep_delay'] < 3],
        ['On time', 'Delayed'], default='Significantly Delayed')
    punctual['weekday'] = punctual['Date'].dt.day_name()
    arr_mean = punctual['arr_punctual'].mean() * 100
    dep_mean = punctual['dep_punctual'].mean() * 100
    total_mean = (punctual['arr_punctual'].sum() + punctual['dep_punctual'].sum()) / (punctual['arr_punctual'].count() + punctual['dep_punctual'].count()) * 100
    by_arr_hour = punctual.groupby('arr_hour')['arr_punctual'].mean() * 100
    by_dep_hour = punctual.groupby('dep_hour')['dep_punctual'].mean() * 100
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = punctual.groupby('weekday')[['arr_punctual', 'dep_punctual']].mean().mul(100).reindex(weekday_order)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Punctuality per hour
    axes[0].plot(by_arr_hour.index, by_arr_hour.values, marker='o', label='Arrival')
    axes[0].plot(by_dep_hour.index, by_dep_hour.values, marker='o', label='Departure')

    axes[0].axhline(arr_mean, linestyle='--', color='C0',
                    label=f'Mean arrival ({arr_mean:.1f}%)')
    axes[0].axhline(dep_mean, linestyle='--', color='C1',
                    label=f'Mean departure ({dep_mean:.1f}%)')
    axes[0].axhline(total_mean, linestyle='--', color='C2',
                    label=f'Mean overall ({total_mean:.1f}%)')

    axes[0].set_ylim(
        min(by_arr_hour.min(), by_dep_hour.min()) - 5,
        100
    )
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Punctual trains (%)')
    axes[0].set_title('Train punctuality per hour')
    axes[0].set_xticks(range(5, 23))
    axes[0].grid()
    axes[0].legend()

    # Punctuality per weekday
    x = np.arange(len(weekday_order))
    width = 0.35

    y_min = min(
        weekly['arr_punctual'].min(),
        weekly['dep_punctual'].min()
    ) - 5

    # Grey areas above each bar, up to 100%
    for i, col in enumerate(['arr_punctual', 'dep_punctual']):
        positions = x + (i - 0.5) * width
        values = weekly[col].values

        axes[1].bar(
            positions, 100 - values, width,
            bottom=values, color='lightgrey',
            edgecolor='none', zorder=1
        )

    for i, (col, label) in enumerate([
        ('arr_punctual', 'Arrival'),
        ('dep_punctual', 'Departure')
    ]):
        positions = x + (i - 0.5) * width
        values = weekly[col].values

        axes[1].bar(
            positions, values - 80, width,
            bottom=80, label=label, zorder=2
        )

    # Mean lines
    axes[1].axhline(
        arr_mean, linestyle='--', color='C0',
        label=f'Mean arrival ({arr_mean:.1f}%)', zorder=3
    )
    axes[1].axhline(
        dep_mean, linestyle='--', color='C1',
        label=f'Mean departure ({dep_mean:.1f}%)', zorder=3
    )
    axes[1].axhline(
        total_mean, linestyle='--', color='C2',
        label=f'Mean overall ({total_mean:.1f}%)', zorder=3
    )

    axes[1].set_ylim(y_min, 100)
    axes[1].set_xlabel('Day of the week')
    axes[1].set_ylabel('Punctual trains (%)')
    axes[1].set_title('Train punctuality per weekday')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(weekday_order, rotation=45, ha='right')
    axes[1].grid(axis='y', zorder=0)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    both = punctual['both_punctual'].mean() * 100
    if verbose:
        print(f'Trains punctual at both arrival and departure: {both:.2f}%')
        print(f'Trains punctual at arrival: {arr_mean:.2f}%')
        print(f'Trains punctual at departure: {dep_mean:.2f}%')
    # return punctual, {'Both arrival and departure punctual (%)': both, 'Arrival punctual (%)': arr_mean, 'Departure punctual (%)': dep_mean}


# ---------------------------------------------------------------------
# 4.4 - Delay absorption and propagation
# ---------------------------------------------------------------------
def analyse_delay_absorption(train, delay_comparison=None,
                             train_punctual=None, verbose=True):

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------
    if train_punctual is None:
        punctual = train.loc[~train['cancelled'].fillna(False)].copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )
        punctual['arr_punctual'] = punctual['arr_delay'] < 3
        punctual['dep_punctual'] = punctual['dep_delay'] < 3
        punctual['both_punctual'] = (
            punctual['arr_punctual'] & punctual['dep_punctual']
        )
        punctual['weekday'] = punctual['Date'].dt.day_name()
    else:
        punctual = train_punctual.copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )
        punctual['weekday'] = punctual['Date'].dt.day_name()

    # All non-cancelled trains with both delay values available
    all_trains = punctual.dropna(
        subset=['arr_delay', 'dep_delay']
    ).copy()

    # Positive = delay absorbed; negative = delay increased
    all_trains['delay_change'] = (
        all_trains['dep_delay'] - all_trains['arr_delay']
    )

    # Selected comparison dataset for the remaining analyses
    if delay_comparison is None:
        comp = all_trains.loc[
            ~(
                (all_trains['arr_delay'] == 0)
                & (all_trains['dep_delay'] == 0)
            )
        ].copy()
    else:
        # Preserve the existing comparison dataset and its filtering
        comp = delay_comparison.copy()
        comp['Date'] = pd.to_datetime(
            comp['Date'], errors='coerce'
        )
        comp = comp.dropna(
            subset=['arr_delay', 'dep_delay']
        ).copy()

    comp['weekday'] = comp['Date'].dt.day_name()
    comp['dep_hour'] = pd.to_numeric(
        comp['dep_hour'], errors='coerce'
    )

    comp['delay_change'] = (
        comp['dep_delay'] - comp['arr_delay']
    )

    # Use the same scheduled dwell-time calculation as the original
    # dwell-time heatmap
    sched_arr = pd.to_datetime(
        comp['scheduled_arrival'], format='%H:%M', errors='coerce'
    )
    sched_dep = pd.to_datetime(
        comp['scheduled_departure'], format='%H:%M', errors='coerce'
    )

    comp['dwell_time'] = (
        (sched_dep - sched_arr).dt.total_seconds() / 60
    ) % 1440

    dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
    dwell_labels = [
        '0-1', '2', '3', '4', '5',
        '6-7', '8-9', '10-14', '15+'
    ]

    comp['dwell_bin'] = pd.cut(
        comp['dwell_time'],
        bins=dwell_bins,
        labels=dwell_labels,
        right=False
    )

    # Classify delay propagation using 3- and 1-minute thresholds
    for threshold in (3, 1):
        comp[f'delay_propagation_{threshold}min'] = np.select(
            [
                comp['delay_change'] > threshold,
                comp['delay_change'] < -threshold
            ],
            ['Delay reduced', 'Delay increased'],
            default='Unchanged delay'
        )

    # ---------------------------------------------------------
    # 2. Summary metrics
    # ---------------------------------------------------------

    # Mean delay change across ALL eligible non-cancelled trains
    mean_all = all_trains['delay_change'].mean()

    # Mean delay change in the selected comparison dataset
    mean_selected = comp['delay_change'].mean()

    arr_sum = comp['arr_delay'].sum()
    dep_sum = comp['dep_delay'].sum()

    # Net percentage of arrival delay recovered in selected dataset
    overall_absorption = (
        (arr_sum - dep_sum) / arr_sum * 100
        if arr_sum > 0 else np.nan
    )

    # Mean net delay recovery per selected train
    net_per_train = (
        comp['delay_change'].mean()
        if len(comp) > 0 else np.nan
    )

    # Delay absorption among trains arriving late
    late = comp.loc[comp['arr_delay'] > 0].copy()

    late_arr_sum = late['arr_delay'].sum()
    late_dep_sum = late['dep_delay'].sum()

    late_absorption = (
        (late_arr_sum - late_dep_sum) / late_arr_sum * 100
        if late_arr_sum > 0 else np.nan
    )

    if verbose:
        print(
            f'Mean net delay change (delayed trains): '
            f'{mean_all:.2f} min/train'
        )
        print(f'Overall delay absorption: {overall_absorption:.2f}%')
        print(f'Late-arrival delay absorption: {late_absorption:.2f}%')


    
    # ---------------------------------------------------------
    # 3. Plot net delay change and absorption
    # ---------------------------------------------------------
    order = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday'
    ]
    hours = list(range(5, 23))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Left: all non-cancelled trains
    # Right: trains arriving late
    for col, subset in enumerate([all_trains, late]):

        is_late = col == 1

        if not is_late:
            # Mean net delay change for all included trains
            weekday = (
                subset.groupby('weekday')['delay_change']
                .mean()
                .reindex(order)
            )

            hour = (
                subset.groupby('dep_hour')['delay_change']
                .mean()
                .reindex(hours)
            )

            ylabel = 'Net delay change (min/train)'
            title = 'All non-cancelled trains'

        else:
            # Percentage of arrival delay recovered among late arrivals
            def absorption_percentage(x):
                arr = x['arr_delay'].sum()
                return (
                    (arr - x['dep_delay'].sum()) / arr * 100
                    if arr > 0 else np.nan
                )

            weekday = (
                subset.groupby('weekday')
                .apply(absorption_percentage, include_groups=False)
                .reindex(order)
            )

            hour = (
                subset.groupby('dep_hour')
                .apply(absorption_percentage, include_groups=False)
                .reindex(hours)
            )

            ylabel = 'Delay absorption (%)'
            title = 'Trains arriving late'

        for row, (values, xlabel) in enumerate([
            (weekday, 'Weekday'),
            (hour, 'Departure hour')
        ]):
            ax = axes[row, col]
            values = values.astype(float)

            colours = [
                'C0' if value > 0
                else 'C3' if value < 0
                else 'lightgrey'
                for value in values.fillna(0)
            ]

            ax.bar(values.index.astype(str), values, color=colours)
            ax.axhline(0, color='black', linewidth=0.9)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(
                f'{title}: {xlabel.lower()}',
                loc='left',
                fontweight='bold'
            )
            ax.grid(axis='y', alpha=0.25)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)

            if row == 0:
                ax.tick_params(axis='x', rotation=35)
            if col == 0:
                ax.set_ylim(-1, 1)

    fig.suptitle(
        'Delay Recovery and Propagation',
        fontsize=15,
        fontweight='bold'
    )
    fig.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. Plot delay propagation categories
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    category_order = [
        'Delay reduced', 'Unchanged delay', 'Delay increased'
    ]

    category_colours = {
        'Delay reduced': 'C0',
        'Unchanged delay': 'lightgrey',
        'Delay increased': 'C3'
    }

    for j, threshold in enumerate((3, 1)):
        category = f'delay_propagation_{threshold}min'

        for i, (group, index, xlabel) in enumerate([
            ('weekday', order, 'Weekday'),
            ('dep_hour', hours, 'Departure hour')
        ]):
            tab = (
                comp.groupby([group, category], observed=False)
                .size()
                .unstack(fill_value=0)
                .reindex(index, fill_value=0)
                .reindex(columns=category_order, fill_value=0)
            )

            pct = tab.div(
                tab.sum(axis=1).replace(0, np.nan), axis=0
            ) * 100

            ax = axes[i, j]
            pct.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                color=[category_colours[c] for c in category_order],
                width=0.8
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Trains (%)')
            ax.set_title(
                f'{xlabel}: {threshold}-minute threshold',
                loc='left', fontweight='bold'
            )
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)

            if i == 0:
                ax.tick_params(axis='x', rotation=35)

            legend = ax.get_legend()
            if i == 0 and j == 1:
                ax.legend(
                    title='Delay change',
                    frameon=False,
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left'
                )
            elif legend is not None:
                legend.remove()

    fig.suptitle(
        'Delay Propagation by Weekday and Departure Hour',
        fontsize=15, fontweight='bold'
    )
    fig.tight_layout()
    plt.show()

    # return {
    #     'comparison': comp,
    #     'weekly_metrics': None,
    #     'mean_net_change_all': mean_all,
    #     'mean_net_change_selected': mean_selected,
    #     'overall_absorption': overall_absorption,
    #     'net_delay_change_per_train': net_per_train,
    #     'late_arrival_absorption': late_absorption
    # }


# ---------------------------------------------------------------------
# 4.5 - Correlation and dwell-time heatmaps
# ---------------------------------------------------------------------
def analyse_correlations(train, delay_comparison=None, train_punctual=None,
                         min_trains=15, min_arr_delay=60):

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------
    if train_punctual is None:
        punctual = train.loc[
            ~train['cancelled'].fillna(False)
        ].copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )
    else:
        punctual = train_punctual.copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )

    punctual['weekday'] = punctual['Date'].dt.day_name()

    if delay_comparison is None:
        comp = punctual.dropna(
            subset=['arr_delay', 'dep_delay']
        ).copy()

        comp = comp.loc[
            ~((comp['arr_delay'] == 0) & (comp['dep_delay'] == 0))
        ].copy()
    else:
        comp = delay_comparison.copy()
        comp['Date'] = pd.to_datetime(
            comp['Date'], errors='coerce'
        )

    comp['weekday'] = comp['Date'].dt.day_name()
    comp['dep_hour'] = pd.to_numeric(
        comp['dep_hour'], errors='coerce'
    )

    # Use exactly the same dwell-time calculation as the original
    sched_arr = pd.to_datetime(
        comp['scheduled_arrival'],
        format='%H:%M',
        errors='coerce'
    )
    sched_dep = pd.to_datetime(
        comp['scheduled_departure'],
        format='%H:%M',
        errors='coerce'
    )

    comp['dwell_time'] = (
        (sched_dep - sched_arr).dt.total_seconds() / 60
    ) % 1440

    dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
    dwell_labels = [
        '0-1', '2', '3', '4', '5',
        '6-7', '8-9', '10-14', '15+'
    ]

    comp['dwell_bin'] = pd.cut(
        comp['dwell_time'],
        bins=dwell_bins,
        labels=dwell_labels,
        right=False
    )

    # ---------------------------------------------------------
    # 2. Weekly correlation analysis
    # ---------------------------------------------------------
    train_weekly = train.copy()
    train_weekly['Date'] = pd.to_datetime(
        train_weekly['Date'], errors='coerce'
    )
    train_weekly = train_weekly.dropna(subset=['Date'])

    train_weekly['week'] = (
        train_weekly['Date'].dt.to_period('W-MON').astype(str)
    )

    weekly = train_weekly.groupby('week').agg(
        arrival_delayed=(
            'arr_delay', lambda x: (x > 3).mean() * 100
        ),
        departure_delayed=(
            'dep_delay', lambda x: (x > 3).mean() * 100
        ),
        cancellation_rate=(
            'cancelled',
            lambda x: x.fillna(False).mean() * 100
        ),
        scheduled_count=('train_id', 'count')
    )

    correlation = weekly.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        linecolor='white',
        cbar_kws={'label': 'Pearson correlation'},
        ax=ax
    )
    ax.set_title(
        'Correlation Between Weekly Operational Metrics',
        loc='left', fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3. Heatmap: delay absorption for late-arriving trains
    # ---------------------------------------------------------
    late = comp.loc[comp['arr_delay'] > 0].copy()

    grouped_late = late.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    grouped_late['absorption'] = (
        (grouped_late['arr_sum'] - grouped_late['dep_sum'])
        / grouped_late['arr_sum'] * 100
    ).where(
        (grouped_late['arr_sum'] >= min_arr_delay)
        & (grouped_late['n_trains'] >= min_trains)
    )

    heat_late = (
        grouped_late['absorption']
        .unstack('dwell_bin')
        .reindex(index=range(5, 23), columns=dwell_labels)
    )

    boundaries = [-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]
    cmap = plt.get_cmap('RdBu', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        heat_late,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'Delay absorption (%)',
            'ticks': boundaries
        },
        ax=ax
    )
    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel('Departure hour')
    ax.set_title(
        'Delay Absorption of Late-Arriving Trains by Dwell Time '
        'and Departure Hour'
    )
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. Heatmap: average net delay change for all trains
    # ---------------------------------------------------------
    grouped_all = comp.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    grouped_all['net_delay_change'] = (
        (grouped_all['dep_sum'] - grouped_all['arr_sum'])
        / grouped_all['n_trains']
    ).where(grouped_all['n_trains'] >= min_trains)

    heat_all = (
        grouped_all['net_delay_change']
        .unstack('dwell_bin')
        .reindex(index=range(5, 23), columns=dwell_labels)
    )

    valid_values = heat_all.to_numpy()
    valid_values = valid_values[np.isfinite(valid_values)]

    limit = (
        max(abs(valid_values.min()), abs(valid_values.max()), 1)
        if len(valid_values) else 1
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        heat_all,
        cmap='RdBu_r',
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'Net delay change (minutes per train)'
        },
        ax=ax
    )
    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel('Departure hour')
    ax.set_title(
        'Net Delay Change by Dwell Time and Departure Hour'
    )
    plt.tight_layout()
    plt.show()

    return {
        'weekly_data': weekly,
        'correlation': correlation,
        'late_arrival_heatmap': heat_late,
        'net_delay_heatmap': heat_all
    }


# ---------------------------------------------------------------------
# 4.5 - Correlation and dwell-time heatmaps
# ---------------------------------------------------------------------
def analyse_correlations(train, delay_comparison=None, train_punctual=None,
                         min_trains=15, min_arr_delay=60):

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------
    if train_punctual is None:
        punctual = train.loc[
            ~train['cancelled'].fillna(False)
        ].copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )
    else:
        punctual = train_punctual.copy()
        punctual['Date'] = pd.to_datetime(
            punctual['Date'], errors='coerce'
        )

    punctual['weekday'] = punctual['Date'].dt.day_name()

    if delay_comparison is None:
        comp = punctual.dropna(
            subset=['arr_delay', 'dep_delay']
        ).copy()

        comp = comp.loc[
            ~((comp['arr_delay'] == 0) & (comp['dep_delay'] == 0))
        ].copy()
    else:
        comp = delay_comparison.copy()
        comp['Date'] = pd.to_datetime(
            comp['Date'], errors='coerce'
        )

    comp['weekday'] = comp['Date'].dt.day_name()
    comp['dep_hour'] = pd.to_numeric(
        comp['dep_hour'], errors='coerce'
    )

    # Use exactly the same dwell-time calculation as the original
    sched_arr = pd.to_datetime(
        comp['scheduled_arrival'],
        format='%H:%M',
        errors='coerce'
    )
    sched_dep = pd.to_datetime(
        comp['scheduled_departure'],
        format='%H:%M',
        errors='coerce'
    )

    comp['dwell_time'] = (
        (sched_dep - sched_arr).dt.total_seconds() / 60
    ) % 1440

    dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
    dwell_labels = [
        '0-1', '2', '3', '4', '5',
        '6-7', '8-9', '10-14', '15+'
    ]

    comp['dwell_bin'] = pd.cut(
        comp['dwell_time'],
        bins=dwell_bins,
        labels=dwell_labels,
        right=False
    )

    # ---------------------------------------------------------
    # 2. Weekly correlation analysis
    # ---------------------------------------------------------
    train_weekly = train.copy()
    train_weekly['Date'] = pd.to_datetime(
        train_weekly['Date'], errors='coerce'
    )
    train_weekly = train_weekly.dropna(subset=['Date'])

    train_weekly['week'] = (
        train_weekly['Date'].dt.to_period('W-MON').astype(str)
    )

    weekly = train_weekly.groupby('week').agg(
        arrival_delayed=(
            'arr_delay', lambda x: (x > 3).mean() * 100
        ),
        departure_delayed=(
            'dep_delay', lambda x: (x > 3).mean() * 100
        ),
        cancellation_rate=(
            'cancelled',
            lambda x: x.fillna(False).mean() * 100
        ),
        scheduled_count=('train_id', 'count')
    )

    correlation = weekly.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        linecolor='white',
        cbar_kws={'label': 'Pearson correlation'},
        ax=ax
    )
    ax.set_title(
        'Correlation Between Weekly Operational Metrics',
        loc='left', fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3. Heatmap: delay absorption for late-arriving trains
    # ---------------------------------------------------------
    late = comp.loc[comp['arr_delay'] > 0].copy()

    grouped_late = late.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    grouped_late['absorption'] = (
        (grouped_late['arr_sum'] - grouped_late['dep_sum'])
        / grouped_late['arr_sum'] * 100
    ).where(
        (grouped_late['arr_sum'] >= min_arr_delay)
        & (grouped_late['n_trains'] >= min_trains)
    )

    heat_late = (
        grouped_late['absorption']
        .unstack('dwell_bin')
        .reindex(index=range(5, 23), columns=dwell_labels)
    )

    boundaries = [-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]
    cmap = plt.get_cmap('RdBu', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        heat_late,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'Delay absorption (%)',
            'ticks': boundaries
        },
        ax=ax
    )
    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel('Departure hour')
    ax.set_title(
        'Delay Absorption of Late-Arriving Trains by Dwell Time '
        'and Departure Hour'
    )
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. Heatmap: average net delay change for all trains
    # ---------------------------------------------------------
    grouped_all = comp.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    grouped_all['net_delay_change'] = (
        (grouped_all['dep_sum'] - grouped_all['arr_sum'])
        / grouped_all['n_trains']
    ).where(grouped_all['n_trains'] >= min_trains)

    heat_all = (
        grouped_all['net_delay_change']
        .unstack('dwell_bin')
        .reindex(index=range(5, 23), columns=dwell_labels)
    )

    valid_values = heat_all.to_numpy()
    valid_values = valid_values[np.isfinite(valid_values)]

    limit = (
        max(abs(valid_values.min()), abs(valid_values.max()), 1)
        if len(valid_values) else 1
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        heat_all,
        cmap='RdBu_r',
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label': 'Net delay change (minutes per train)'
        },
        ax=ax
    )
    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel('Departure hour')
    ax.set_title(
        'Net Delay Change by Dwell Time and Departure Hour'
    )
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Full analysis - retains the original two-value return pattern
# ---------------------------------------------------------------------
def train_analysis(data='data/traindata_total.csv', start_period='2025-12-15',
                   end_period='2026-04-30', verbose=True,
                   cancellation_percentile=75):
    train = load_train_data(data=data, start_period=start_period, end_period=end_period)

    # By default, exclude days above the 75th percentile of daily cancellations.
    train = filter_cancellation_days(
        train,
        cancellation_percentile=cancellation_percentile,
        verbose=verbose
    )

    results = analyse_delays(train, start_period, end_period, verbose=verbose)
    analyse_delays(train)
    analyse_delays_per_hour(train)
    train_punctual, punctuality_results = analyse_punctuality(train, verbose=verbose)
    delay_comparison, absorption_results = analyse_delay_absorption(
        train, train_punctual=train_punctual, verbose=verbose)
    analyse_correlations(train, delay_comparison=delay_comparison, train_punctual=train_punctual)
    results.update(punctuality_results)
    results.update(absorption_results)
    return train, results


if __name__ == '__main__':
    train, results = train_analysis()
    print('\nReturned results:')
    print(pd.Series(results))

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

    if verbose:
        print('\nCANCELLATION-DAY FILTER')
        print(f'Percentile used: {cancellation_percentile}th')
        print(f'Cancellation threshold: {threshold:.1f} per day')
        print(f'Days excluded: {len(excluded_days)} of {len(daily_cancellations)}')
        print(f'Trains removed: {len(train) - len(filtered_train)}')
        print(f'Trains remaining: {len(filtered_train)}')
        print("")

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
        ax.set_ylim(bottom=0)
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

    data = train.copy()
    if 'week' not in data:
        data['week'] = data['Date'].dt.to_period('W').astype(str)
    fig, axes = plt.subplots(2, 1, figsize=(14, 15))
    sns.boxplot(data=data[data['arr_delay'] > 3], x='week', y='arr_delay', showfliers=False, ax=axes[0])
    axes[0].set_title('Spread of Arrival delay > 3 minutes per week'); axes[0].set_ylabel('Arrival delay (minutes)')
    sns.boxplot(data=data[data['dep_delay'] > 3], x='week', y='dep_delay', showfliers=False, ax=axes[1])
    axes[1].set_title('Spread of Departure delay > 3 minutes per week'); axes[1].set_ylabel('Departure delay (minutes)')
    for ax in axes:
        ax.set_xlabel(''); ax.tick_params(axis='x', bottom=False, labelbottom=False, rotation=90)
    plt.tight_layout(); plt.show()

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
        arrival_mean=('arr_delay', 'mean'), departure_mean=('dep_delay', 'mean'),
        arrival_delayed=('arrival_delayed', 'mean'), departure_delayed=('departure_delayed', 'mean'),
        arrival_delayed_3=('arrival_delayed_3', 'mean'), departure_delayed_3=('departure_delayed_3', 'mean'),
        cancelled=('cancelled', 'mean'))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(hourly.index, hourly['arrival_mean'], marker='o', label='Arrival')
    axes[0].plot(hourly.index, hourly['departure_mean'], marker='o', label='Departure')
    axes[0].set_ylabel('Average delay (minutes)'); axes[0].set_title('Average delay by hour of day'); axes[0].legend(); axes[0].grid(alpha=.3)
    for col, label in [('arrival_delayed', 'Arrival > 0'), ('departure_delayed', 'Departure > 0'), ('arrival_delayed_3', 'Arrival >= 3'), ('departure_delayed_3', 'Departure >= 3')]:
        axes[1].plot(hourly.index, hourly[col] * 100, marker='o', label=label)
    axes[1].set_ylabel('Trains delayed (%)'); axes[1].set_title('Percentage of trains delayed'); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(hourly.index, hourly['cancelled'] * 100, marker='o', label='Cancelled')
    axes[2].set_xlabel('Hour of day'); axes[2].set_ylabel('Trains cancelled (%)'); axes[2].set_title('Percentage of trains cancelled'); axes[2].legend(); axes[2].grid(alpha=.3)
    axes[2].set_xticks(range(5, 23)); plt.tight_layout(); plt.show()
    # return hourly


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
    axes[0].plot(by_arr_hour.index, by_arr_hour.values, marker='o', label='Arrival')
    axes[0].plot(by_dep_hour.index, by_dep_hour.values, marker='o', label='Departure')
    axes[0].axhline(arr_mean, linestyle='--', color='C0', label=f'Mean arrival ({arr_mean:.1f}%)')
    axes[0].axhline(dep_mean, linestyle='--', color='C1', label=f'Mean departure ({dep_mean:.1f}%)')
    axes[0].axhline(total_mean, linestyle='--', color='C2', label=f'Mean overall ({total_mean:.1f}%)')
    axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Punctual trains (%)'); axes[0].set_title('Train punctuality per hour'); axes[0].set_xticks(range(5, 23)); axes[0].set_ylim(0, 100); axes[0].grid(); axes[0].legend()
    x = np.arange(len(weekday_order)); width = .35
    axes[1].bar(x - width/2, weekly['arr_punctual'], width, label='Arrival')
    axes[1].bar(x + width/2, weekly['dep_punctual'], width, label='Departure')
    axes[1].axhline(arr_mean, linestyle='--', color='C0', label=f'Mean arrival ({arr_mean:.1f}%)')
    axes[1].axhline(dep_mean, linestyle='--', color='C1', label=f'Mean departure ({dep_mean:.1f}%)')
    axes[1].axhline(total_mean, linestyle='--', color='C2', label=f'Mean overall ({total_mean:.1f}%)')
    axes[1].set_xlabel('Day of the week'); axes[1].set_ylabel('Punctual trains (%)'); axes[1].set_title('Train punctuality per weekday'); axes[1].set_xticks(x); axes[1].set_xticklabels(weekday_order, rotation=45, ha='right'); axes[1].set_ylim(0, 100); axes[1].grid(axis='y'); axes[1].legend()
    plt.tight_layout(); plt.show()

    both = punctual['both_punctual'].mean() * 100
    if verbose:
        print(f'Trains punctual at both arrival and departure: {both:.2f}%')
        print(f'Trains punctual at arrival: {arr_mean:.2f}%')
        print(f'Trains punctual at departure: {dep_mean:.2f}%')
    # return punctual, {'Both arrival and departure punctual (%)': both, 'Arrival punctual (%)': arr_mean, 'Departure punctual (%)': dep_mean}


# ---------------------------------------------------------------------
# 4.4 - Delay absorption and propagation
# ---------------------------------------------------------------------
def analyse_delay_absorption(train, delay_comparison=None, train_punctual=None, verbose=True):
    if train_punctual is None:
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
    else:
        punctual = train_punctual.copy()

    if delay_comparison is None:
        comp = punctual.dropna(subset=['arr_delay', 'dep_delay']).copy()
        comp = comp[~((comp['arr_delay'] == 0) & (comp['dep_delay'] == 0))].copy()
        comp['delay_change'] = comp['arr_delay'] - comp['dep_delay']  # positive = absorbed
        comp['weekday'] = comp['Date'].dt.day_name()
        sched_arr = pd.to_datetime(comp['scheduled_arrival'], format='%H:%M')
        sched_dep = pd.to_datetime(comp['scheduled_departure'], format='%H:%M')
        comp['dwell_time'] = ((sched_dep - sched_arr).dt.total_seconds() / 60) % 1440
        dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
        dwell_labels = ['0-1', '2', '3', '4', '5', '6-7', '8-9', '10-14', '15+']
        comp['dwell_bin'] = pd.cut(
            comp['dwell_time'], bins=dwell_bins, labels=dwell_labels, right=False)
        for threshold in (3, 1):
            comp[f'delay_propagation_{threshold}min'] = np.select(
                [comp['delay_change'] > threshold, comp['delay_change'] < -threshold],
                ['Delay increased', 'Delay reduced'], default='Unchanged delay')
    else:
        comp = delay_comparison.copy()
    mean_all = (punctual['arr_delay'] - punctual['dep_delay']).mean().round(1)
    mean_selected = comp['delay_change'].mean().round(1)
    arr_sum, dep_sum = comp['arr_delay'].sum(), comp['dep_delay'].sum()
    overall_absorption = (arr_sum - dep_sum) / arr_sum * 100 if arr_sum else np.nan
    net_per_train = (dep_sum - arr_sum) / comp['train_id'].count() if len(comp) else np.nan
    late = comp[comp['arr_delay'] > 0]
    late_absorption = ((late['arr_delay'].sum() - late['dep_delay'].sum()) / late['arr_delay'].sum() * 100) if late['arr_delay'].sum() else np.nan
    net_change = dep_sum - arr_sum
    if verbose:
        print(f'The mean train delay change in total set equals: {mean_all} minutes')
        print(f'The mean train delay change in selected set equals: {mean_selected} minutes')
        print(f"The maximum delay accumulated is {comp['delay_change'].min()} minutes and the maximum delay absorbed is {comp['delay_change'].max()} minutes")
        print(f'Overall delay absorption: {overall_absorption:.2f}%')
        print(f'Overall net delay change: {net_per_train:.2f} minutes per train')
        print(f'Late arrival delay absorption: {late_absorption:.2f}%')
        print(f'Net delay change (departure delay minus arrival delay): {net_change:.1f} minutes')

    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    late = comp[comp['arr_delay'] > 0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, subset in enumerate([comp, late]):
        if col == 0:
            weekday = subset.groupby('weekday').apply(lambda x: (x['dep_delay'].sum() - x['arr_delay'].sum()) / len(x)).reindex(order)
            hour = subset.groupby('dep_hour').apply(lambda x: (x['dep_delay'].sum() - x['arr_delay'].sum()) / len(x)).reindex(range(5, 23))
            ylabel = 'Net delay change (min/train)'
        else:
            weekday = subset.groupby('weekday').apply(lambda x: ((x['arr_delay'].sum() - x['dep_delay'].sum()) / x['arr_delay'].sum() * 100) if x['arr_delay'].sum() else np.nan).reindex(order)
            hour = subset.groupby('dep_hour').apply(lambda x: ((x['arr_delay'].sum() - x['dep_delay'].sum()) / x['arr_delay'].sum() * 100) if x['arr_delay'].sum() else np.nan).reindex(range(5, 23))
            ylabel = 'Delay absorption (%)'
        weekday.plot(kind='bar', ax=axes[0, col]); hour.plot(kind='bar', ax=axes[1, col])
        axes[0, col].set_title(['All trains', 'Late arrivals'][col] + ': Weekday')
        axes[1, col].set_title(['All trains', 'Late arrivals'][col] + ': Departure hour')
        axes[0, col].set_xlabel('Weekday'); axes[1, col].set_xlabel('Departure hour')
        for ax in axes[:, col]: ax.set_ylabel(ylabel); ax.axhline(0, color='black', linewidth=.8)
        axes[0, col].tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for j, threshold in enumerate((3, 1)):
        category = f'delay_propagation_{threshold}min'
        for i, (group, index, xlabel) in enumerate([
            ('weekday', order, 'Weekday'), ('dep_hour', list(range(5, 23)), 'Departure hour')]):
            tab = comp.groupby([group, category]).size().unstack(fill_value=0).reindex(index, fill_value=0)
            pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100
            ax = axes[i, j]
            pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_xlabel(xlabel); ax.set_ylabel('Trains (%)')
            ax.set_title(f'Delay propagation by {xlabel.lower()} ({threshold}-minute threshold)')
            ax.set_ylim(0, 100)
            if i == 0: ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.show()

    # return comp, {
    #     'Mean delay change (all non-cancelled trains, min)': mean_all,
    #     'Mean delay change (selected set, min)': mean_selected,
    #     'Overall delay absorption (%)': overall_absorption,
    #     'Overall net delay change (min/train)': net_per_train,
    #     'Late arrival delay absorption (%)': late_absorption,
    #     'Total arrival delay (min)': arr_sum,
    #     'Total departure delay (min)': dep_sum,
    #     'Net delay change (min)': net_change
    # }


# ---------------------------------------------------------------------
# 4.5 - Correlation and dwell-time heatmaps
# ---------------------------------------------------------------------
def analyse_correlations(train, delay_comparison=None, train_punctual=None,
                         min_trains=15, min_arr_delay=60):
    if train_punctual is None:
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
    else:
        punctual = train_punctual.copy()

    if delay_comparison is None:
        comp = punctual.dropna(subset=['arr_delay', 'dep_delay']).copy()
        comp = comp[~((comp['arr_delay'] == 0) & (comp['dep_delay'] == 0))].copy()
        comp['delay_change'] = comp['arr_delay'] - comp['dep_delay']  # positive = absorbed
        comp['weekday'] = comp['Date'].dt.day_name()
        sched_arr = pd.to_datetime(comp['scheduled_arrival'], format='%H:%M')
        sched_dep = pd.to_datetime(comp['scheduled_departure'], format='%H:%M')
        comp['dwell_time'] = ((sched_dep - sched_arr).dt.total_seconds() / 60) % 1440
        dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
        dwell_labels = ['0-1', '2', '3', '4', '5', '6-7', '8-9', '10-14', '15+']
        comp['dwell_bin'] = pd.cut(
            comp['dwell_time'], bins=dwell_bins, labels=dwell_labels, right=False)
        for threshold in (3, 1):
            comp[f'delay_propagation_{threshold}min'] = np.select(
                [comp['delay_change'] > threshold, comp['delay_change'] < -threshold],
                ['Delay increased', 'Delay reduced'], default='Unchanged delay')
    else:
        comp = delay_comparison.copy()
    weekly = train.groupby('week').agg(
        arrival_delayed=('arr_delay', lambda x: (x > 3).mean() * 100),
        departure_delayed=('dep_delay', lambda x: (x > 3).mean() * 100),
        cancellation_rate=('cancelled', 'mean'), scheduled_count=('train_id', 'count'))
    weekly['cancellation_rate'] *= 100
    sns.heatmap(weekly.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Between Weekly Delay and Cancellation Rates'); plt.tight_layout(); plt.show()

    dwell_labels = ['0-1', '2', '3', '4', '5', '6-7', '8-9', '10-14', '15+']
    late = comp[comp['arr_delay'] > 0]
    grouped = late.groupby(['dep_hour', 'dwell_bin'], observed=False).agg(
        arr_sum=('arr_delay', 'sum'), dep_sum=('dep_delay', 'sum'), n_trains=('train_id', 'count'))
    grouped['absorption'] = ((grouped['arr_sum'] - grouped['dep_sum']) / grouped['arr_sum'] * 100).where(
        (grouped['arr_sum'] >= min_arr_delay) & (grouped['n_trains'] >= min_trains))
    heat_late = grouped['absorption'].unstack('dwell_bin').reindex(index=range(5, 23), columns=dwell_labels)
    boundaries = [-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]
    cmap = plt.get_cmap('RdBu', len(boundaries) - 1); norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(heat_late, cmap=cmap, norm=norm, annot=True, fmt='.0f', linewidths=.5, linecolor='white', cbar_kws={'label': 'Delay absorption (%)', 'ticks': boundaries}, ax=ax)
    ax.set_facecolor('lightgrey'); ax.set_xlabel('Scheduled dwell time (minutes)'); ax.set_ylabel('Departure hour')
    ax.set_title('Delay absorption of late-arriving trains by dwell time and departure hour'); plt.tight_layout(); plt.show()

    grouped_all = comp.groupby(['dep_hour', 'dwell_bin'], observed=False).agg(
        arr_sum=('arr_delay', 'sum'), dep_sum=('dep_delay', 'sum'), n_trains=('train_id', 'count'))
    grouped_all['net_delay_change'] = ((grouped_all['dep_sum'] - grouped_all['arr_sum']) / grouped_all['n_trains']).where(grouped_all['n_trains'] >= min_trains)
    heat_all = grouped_all['net_delay_change'].unstack('dwell_bin').reindex(index=range(5, 23), columns=dwell_labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(heat_all, cmap='RdBu_r', center=0, annot=True, fmt='.1f', linewidths=.5, linecolor='white', cbar_kws={'label': 'Net delay change (minutes per train)'}, ax=ax)
    ax.set_facecolor('lightgrey'); ax.set_xlabel('Scheduled dwell time (minutes)'); ax.set_ylabel('Departure hour')
    ax.set_title('Net delay change by dwell time and departure hour'); plt.tight_layout(); plt.show()
    # return {'weekly_data': weekly, 'late_arrival_heatmap': heat_late, 'net_delay_heatmap': heat_all}


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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm

def train_analysis(data='data/traindata_total.csv',
                   start_period='2025-12-15',
                   end_period='2026-04-30',
                   verbose = False):

    train = pd.read_csv(data)

    train['Service:Date'] = pd.to_datetime(train['Service:Date'])

    train = train[
        (train['Service:Date'] >= start_period) &
        (train['Service:Date'] <= end_period)
    ].copy()

    # Data cleaning & column modification
    train = train.drop(columns=['index', 'Service:RDT-ID'])
    train = train.rename(columns={
            'Service:Type': 'Type',
            'Service:Date': 'Date',
            'Stop:Arrival time': 'scheduled_arrival',
            'Stop:Departure time': 'scheduled_departure',
            'Stop:Arrival cancelled': 'arr_cancelled',
            'Stop:Departure cancelled': 'dep_cancelled',
            'Stop:Arrival delay': 'arr_delay',
            'Stop:Departure delay': 'dep_delay'
        })

    train.columns = (train.columns
                        .str.replace('Service:', '', regex=False)
                        .str.replace('Stop:', '', regex=False)
                        .str.replace('RDT-ID', 'train_id', regex=False))

    cols_drop = [
            'Completely cancelled', 'Partly cancelled', 'Maximum delay',
            'Train number', 'Station name', 'Station code',
            'Platform change', 'Planned platform', 'Actual platform', 'Company'
        ]

    train = train.drop(columns=cols_drop)
    train = train[train['Type'].isin(['Intercity', 'Sprinter'])]

    time_cols = ['scheduled_arrival', 'scheduled_departure']
    train[time_cols] = train[time_cols].apply(
            lambda col: pd.to_datetime(col, utc=True).dt.strftime('%H:%M')
        )


    for time, delay in [
        ('arrival', 'arr_delay'),
        ('departure', 'dep_delay')
    ]:
        train[f'scheduled_{time}'] = pd.to_datetime(
            train[f'scheduled_{time}'], format='%H:%M'
        )

        train[f'actual_{time}'] = (
            train[f'scheduled_{time}'] +
            pd.to_timedelta(train[delay], unit='m')
        )

        train[f'scheduled_{time}'] = train[f'scheduled_{time}'].dt.strftime('%H:%M')
        train[f'actual_{time}'] = train[f'actual_{time}'].dt.strftime('%H:%M')


    train['arr_hour'] = pd.to_datetime(
            train['actual_arrival'], format='%H:%M', errors='coerce'
        ).dt.hour

    train['dep_hour'] = pd.to_datetime(
            train['actual_departure'], format='%H:%M', errors='coerce'
        ).dt.hour

    train = train[
            train['dep_hour'].between(5, 22) &
            train['arr_hour'].between(5, 22)
        ]

    train = train.sort_values(
            ['Date', 'actual_arrival']
        ).reset_index(drop=True)


    # Initializing new variable to flag delay and delay with a three minute threshold
    train['arrival_delayed'] = train['arr_delay'] > 0
    train['departure_delayed'] = train['dep_delay'] > 0
    train['arrival_delayed_3'] = train['arr_delay'] >= 3
    train['departure_delayed_3'] = train['dep_delay'] >= 3

    # Train is cancelled if either arrival or departure is cancelled. Fill NaN values with False for this.
    train['cancelled'] = train['arr_cancelled'].fillna(False) | train['dep_cancelled'].fillna(False)

# ----------------------------- #
# ---- Data Analayis start ---- #
# ----------------------------- #

# -------------------------------------------------- #
# ---- 4.1 - Delays and cancellations in numbers --- #
# -------------------------------------------------- #

    delayed_trains = (len(train[(train['arrival_delayed'] == True) | (train['departure_delayed'] == True)])) / len(train) * 100
    delayed_operations = len(train[(train['arrival_delayed_3'] == True) | (train['departure_delayed_3'] == True)]) / len(train) * 100
    total_cancellations = len(train[train['cancelled'] == True])
    total_cancellations_perct = total_cancellations / len(train) * 100

    # Relative delay and cancellations
    total_vehicles = len(train)
    total_days = train['Date'].dt.normalize().nunique()

    total_delay_minutes = (
        train['arr_delay'].clip(lower=0).fillna(0).sum() +
        train['dep_delay'].clip(lower=0).fillna(0).sum()
    )

    delay_per_vehicle = total_delay_minutes / total_vehicles
    delay_per_day = total_delay_minutes / total_days
    cancellations_per_100_vehicles = total_cancellations / total_vehicles * 100
    cancellations_per_day = total_cancellations / total_days

    print(f"DELAYS AND CANCELLATIONS BETWEEN: {start_period} AND {end_period}")
    print(f"Number of complete cancellations: {total_cancellations}")
    print(f"    {delayed_trains:.1f}% of trains were delayed.")
    print(f"    {delayed_operations:.1f}% of train operations were delayed by more than 3 minutes.")
    print(f"    {total_cancellations_perct:.1f}% of trains were cancelled.")

    print("\nRELATIVE PERFORMANCE")
    print(f"    Delay per vehicle: {delay_per_vehicle:.2f} minutes")
    print(f"    Delay per day: {delay_per_day:.1f} minutes")
    print(f"    Cancellations per 100 vehicles: {cancellations_per_100_vehicles:.2f}")
    print(f"    Cancellations per day: {cancellations_per_day:.2f}")


# ---- Delays with 0 and 3 minute thresholds over analysis time
    daily_delay = train.groupby('Date').agg(
        arrival_delayed=('arrival_delayed', 'mean'),
        departure_delayed=('departure_delayed', 'mean'),
        arrival_delayed_3=('arrival_delayed_3', 'mean'),
        departure_delayed_3=('departure_delayed_3', 'mean'),
        cancelled=('cancelled', 'mean'),
        arr_cancelled=('arr_cancelled', 'mean'),
        dep_cancelled=('dep_cancelled', 'mean')
    ) * 100

    average_arrival_delay = train['arrival_delayed'].mean() * 100
    average_departure_delay = train['departure_delayed'].mean() * 100
    average_arrival_delay_3 = train['arrival_delayed_3'].mean() * 100
    average_departure_delay_3 = train['departure_delayed_3'].mean() * 100
    cancelled_percentage = train['cancelled'].mean() * 100
    arr_cancelled_percentage = train['arr_cancelled'].mean() * 100
    dep_cancelled_percentage = train['dep_cancelled'].mean() * 100

    daily_delay.index = pd.to_datetime(daily_delay.index)

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # 0-minute threshold
    ax[0].plot(daily_delay.index, daily_delay['arrival_delayed'], label='Arrival Delays')
    ax[0].plot(daily_delay.index, daily_delay['departure_delayed'], label='Departure Delays')

    ax[0].axhline(average_arrival_delay, linestyle='--',
                color=ax[0].lines[0].get_color(),
                label=f'Average Arrival Delay: {average_arrival_delay:.2f}%')
    ax[0].axhline(average_departure_delay, linestyle='--',
                color=ax[0].lines[1].get_color(),
                label=f'Average Departure Delay: {average_departure_delay:.2f}%')

    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Mean Percentage of Delayed Trains')
    ax[0].set_title('Delays with 0 Minute Threshold')
    ax[0].legend()


    # 3-minute threshold
    ax[1].plot(daily_delay.index, daily_delay['arrival_delayed_3'], label='Arrival Delays')
    ax[1].plot(daily_delay.index, daily_delay['departure_delayed_3'], label='Departure Delays')

    ax[1].axhline(average_arrival_delay_3, linestyle='--',
                color=ax[1].lines[0].get_color(),
                label=f'Average Arrival Delay: {average_arrival_delay_3:.2f}%')
    ax[1].axhline(average_departure_delay_3, linestyle='--',
                color=ax[1].lines[1].get_color(),
                label=f'Average Departure Delay: {average_departure_delay_3:.2f}%')

    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Mean Percentage of Delayed Trains')
    ax[1].set_title('Delays with 3 Minute Threshold')
    ax[1].legend()

    max_value = max(
        daily_delay[['arrival_delayed', 'departure_delayed',
                    'arrival_delayed_3', 'departure_delayed_3']].max().max(),
        average_arrival_delay,
        average_departure_delay,
        average_arrival_delay_3,
        average_departure_delay_3
    )

    for a in ax:
        a.set_ylim(0, max_value + 5)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------- #
# ---- 4.2 - Spread of delays using boxplots ---- #
# ----------------------------------------------- #
    train['week'] = train['Date'].dt.to_period('W').astype(str)

    # Filter delays greater than 3 minutes
    train_arr_delayed = train[train['arr_delay'] > 3]
    train_dep_delayed = train[train['dep_delay'] > 3]

    fig, ax = plt.subplots(2, 1, figsize=(14, 15))

    # Arrival delays
    sns.boxplot(
        data=train_arr_delayed,
        x='week',
        y='arr_delay',
        showfliers=False,
        ax=ax[0]
    )
    ax[0].set_title('Spread of Arrival delay > 3 minutes per week')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Arrival delay (minutes)')
    ax[0].tick_params(axis='x', bottom = False, labelbottom = False)

    sns.boxplot(
        data=train_dep_delayed,
        x='week',
        y='dep_delay',
        showfliers=False,
        ax=ax[1]
    )
    ax[1].set_title('Spread of Departure delay > 3 minutes per week')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Departure delay (minutes)')
    ax[1].tick_params(axis='x', bottom = False, labelbottom = False)

    # Formatting
    for a in ax:
        a.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

# ----------------------------- #
# ---- 4.3 Delays per hour ---- #
# ----------------------------- #
    hourly_train = train.groupby('arr_hour').agg(
        arrival_mean = ('arr_delay', 'mean'),
        departure_mean = ('dep_delay', 'mean'),
        arrival_delayed = ('arrival_delayed', 'mean'),
        departure_delayed = ('departure_delayed', 'mean'),
        arrival_delayed_3 = ('arrival_delayed_3', 'mean'),
        departure_delayed_3 = ('departure_delayed_3', 'mean'),
        cancelled = ('cancelled', 'mean')
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Average delay
    ax1.plot(
        hourly_train.index,
        hourly_train['arrival_mean'],
        marker='o',
        label='Arrival'
    )

    ax1.plot(
        hourly_train.index,
        hourly_train['departure_mean'],
        marker='o',
        label='Departure'
    )

    ax1.set_ylabel('Average delay (minutes)')
    ax1.set_title('Average delay by hour of day')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Percentage delayed by at least 3 minutes
    ax2.plot(
        hourly_train.index,
        hourly_train['arrival_delayed'] * 100,
        marker='o',
        label='Arrival'
    )

    ax2.plot(
        hourly_train.index,
        hourly_train['departure_delayed'] * 100,
        marker='o',
        label='Departure'
    )


    ax2.plot(
        hourly_train.index,
        hourly_train['arrival_delayed_3'] * 100,
        marker='o',
        label='Arrival >= 3'
    )

    ax2.plot(
        hourly_train.index,
        hourly_train['departure_delayed_3'] * 100,
        marker='o',
        label='Departure >= 3'
    )

    ax2.set_xlabel('Hour of day')
    ax2.set_ylabel('Trains delayed >= 3 min (%)')
    ax2.set_title('Percentage of trains delayed')
    ax2.legend()
    ax2.set_xticks(range(5, 23))
    ax2.grid(alpha=0.3)


    # Plot 3: Average cancellations
    ax3.plot(
        hourly_train.index,
        hourly_train['cancelled'] * 100,
        marker='o',
        label='Cancelled'
    )

    ax3.set_xlabel('Hour of day')
    ax3.set_ylabel('Trains cancelled (%)')
    ax3.set_title('Percentage of trains cancelled')
    ax3.legend()
    ax3.set_xticks(range(5, 23))
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---- 4.4 - Punctuality indicator ---- #
    train_punctual = train[train['cancelled'] == False] # Keeping only non-cancelled trains for punctuality analysis

    train_punctual['delay_status'] = train_punctual['arr_delay'].apply(lambda x: 'On time' if x <= 0 else ('Delayed' if x <= 3 else 'Significantly Delayed'))
    train_punctual['arr_punctual'] = train_punctual['arr_delay'] < 3

    train_punctual["dep_delay_status"] = train_punctual["dep_delay"].apply(lambda x: "On time" if x <= 0 else ("Delayed" if x < 3 else "Significantly Delayed"))
    train_punctual["dep_punctual"] = train_punctual["dep_delay"] < 3

    # Plot train punctuality per hour and weekday
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    arr_mean = train_punctual["arr_punctual"].mean() * 100
    dep_mean = train_punctual["dep_punctual"].mean() * 100
    tot_mean = (
        train_punctual["arr_punctual"].sum()
        + train_punctual["dep_punctual"].sum()
    ) / (
        train_punctual["arr_punctual"].count()
        + train_punctual["dep_punctual"].count()
    ) * 100


    # Plot 1: Punctuality per hour
    arr_punctual = train_punctual.groupby("arr_hour")["arr_punctual"].mean() * 100
    dep_punctual = train_punctual.groupby("dep_hour")["dep_punctual"].mean() * 100

    ax[0].plot(arr_punctual.index, arr_punctual.values,
            marker="o", label="Arrival")
    ax[0].plot(dep_punctual.index, dep_punctual.values,
            marker="o", label="Departure")

    ax[0].axhline(arr_mean, linestyle="--", color="C0", label=f"Mean arrival ({arr_mean:.1f}%)")
    ax[0].axhline(dep_mean, linestyle="--", color="C1",label=f"Mean departure ({dep_mean:.1f}%)")
    ax[0].axhline(tot_mean, linestyle="--", color="C2",label=f"Mean punctuality ({tot_mean:.1f}%)")

    ax[0].set_xlabel("Hour")
    ax[0].set_ylabel("Punctual trains (%)")
    ax[0].set_title("Train punctuality per hour")
    ax[0].set_xticks(range(5, 23))
    ax[0].set_ylim(min(arr_punctual.min(), dep_punctual.min()) - 5, 100)
    ax[0].grid()
    ax[0].legend()


    # Plot 2: Punctuality per weekday
    train_punctual["Date"] = pd.to_datetime(train_punctual["Date"])
    train_punctual["weekday"] = train_punctual["Date"].dt.day_name()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

    weekly = train_punctual.groupby("weekday")[
        ["arr_punctual", "dep_punctual"]
    ].mean().mul(100).reindex(weekday_order)

    x = range(len(weekday_order))
    width = 0.35

    # Grey areas above each bar, up to 100%
    for i, col in enumerate(["arr_punctual", "dep_punctual"]):
        positions = [j + (i - 0.5) * width for j in x]
        values = weekly[col].values

        ax[1].bar(
            positions, 100 - values, width,
            bottom=values, color="lightgrey",
            edgecolor="none", zorder=1
        )

    for i, (col, label) in enumerate([
        ("arr_punctual", "Arrival"),
        ("dep_punctual", "Departure")
    ]):
        positions = [j + (i - 0.5) * width for j in x]
        values = weekly[col].values

        ax[1].bar(
            positions, values - 80, width,
            bottom=80, label=label, zorder=2
        )

    # Overall mean lines
    ax[1].axhline(arr_mean, linestyle="--", color="C0", label=f"Mean arrival ({arr_mean:.1f}%)", zorder=3)
    ax[1].axhline(dep_mean, linestyle="--", color="C1", label=f"Mean departure ({dep_mean:.1f}%)", zorder=3)
    ax[1].axhline(tot_mean, linestyle="--", color="C2",label=f"Mean punctuality ({tot_mean:.1f}%)")

    ax[1].set_xlabel("Day of the week")
    ax[1].set_ylabel("Punctual trains (%)")
    ax[1].set_title("Train punctuality per weekday")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(weekday_order, rotation=45, ha="right")
    ax[1].set_ylim(min(weekly["arr_punctual"].min(), weekly["dep_punctual"].min()) - 5, 100)
    ax[1].grid(axis="y", zorder=0)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Print punctuality statistics
    train_punctual["both_punctual"] = (train_punctual["arr_punctual"] & train_punctual["dep_punctual"])
    total_punctuality = train_punctual["both_punctual"].mean() * 100

    print(f"Trains punctual at both arrival and departure:  {total_punctuality:.2f}%")
    print(f"Trains punctual at arrival:                     {arr_mean:.2f}%")
    print(f"Trains punctual at departure:                   {dep_mean:.2f}%")

# -------------------------------- #
# ---- 4.5 - Delay absorption ---- #
# -------------------------------- #
    # Delay change statistic
    # Total set
    train_punctual['delay_change'] = train_punctual['arr_delay'] - train_punctual['dep_delay']

    mean_delay_change = train_punctual['delay_change'].mean().round(1)

    if mean_delay_change > 0:
        print(f"The mean train delay change in total set equals: {mean_delay_change} minutes (delays absorbed)")
    elif mean_delay_change == 0:
        print(f"The mean train delay change in total set equals: {mean_delay_change} minutes (no change)")
    else:
        print(f"The mean train delay change in total set equals: {mean_delay_change} minutes (delays accumulated)")
    print("")

    # Selected set -> late arrivals only
    delay_comparison = train_punctual.dropna(subset = ['arr_delay', 'dep_delay']).copy() # Filter for arrivals and departures both only -> no arrival or departure solo as a delay metric.
    delay_comparison = delay_comparison.drop(columns = ['week', 'delay_status', 'arr_punctual', 'dep_delay_status', 'dep_punctual','both_punctual'])
    delay_comparison = delay_comparison[~((delay_comparison["arr_delay"] == 0) & (delay_comparison['dep_delay'] == 0))]               # Filter for delays actually being present
    mean_delay_change_comparison = delay_comparison['delay_change'].mean().round(1)

    if mean_delay_change_comparison > 0:
        print(f"The mean train delay change in selected set equals: {mean_delay_change_comparison} minutes (delays absorbed)")
    elif mean_delay_change_comparison == 0:
        print(f"The mean train delay change in selected set equals: {mean_delay_change_comparison} minutes (no change)")
    else:
        print(f"The mean train delay change in selected set equals: {mean_delay_change_comparison} minutes (delays accumulated)")

    print(f"The maximum delay accumulated is {delay_comparison['delay_change'].min()} minutes and the maximum delay absorbed is {delay_comparison['delay_change'].max()} minutes")

    # -- Absorption factor! -- #

    # Absorption/accumulation of delays of all delayed trains (dep & arr)
    delay_comparison['absorption'] = (
        (delay_comparison["arr_delay"] - delay_comparison["dep_delay"])
        / delay_comparison["arr_delay"]
        * 100
    )

    overall_absorption = (
        (delay_comparison["arr_delay"].sum() - delay_comparison["dep_delay"].sum())
        / delay_comparison["arr_delay"].sum()
        * 100
    )

    print(f"Overall delay absorption: {overall_absorption:.2f}%")

    # Net delay change for all trains (dep & arr)
    overall_net_delay_change = (
        (delay_comparison["dep_delay"].sum() - delay_comparison["arr_delay"].sum())
        / delay_comparison["train_id"].count()
    )

    print(f"Overall net delay change: {overall_net_delay_change:.2f} minutes per train")


    # Absorption/accumulation of delays of late-arriving trains
    valid_delays = delay_comparison[delay_comparison["arr_delay"] > 0]

    late_absorption = (
        (valid_delays["arr_delay"].sum() - valid_delays["dep_delay"].sum())
        / valid_delays["arr_delay"].sum()
        * 100
    )

    print(f"Late arrival delay absorption: {late_absorption:.2f}%")

    total_arr_delay = delay_comparison["arr_delay"].sum()
    total_dep_delay = delay_comparison["dep_delay"].sum()

    net_delay_change = total_dep_delay - total_arr_delay

    if net_delay_change < 0:
        print(f"Over the period: The total departure delay was {abs(net_delay_change):.1f} minutes lower than the total arrival delay")
        print(f"Per day:         The total departure delay was {abs(net_delay_change/train['Date'].nunique()):.1f} minutes lower than the total arrival delay")
    else:
        print(f"The total departure delay was {net_delay_change:.2f} minutes higher than the total arrival delay over the period")


    # Select all trains and late arrivals
    all_trains = delay_comparison
    late_arrivals = delay_comparison[delay_comparison['arr_delay'] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col, data in enumerate([all_trains, late_arrivals]):
        if col == 0:
            # Net delay change for all trains (minutes per train)
            weekday_metric = (
                data.groupby('weekday')
                .apply(lambda x: (x['dep_delay'].sum() - x['arr_delay'].sum())
                    / len(x))
                .reindex(weekday_order)
            )

            hour_metric = (
                data.groupby('dep_hour')
                .apply(lambda x: (x['dep_delay'].sum() - x['arr_delay'].sum())
                    / len(x))
                .reindex(range(5, 23))
            )

            ylabel = 'Net delay change (min/train)'
        else:
            # Delay absorption for late-arriving trains (%)
            weekday_metric = (
                data.groupby('weekday')
                .apply(lambda x: (x['arr_delay'].sum() - x['dep_delay'].sum())
                    / x['arr_delay'].sum() * 100
                    if x['arr_delay'].sum() != 0 else np.nan)
                .reindex(weekday_order)
            )

            hour_metric = (
                data.groupby('dep_hour')
                .apply(lambda x: (x['arr_delay'].sum() - x['dep_delay'].sum())
                    / x['arr_delay'].sum() * 100
                    if x['arr_delay'].sum() != 0 else np.nan)
                .reindex(range(5, 23))
            )

            ylabel = 'Delay absorption (%)'

        weekday_metric.plot(kind='bar', ax=axes[0, col])
        hour_metric.plot(kind='bar', ax=axes[1, col])

        axes[0, col].set_title(['All trains', 'Late arrivals'][col] + ': Weekday')
        axes[1, col].set_title(['All trains', 'Late arrivals'][col] + ': Departure hour')

        axes[0, col].set_xlabel('Weekday')
        axes[1, col].set_xlabel('Departure hour')

        for ax in axes[:, col]:
            ax.set_ylabel(ylabel)
            ax.axhline(0, color='black', linewidth=0.8)

        axes[0, col].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


    # Delay propagation
    # Classify delay changes using 3-minute and 1-minute thresholds -> Compacted 4 loose codes using AI
    for threshold in [3, 1]:
        conditions = [
            delay_comparison['delay_change'] > threshold,
            delay_comparison['delay_change'] < -threshold
        ]

        choices = [
            'Delay increased',
            'Delay reduced'
        ]

        delay_comparison[f'delay_propagation_{threshold}min'] = np.select(
            conditions, choices, default='Unchanged delay'
        )

    delay_comparison['weekday'] = pd.to_datetime(
        delay_comparison['Date']
    ).dt.day_name()

    weekday_order = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, threshold in enumerate([3, 1]):

        category = f'delay_propagation_{threshold}min'

        # Weekday distribution
        weekday_propagation = (
            delay_comparison
            .groupby(['weekday', category])
            .size()
            .unstack(fill_value=0)
            .reindex(weekday_order, fill_value=0)
        )

        weekday_propagation_pct = weekday_propagation.div(
            weekday_propagation.sum(axis=1).replace(0, np.nan),
            axis=0
        ) * 100

        ax = weekday_propagation_pct.plot(
            kind='bar',
            stacked=True,
            ax=axes[0, col]
        )

        for container in ax.containers:
            labels = [
                f'{v:.1f}%' if v >= 3 else ''
                for v in container.datavalues
            ]
            ax.bar_label(
                container,
                labels=labels,
                label_type='center',
                fontsize=8
            )

        ax.set_xlabel('Weekday')
        ax.set_ylabel('Percentage of trains (%)')
        ax.set_title(f'Delay Propagation by Weekday ({threshold}-minute threshold)')
        ax.set_xticklabels(weekday_order, rotation=45)
        ax.set_ylim(0, 100)
        ax.legend(title='Delay change')

        # Departure-hour distribution
        hourly_propagation = (
            delay_comparison
            .groupby(['dep_hour', category])
            .size()
            .unstack(fill_value=0)
            .reindex(range(5, 23), fill_value=0)
        )

        hourly_propagation_pct = hourly_propagation.div(
            hourly_propagation.sum(axis=1).replace(0, np.nan),
            axis=0
        ) * 100

        ax = hourly_propagation_pct.plot(
            kind='bar',
            stacked=True,
            ax=axes[1, col]
        )

        for container in ax.containers:
            labels = [
                f'{v:.1f}%' if v >= 3 else ''
                for v in container.datavalues
            ]
            ax.bar_label(
                container,
                labels=labels,
                label_type='center',
                fontsize=7
            )

        ax.set_xlabel('Scheduled departure hour')
        ax.set_ylabel('Percentage of trains (%)')
        ax.set_title(f'Delay Propagation by Departure Hour ({threshold}-minute threshold)')
        ax.set_ylim(0, 100)
        ax.legend(title='Delay change')

    plt.tight_layout()
    plt.show()


# ----------------------------------- #
# ---- 4.6 - Heatmap correlation ---- #
# ----------------------------------- #
    # General correlation
    weekly_data = train.groupby('week').agg(
        arrival_delayed=('arr_delay', lambda x: (x > 3).mean() * 100),
        departure_delayed=('dep_delay', lambda x: (x > 3).mean() * 100),
        cancellation_rate=('cancelled', 'mean'),
        scheduled_count=('train_id', 'count')
    )

    weekly_data['cancellation_rate'] *= 100

    sns.heatmap(
        weekly_data.corr(),
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        fmt='.2f'
    )

    plt.title('Correlation Between Weekly Delay and Cancellation Rates')
    plt.tight_layout()
    plt.show()

    # Dwell time and delay absorption correlation of late arrivals
    delay_comparison['dwell_time'] = (
        pd.to_datetime(delay_comparison['scheduled_departure'], format='%H:%M')
        - pd.to_datetime(delay_comparison['scheduled_arrival'], format='%H:%M')
    ).dt.total_seconds() / 60

    delay_comparison['dwell_time'] = delay_comparison['dwell_time'] % 1440

    # Dwell time intervals
    dwell_bins = [0, 2, 3, 4, 5, 6, 8, 10, 15, np.inf]
    dwell_labels = ['0-1', '2', '3', '4', '5', '6-7', '8-9', '10-14', '15+']

    delay_comparison['dwell_bin'] = pd.cut(
        delay_comparison['dwell_time'],
        bins=dwell_bins,
        labels=dwell_labels,
        right=False
    )

    # Settings
    min_trains = 15
    min_arr_delay = 60

    # Only late-arriving trains
    late_arrivals = delay_comparison[delay_comparison['arr_delay'] > 0]

    grouped_late = late_arrivals.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    # Delay absorption (%)
    grouped_late['absorption'] = (
        (grouped_late['arr_sum'] - grouped_late['dep_sum'])
        / grouped_late['arr_sum'] * 100
    ).where(
        (grouped_late['arr_sum'] >= min_arr_delay)
        & (grouped_late['n_trains'] >= min_trains)
    )

    # Prepare heatmap data
    heatmap_late = grouped_late['absorption'].unstack('dwell_bin').reindex(
        index=range(5, 23),
        columns=dwell_labels
    )

    # Finer colour intervals around zero
    boundaries = [-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]

    cmap = plt.get_cmap('RdBu', len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        heatmap_late,
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

    # Grey cells without sufficient data
    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel('Departure hour')
    ax.set_title(
        'Delay absorption of late-arriving trains by scheduled dwell time '
        'and departure hour'
    )

    plt.tight_layout()
    plt.show()


    # Correlation of dwell time and all delayed trains
    min_trains = 15
    row_var = 'dep_hour'
    row_labels = {
        'dep_hour': 'Departure hour',
        'Type': 'Train type',
        'weekday': 'Weekday'
    }

    # Group all trains by departure hour and scheduled dwell time
    grouped_all = delay_comparison.groupby(
        ['dep_hour', 'dwell_bin'], observed=False
    ).agg(
        arr_sum=('arr_delay', 'sum'),
        dep_sum=('dep_delay', 'sum'),
        n_trains=('train_id', 'count')
    )

    # Net delay change in minutes per train
    # Positive = delay accumulation; negative = delay recovery
    grouped_all['net_delay_change'] = (
        (grouped_all['dep_sum'] - grouped_all['arr_sum'])
        / grouped_all['n_trains']
    ).where(grouped_all['n_trains'] >= min_trains)

    # Prepare heatmap data
    heatmap_all = grouped_all['net_delay_change'].unstack('dwell_bin')

    if row_var == 'dep_hour':
        heatmap_all = heatmap_all.reindex(index=range(5, 23))

    heatmap_all = heatmap_all.reindex(columns=dwell_labels)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        heatmap_all,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Net delay change (minutes per train)'},
        ax=ax
    )

    ax.set_facecolor('lightgrey')
    ax.set_xlabel('Scheduled dwell time at Utrecht Centraal (minutes)')
    ax.set_ylabel(row_labels[row_var])
    ax.set_title(
        'Net delay change per train by scheduled dwell time and '
        + row_labels[row_var].lower()
    )

    plt.tight_layout()
    plt.show()

    results = {
        'Vehicles': total_vehicles,
        'Days': total_days,
        'Delayed vehicles (%)': delayed_trains,
        'Delayed operations (%)': delayed_operations,
        'Delay per vehicle (min)': delay_per_vehicle,
        'Delay per day (min)': delay_per_day,
        'Cancellations per 100 vehicles': cancellations_per_100_vehicles,
        'Cancellations per day': cancellations_per_day
    }

    if verbose:
        print(f"DELAYS AND CANCELLATIONS BETWEEN: {start_period} AND {end_period}")
        print(f"Number of complete cancellations: {total_cancellations}")
        print(f"    {delayed_trains:.1f}% of trains were delayed.")
        print(f"    {delayed_operations:.1f}% of train operations were delayed by more than 3 minutes.")
        print(f"    {total_cancellations_perct:.1f}% of trains were cancelled.")

        print("\nRELATIVE PERFORMANCE")
        print(f"    Delay per vehicle: {delay_per_vehicle:.2f} minutes")
        print(f"    Delay per day: {delay_per_day:.1f} minutes")
        print(f"    Cancellations per 100 vehicles: {cancellations_per_100_vehicles:.2f}")
        print(f"    Cancellations per day: {cancellations_per_day:.2f}")

    return results
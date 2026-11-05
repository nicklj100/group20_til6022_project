
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyse_delays_per_hour(df_train):

    # Copying the train data
    df_hour = df_train.copy()

    # Combining the time (scheduled arrival) and date
    df_hour['Start_Datetime'] = pd.to_datetime(
        df_hour['Date'].astype(str) + ' ' + df_hour['scheduled_arrival']
    )

    # Rounding hours down to the nearest whole hour
    df_hour['Hour'] = df_hour['Start_Datetime'].dt.floor('h')

    # Taking the total trains, cancelled and delayed trains each hour
    hourly = df_hour.groupby('Hour').agg(
        trains=('Type', 'size'),
        cancelled=('cancelled', 'sum'),
        arrival_delayed=('arrival_delayed', 'sum'),
        departure_delayed=('departure_delayed', 'sum')
    )

    # Setting up the time period based on the data
    time_period = pd.date_range(
        start=df_hour['Start_Datetime'].min().normalize(),
        end=df_hour['Start_Datetime'].max().normalize() + pd.Timedelta(hours=23),
        freq='h'
    )

    # Reindexing the trains to each hour
    hourly = hourly.reindex(time_period, fill_value=0)

    return hourly


# Making a function to convert to percentage
def total_to_percentage(input, total):
    return np.where(total == 0, 0, 100 * input / total)


def cancellation_chart(df_train, df_weather):

    # Analysing train data per hour
    hourly = analyse_delays_per_hour(df_train)

    # Visualisation total cancelled and delayed trains each hour
    fig1, ax1 = plt.subplots(figsize=(20, 8))

    # Plotting weather data
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        df_weather['datetime'], df_weather['T_C'],
        label='Temperature (°C)', color='tab:blue', linewidth=3.0
    )

    snow_df = df_weather[df_weather['S'] == 1]
    scatter1 = ax1.scatter(
        snow_df['datetime'], snow_df['T_C'],
        color='tab:red', s=60, zorder=5, label='Snow (S=1)'
    )

    # Setting up train data
    bar_width = pd.Timedelta(hours=1)

    bar1 = ax2.bar(
        hourly.index, hourly['cancelled'],
        color='crimson', edgecolor='black',
        width=bar_width, alpha=0.7, label='Cancelled trains'
    )

    bar2 = ax2.bar(
        hourly.index, hourly['arrival_delayed'],
        color='blue', edgecolor='black',
        width=bar_width, label='Arrival delayed'
    )

    bar3 = ax2.bar(
        hourly.index, hourly['departure_delayed'],
        color='green', edgecolor='black',
        width=bar_width, label='Departure delayed'
    )

    # Plot formatting
    plt.title('Total cancelled trains each hour', fontsize=14, fontweight='bold')
    plt.xlabel('Date and time', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Total cancellations and delays', fontsize=12)

    # Setting up legend
    lines = line1 + [scatter1, bar1, bar2, bar3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def cancellation_snow_chart(df_train, df_weather):

    # Analysing train data per hour
    hourly = analyse_delays_per_hour(df_train)

    # Visualisation cancelled trains
    fig1, ax1 = plt.subplots(figsize=(20, 8))

    # Plotting weather data
    line1 = ax1.plot(
        df_weather['datetime'], df_weather['T_C'],
        label='Temperature', color='tab:blue', linewidth=3.0
    )

    snow_df = df_weather[df_weather['S'] == 1]
    scatter1 = ax1.scatter(
        snow_df['datetime'], snow_df['T_C'],
        color='black', s=60, zorder=5, label='Snow (S=1)'
    )

    # Setting up the train data
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(hours=1)

    bar1 = ax2.bar(
        hourly.index,
        total_to_percentage(hourly['cancelled'], hourly['trains']),
        label='Cancelled trains',
        color='tab:red', edgecolor='black',
        width=bar_width, alpha=0.7
    )

    # Setting up labels and titles
    ax1.set_title('% of cancelled trains each hour during snowstorm', fontsize=14, fontweight='bold')
    plt.xlabel('Date and time', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('% cancelled', fontsize=12)

    # Setting up legend
    lines = line1 + [scatter1, bar1]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def punctuality_chart(df_train, df_weather):

    # Analysing train data per hour
    hourly = analyse_delays_per_hour(df_train)

    # Visualisation punctuality
    fig1, ax1 = plt.subplots(figsize=(20, 8))

    # Plotting weather data
    line1 = ax1.plot(
        df_weather['datetime'], df_weather['T_C'],
        label='Temperature', color='tab:blue', linewidth=3.0
    )

    snow_df = df_weather[df_weather['S'] == 1]
    scatter1 = ax1.scatter(
        snow_df['datetime'], snow_df['T_C'],
        color='black', s=60, zorder=5, label='Snow (S=1)'
    )

    # Setting up the train data
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(hours=1)

    # Excluding cancelled trains
    running = df_train[~df_train['cancelled'].fillna(False)].copy()

    # Combining the scheduled arrival time and date
    running['Start_Datetime'] = pd.to_datetime(
        running['Date'].astype(str) + ' ' + running['scheduled_arrival']
    )

    # Grouping trains by hour
    hourly_running = running.groupby(
        running['Start_Datetime'].dt.floor('h')
    )

    # Calculating arrival punctuality (< 180 seconds)
    hourly_punctuality = (
        hourly_running['arr_delay']
        .apply(lambda x: (x < 3).mean() * 100)
    )

    # Reindexing to include hours without trains
    hourly_punctuality = hourly_punctuality.reindex(hourly.index, fill_value=0)

    bar1 = ax2.bar(
        hourly.index, hourly_punctuality,
        label='Arrival punctuality',
        color='tab:red', edgecolor='black',
        width=bar_width, alpha=0.7
    )

    # Setting up labels and titles
    ax1.set_title('Arrival punctuality each hour during snowstorm', fontsize=14, fontweight='bold')
    plt.xlabel('Date and time', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Punctuality (%)', fontsize=12)

    # Setting up legend
    lines = line1 + [scatter1, bar1]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

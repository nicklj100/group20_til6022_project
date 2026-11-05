import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_weather_data(data='./data/weather_selected.csv',
                     start_date='2026-01-01', end_date='2026-01-14'):

    # 1. Import the weather data from the CSV file
    df = pd.read_csv(data)

    # 2. Process Dates and Hours
    # KNMI hourly data uses hours 1 to 24. We convert hour 24 properly by adjusting HH - 1.
    df['HH_adjusted'] = df['HH'] - 1
    df['datetime'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['HH_adjusted'], unit='h')

    # Convert KNMI scaled values to standard units:
    # T: Temperature in 0.1 °C -> divide by 10
    # TD: Dew point temperature in 0.1 °C -> divide by 10
    df['T_C'] = df['T'] / 10.0
    df['TD_C'] = df['TD'] / 10.0

    # Convert Wind Speed in 0.1 m/s to km/hr
    df['FF'] = df['FF'] * (0.36)

    # RH: Hourly precipitation amount in 0.1 mm (-1 means <0.05 mm trace amount)
    # We convert -1 to 0.0 for numerical plotting, and divide valid values by 10.
    df['RH_mm'] = df['RH'].apply(lambda x: 0.0 if x == -1 else x / 10.0)

    # 3. Filter data for the specified period
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date + ' 23:59:59')
    df_filtered = df.loc[mask].copy()

    # Select columns related to weather variables
    columns_of_interest = ['datetime', 'T_C', 'TD_C', 'FF', 'RH_mm', 'S', 'R', 'WW', 'VV', 'U']
    df_weather = df_filtered[columns_of_interest].dropna(subset=['datetime'])

    print(f"Filtered dataset contains {len(df_weather)} records for the selected period.")

    return df_weather

# 4. Visualizations - Three Separate Figures

def set_format(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

def plot_temp_and_snow_occur(df_weather):

    # --- Figure 1: Temperature and Snow Occurrences ---
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_weather['datetime'], df_weather['T_C'], label='Temperature (°C)', color='tab:blue', linewidth=3.0)

    snow_df = df_weather[df_weather['S'] == 1]
    ax1.scatter(snow_df['datetime'], snow_df['T_C'], color='tab:red', s=60, zorder=5, label='Snow (S=1)')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Temperature and Snow Occurrences')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    set_format(ax1)

    plt.show()

def plot_temp_and_sleet_occur(df_weather):

    # --- Figure 1: Temperature and Sleet Occurrences ---
    fig4, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_weather['datetime'], df_weather['T_C'], label='Temperature (°C)', color='tab:blue', linewidth=3.0)

    sleet_df = df_weather[df_weather['S'] == 1]
    sleet_df = sleet_df[sleet_df['R'] == 1]
    ax1.scatter(sleet_df['datetime'], sleet_df['T_C'], color='tab:red', s=60, zorder=5, label='Sleet (S=1, R=1)')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Temperature and Sleet Occurrences')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    set_format(ax1)

    plt.show()

def plot_temp_vs_dew_point(df_weather):

    # --- Figure 2: Temperature vs. Dew Point Comparison ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_weather['datetime'], df_weather['T_C'], label='Temperature (°C)', color='tab:blue', linewidth=3.0)
    ax2.plot(df_weather['datetime'], df_weather['TD_C'], label='Dew Point (°C)', color='tab:orange', linewidth=3.0, linestyle='--')

    ax2.set_ylabel('Temperature / Dew Point (°C)')
    ax2.set_title('Temperature and Dew Point Comparison')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    set_format(ax2)
    
    plt.show()

def plot_temp_and_precipitation(df_weather):

    # --- Figure 3: Temperature and Precipitation (RH) ---
    fig3, ax3_temp = plt.subplots(figsize=(12, 5))
    ax3_precip = ax3_temp.twinx()  # Secondary Y-axis for precipitation

    ax3_temp.plot(df_weather['datetime'], df_weather['T_C'], label='Temperature (°C)', color='tab:blue', linewidth=3.0)
    ax3_precip.bar(df_weather['datetime'], df_weather['RH_mm'], label='Precipitation (mm)', color='tab:green', alpha=0.5, width=0.03)

    ax3_temp.set_ylabel('Temperature (°C)', color='tab:blue')
    ax3_precip.set_ylabel('Precipitation (mm)', color='tab:green')
    ax3_temp.set_title('Temperature and Precipitation Comparison')
    ax3_temp.grid(True, linestyle='--', alpha=0.6)

    # Combine legends from both axes on Figure 3
    lines_1, labels_1 = ax3_temp.get_legend_handles_labels()
    lines_2, labels_2 = ax3_precip.get_legend_handles_labels()
    ax3_temp.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    set_format(ax3_temp)

    plt.show()

def plot_temp_and_wind_speed(df_weather):
        # --- Figure 2: Temperature vs. Dew Point Comparison ---
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df_weather['datetime'], df_weather['T_C'], label='Temperature (°C)', color='tab:blue', linewidth=3.0)
    ax4.plot(df_weather['datetime'], df_weather['FF'], label='Wind Speed (km/hr)', color='tab:orange', linewidth=3.0, linestyle='--')

    ax4.set_ylabel('Temperature / Wind Speed (km/hr)')
    ax4.set_title('Temperature and Wind Speed Comparison')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper right')

    set_format(ax4)
    
    plt.show()

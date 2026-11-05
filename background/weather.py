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

"""
Weather Patterns around Utrecht Centraal: Baseline and Snowstorm Event

Data was obtained from the KNMI hourly observations, station 260 (De Bilt) and included weather data from 2021 to 2026.

For the typical (baseline) weather at Utrecht Centraal during the research period, KNMI data was filtered on the
dates: January 2nd to January 11th for the years 2021 to 2025.

The following five weather indicators were chosen and analysed:
- Temperature
- Dew point
- Precipitation
- Wind speed
- Snow occurrence

Note that the minimum and maximum value were also displayed along with the average for each of the abovementioned
weather indicators. This more clearly visualizes the range in which typical weather fluctuates.

Figures (each function returns a matplotlib Figure, nothing is shown automatically):
    plot_baseline_weather()    typical weather, baseline years only
    plot_event_vs_baseline()   event year against the baseline
    plot_year_by_year()        yearly indicators, 2021-2026
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Settings -------------------------------------------------------------------------------
WEATHER_CSV = None
STATION = 260                 # KNMI De Bilt
BASELINE_YEARS = list(range(2021, 2026))
EVENT_YEAR = 2026
WINDOW_START, WINDOW_END = (1, 2), (1, 11)     # (month, day), both inclusive

# Thresholds used by yearly_indicators() (not shown in the figures, but needed for the code to run)
FROST_C, WET_MM, HEAVY_MM = 0.0, 0.1, 2.0

# Look
C_BASE, C_EVENT = "gray", "tab:red"
RC_PARAMS = {"figure.dpi": 100, "axes.titlesize": 11, "axes.titleweight": "bold",
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True}


def with_report_style(func):
    """Apply RC_PARAMS only while one of these figures is built, so other figures in the report keep their own style."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC_PARAMS):
            return func(*args, **kwargs)
    return wrapper


# Labels for the days of the window (a non-leap year is used only to build the labels)
N_DAYS = (pd.Timestamp(2001, *WINDOW_END) - pd.Timestamp(2001, *WINDOW_START)).days + 1
DAY_LABELS = [(pd.Timestamp(2001, *WINDOW_START) + pd.Timedelta(days=i)).strftime("%d %b") for i in range(N_DAYS)]
WINDOW_TEXT = f"{DAY_LABELS[0]}–{DAY_LABELS[-1]}"
BASE_TEXT = f"{BASELINE_YEARS[0]}–{BASELINE_YEARS[-1]}"


# ----- 1. Read the KNMI file -------------------------------------------------------------------
def find_file(explicit, names=("weather_total_cleaned.csv", "weather_selected.csv")):
    """Use the given path, or look in the current folder, ./data, ../data and ../../data."""
    if explicit is not None:
        return Path(explicit)
    roots = [Path.cwd()]
    if "__file__" in globals():                       # when imported from the main report, also search next to this file
        roots.append(Path(__file__).resolve().parent)
    for here in roots:
        for folder in [here, here / "data", here.parent / "data", here.parent, here.parent.parent / "data"]:
            for name in names:
                if (folder / name).is_file():
                    return folder / name
    raise FileNotFoundError(f"Weather file not found (running from {Path.cwd()}). Set WEATHER_CSV in the settings.")


raw = pd.read_csv(find_file(WEATHER_CSV), skipinitialspace=True)
raw.columns = raw.columns.str.replace("#", "", regex=False).str.strip()
raw = raw[raw["STN"] == STATION]

# ----- 2. Process dates/hours and convert KNMI units -------------------------------------------
# HH is hour-ending in UTC (1..24), so HH - 1 is the start of the hour.

start_of_day = pd.to_datetime(raw["YYYYMMDD"].astype(str), format="%Y%m%d")
wx = pd.DataFrame(index=pd.DatetimeIndex(start_of_day + pd.to_timedelta(raw["HH"] - 1, unit="h"), name="datetime"))
wx["temp_c"] = raw["T"].to_numpy() / 10                                           # 0.1 degC -> degC
wx["dew_c"] = raw["TD"].to_numpy() / 10                                           # 0.1 degC -> degC
wx["precip_mm"] = np.where(raw["RH"].to_numpy() == -1, 0, raw["RH"].to_numpy() / 10)   # 0.1 mm -> mm, -1 = trace -> 0
wx["wind_kmh"] = raw["FH"].to_numpy() * 0.1 * 3.6                                 # hourly mean wind, 0.1 m/s -> km/h
wx["snow"] = raw["S"].to_numpy() == 1                                             # snow indicator
wx = wx.sort_index()


# ----- 3. Select 2-11 January of every year ---------------------------------------------------
def get_window(year):
    start = pd.Timestamp(year, *WINDOW_START)
    end = pd.Timestamp(year, *WINDOW_END) + pd.Timedelta(days=1)       # exclusive upper bound
    f = wx[(wx.index >= start) & (wx.index < end)].copy()
    f["wday"] = (f.index.normalize() - start).days                     # 0 = first day of the window
    return f


windows = {y: get_window(y) for y in BASELINE_YEARS + [EVENT_YEAR]}


# ----- 4. Weather indicators --------------------------------------------------------------------
def daily_table(f):
    """One row per day: used to compare the days of 2026 with the same calendar days of the baseline."""
    g = f.groupby("wday")
    return pd.DataFrame({
        "temp_mean": g["temp_c"].mean(),
        "dew_mean": g["dew_c"].mean(),
        "precip_mm": g["precip_mm"].sum(),
        "wind_mean": g["wind_kmh"].mean(),
        "snow_hours": g["snow"].sum(),
    })


def yearly_indicators(f):
    """One value per indicator for a whole window (one year)."""
    return pd.Series({
        "temp_mean": f["temp_c"].mean(),
        "temp_min": f["temp_c"].min(),
        "precip_total": f["precip_mm"].sum(),
        "wind_mean": f["wind_kmh"].mean(),
        "snow_hours": f["snow"].sum(),
        "frost_hours": (f["temp_c"] < FROST_C).sum(),
        "wet_hours": (f["precip_mm"] >= WET_MM).sum(),
        "heavy_hours": (f["precip_mm"] >= HEAVY_MM).sum(),
    }, dtype=float)


daily_all = pd.concat({y: daily_table(f) for y, f in windows.items()}, names=["year", "wday"])   # rows: (year, day)
yearly = pd.DataFrame({y: yearly_indicators(f) for y, f in windows.items()}).T                   # rows: year
base_years = yearly.loc[BASELINE_YEARS]                                                          # baseline years only


# ----- Plot helpers -----------------------------------------------------------------------------
def style_days(ax):
    ax.set_xticks(range(N_DAYS))
    ax.set_xticklabels(DAY_LABELS, rotation=60, ha="right", fontsize=8)


def band_panel(ax, metric, title, ylabel, with_event=False, zero_line=False):
    """Daily baseline mean with a min-max band across the baseline years (+ the event year if requested)."""
    piv = daily_all[metric].unstack("year")
    base = piv[BASELINE_YEARS]
    x = base.index
    if not with_event:                                   # thin lines show the individual baseline years
        for y in BASELINE_YEARS:
            ax.plot(x, base[y], color=C_BASE, lw=0.8, alpha=0.6)
    ax.fill_between(x, base.min(axis=1), base.max(axis=1), color=C_BASE, alpha=0.25, label="Baseline range (min-max)")
    ax.plot(x, base.mean(axis=1), color="dimgray", lw=2.5, label=f"Baseline mean {BASE_TEXT}")
    if with_event:
        ax.plot(x, piv[EVENT_YEAR], color=C_EVENT, lw=2.5, marker="o", ms=4, label=str(EVENT_YEAR))
    if zero_line:
        ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    style_days(ax)


def bar_panel(ax, metric, title, ylabel, with_event=False):
    """Daily baseline mean bars (whiskers = min-max across years), with event-year bars next to them if requested."""
    piv = daily_all[metric].unstack("year")
    base = piv[BASELINE_YEARS]
    x = np.asarray(base.index, dtype=float)
    mean = base.mean(axis=1)
    err = [(mean - base.min(axis=1)).to_numpy(), (base.max(axis=1) - mean).to_numpy()]
    w = 0.4 if with_event else 0.7
    shift = w / 2 if with_event else 0
    ax.bar(x - shift, mean, width=w, color="lightgray", edgecolor="gray", yerr=err, capsize=2,
           error_kw={"lw": 0.8, "ecolor": "dimgray"}, label="Baseline mean (whiskers = min-max)")
    if with_event:
        ax.bar(x + shift, piv[EVENT_YEAR], width=w, color=C_EVENT, label=str(EVENT_YEAR))
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    style_days(ax)


# ----- Figure 1: typical weather (baseline only) ---------------------------------------------
def plot_baseline_weather():
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # 1. Temperature
    band_panel(
        axes[0, 0], "temp_mean",
        "Daily mean temperature", "°C",
        zero_line=True
    )

    # 2. Precipitation
    bar_panel(
        axes[0, 1], "precip_mm",
        "Daily precipitation", "mm per day"
    )

    # 3. Dew point
    band_panel(
        axes[1, 0], "dew_mean",
        "Daily mean dew point", "°C",
        zero_line=True
    )

    # 4. Wind speed
    band_panel(
        axes[1, 1], "wind_mean",
        "Daily mean wind speed", "km/h"
    )

    # 5. Snow occurrence
    bar_panel(
        axes[2, 0], "snow_hours",
        "Snow occurrence", "hours with snow"
    )

    # Hide unused sixth panel
    axes[2, 1].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Typical weather, {WINDOW_TEXT}, {BASE_TEXT} (thin lines = individual years)",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()
    return fig

def baseline_daily_statistics():
    """
    Statistics used for Figure 1.
    Returns one row per day.
    """

    metrics = {
        "temp_mean": "Temperature (°C)",
        "dew_mean": "Dew Point (°C)",
        "precip_mm": "Precipitation (mm)",
        "wind_mean": "Wind Speed (km/h)",
        "snow_hours": "Snow Hours"
    }

    tables = []

    for metric, name in metrics.items():

        piv = daily_all[metric].unstack("year")
        base = piv[BASELINE_YEARS]

        df = pd.DataFrame({
            "Day": DAY_LABELS,
            "Indicator": name,
            "Mean": base.mean(axis=1).values,
            "Minimum": base.min(axis=1).values,
            "Maximum": base.max(axis=1).values
        })

        tables.append(df)

    return pd.concat(tables, ignore_index=True)

baseline_stats = baseline_daily_statistics()

# ----- Figure 2: the event year against the baseline -----------------------------------------
def plot_event_vs_baseline():
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # 1. Temperature
    band_panel(
        axes[0, 0], "temp_mean",
        "Daily mean temperature", "°C",
        with_event=True, zero_line=True
    )

    # 2. Precipitation
    bar_panel(
        axes[0, 1], "precip_mm",
        "Daily precipitation", "mm per day",
        with_event=True
    )

    # 3. Dew point
    band_panel(
        axes[1, 0], "dew_mean",
        "Daily mean dew point", "°C",
        with_event=True, zero_line=True
    )

    # 4. Wind speed
    band_panel(
        axes[1, 1], "wind_mean",
        "Daily mean wind speed", "km/h",
        with_event=True
    )

    # 5. Snow occurrence
    bar_panel(
        axes[2, 0], "snow_hours",
        "Snow occurrence", "hours with snow",
        with_event=True
    )

    # Hide unused sixth panel
    axes[2, 1].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{EVENT_YEAR} against the {BASE_TEXT} baseline, {WINDOW_TEXT}",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()
    return fig

def event_vs_baseline_statistics():
    """
    Statistics used for Figure 2.
    """

    metrics = {
        "temp_mean": "Temperature (°C)",
        "dew_mean": "Dew Point (°C)",
        "precip_mm": "Precipitation (mm)",
        "wind_mean": "Wind Speed (km/h)",
        "snow_hours": "Snow Hours"
    }

    tables = []

    for metric, name in metrics.items():

        piv = daily_all[metric].unstack("year")
        base = piv[BASELINE_YEARS]

        baseline_mean = base.mean(axis=1)

        df = pd.DataFrame({
            "Day": DAY_LABELS,
            "Indicator": name,
            "Baseline Mean": baseline_mean.values,
            "Baseline Min": base.min(axis=1).values,
            "Baseline Max": base.max(axis=1).values,
            "2026 Value": piv[EVENT_YEAR].values,
            "Difference": (
                piv[EVENT_YEAR].values -
                baseline_mean.values
            )
        })

        tables.append(df)

    return pd.concat(tables, ignore_index=True)

event_stats = event_vs_baseline_statistics()

# ----- Figure 3: year by year -------------------------------------------------------------------

def plot_baseline_year_by_year():

    yearly_baseline = yearly.loc[BASELINE_YEARS]

    years_all = list(yearly_baseline.index)
    year_labels = [str(y) for y in years_all]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

    for ax, (key, name, unit, dec, dots) in zip(axes.ravel(), PANELS):

        vals = yearly_baseline[key]

        title = f"{name} [{unit}]"

        if dots:
            ax.scatter(
                year_labels,
                vals,
                s=80,
                zorder=3,
                color="gray"
            )

            pad = (vals.max() - vals.min()) * 0.3 + 0.3

            ax.set_ylim(
                vals.min() - pad,
                vals.max() + pad
            )

            for lab, v in zip(year_labels, vals):

                ax.annotate(
                    f"{v:.{dec}f}",
                    (lab, v),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7
                )

        else:

            bars = ax.bar(
                year_labels,
                vals,
                edgecolor="gray",
                color="lightgray"
            )

            for bar, v in zip(bars, vals):

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.{dec}f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7
                )

        # Baseline average line
        ax.axhline(
            yearly_baseline[key].mean(),
            color="dimgray",
            ls="--",
            lw=1.2
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=9
        )

        ax.tick_params(
            axis="x",
            labelsize=8
        )

    axes.ravel()[5].set_visible(False)

    fig.suptitle(
        f"Baseline weather indicators, {WINDOW_TEXT} ({BASE_TEXT})",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()
    return fig

def baseline_yearly_statistics():
    """
    Values shown in the baseline-only Figure 3.
    One row per baseline year.
    """

    baseline_table = yearly.loc[BASELINE_YEARS].copy()

    baseline_table.index.name = "Year"

    return baseline_table.round(2)

# ----- Figure 4: year by year snowstorm -------------------------------------------------------------------
PANELS = [
    ("temp_mean", "Mean temperature", "°C", 1, False),
    ("temp_min", "Lowest temperature", "°C", 1, False),
    ("precip_total", "Total precipitation", "mm", 1, False),
    ("snow_hours", "Snow hours (S = 1)", "h", 0, False),
    ("wind_mean", "Mean wind speed", "km/h", 1, False),
]

def plot_year_by_year():
    years_all = list(yearly.index)
    year_labels = [str(y) for y in years_all]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

    for ax, (key, name, unit, dec, dots) in zip(axes.ravel(), PANELS):
        vals = yearly[key]
        title = f"{name} [{unit}]"

        if dots:
            ax.scatter(year_labels, vals, s=80, zorder=3,
                       color=[C_EVENT if y == EVENT_YEAR else "gray" for y in years_all])
            pad = (vals.max() - vals.min()) * 0.3 + 0.3
            ax.set_ylim(vals.min() - pad, vals.max() + pad)

            for lab, v in zip(year_labels, vals):
                ax.annotate(f"{v:.{dec}f}", (lab, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7)
        else:
            bars = ax.bar(
                year_labels, vals,
                edgecolor="gray",
                color=[C_EVENT if y == EVENT_YEAR else "lightgray" for y in years_all]
            )

            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.{dec}f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7
                )

        ax.axhline(base_years[key].mean(), color="dimgray", ls="--", lw=1.2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    axes.ravel()[5].set_visible(False)

    fig.suptitle(
        f"Year by year, {WINDOW_TEXT} (red = {EVENT_YEAR}, dashed line = baseline mean)",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()
    return fig

def yearly_statistics_table():
    """
    Statistics used for Figure 4.
    """

    table = yearly.copy()

    baseline_mean = base_years.mean()

    table.loc["Baseline Mean"] = baseline_mean

    return table.round(2)

yearly_table = yearly_statistics_table()
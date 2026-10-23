import pandas as pd


def _analyse_period(data, start_period, end_period, filter_outliers=False):
    """Calculate delay and cancellation metrics for one date period, without plots."""
    train = pd.read_csv(data)

    train["Service:Date"] = pd.to_datetime(train["Service:Date"])

    train = train[
        (train["Service:Date"] >= start_period)
        & (train["Service:Date"] <= end_period)
    ].copy()

    train = train.drop(columns=["index", "Service:RDT-ID"])
    train = train.rename(columns={
        "Service:Type": "Type",
        "Service:Date": "Date",
        "Stop:Arrival time": "scheduled_arrival",
        "Stop:Departure time": "scheduled_departure",
        "Stop:Arrival cancelled": "arr_cancelled",
        "Stop:Departure cancelled": "dep_cancelled",
        "Stop:Arrival delay": "arr_delay",
        "Stop:Departure delay": "dep_delay",
    })

    train.columns = (
        train.columns
        .str.replace("Service:", "", regex=False)
        .str.replace("Stop:", "", regex=False)
        .str.replace("RDT-ID", "train_id", regex=False)
    )

    cols_drop = [
        "Completely cancelled", "Partly cancelled", "Maximum delay",
        "Train number", "Station name", "Station code",
        "Platform change", "Planned platform", "Actual platform", "Company",
    ]
    train = train.drop(columns=cols_drop)
    train = train[train["Type"].isin(["Intercity", "Sprinter"])]

    time_cols = ["scheduled_arrival", "scheduled_departure"]
    train[time_cols] = train[time_cols].apply(
        lambda col: pd.to_datetime(col, utc=True).dt.strftime("%H:%M")
    )

    for time, delay in [
        ("arrival", "arr_delay"),
        ("departure", "dep_delay"),
    ]:
        train[f"scheduled_{time}"] = pd.to_datetime(
            train[f"scheduled_{time}"], format="%H:%M"
        )
        train[f"actual_{time}"] = (
            train[f"scheduled_{time}"]
            + pd.to_timedelta(train[delay], unit="m")
        )
        train[f"scheduled_{time}"] = train[f"scheduled_{time}"].dt.strftime("%H:%M")
        train[f"actual_{time}"] = train[f"actual_{time}"].dt.strftime("%H:%M")

    train["arr_hour"] = pd.to_datetime(
        train["actual_arrival"], format="%H:%M", errors="coerce"
    ).dt.hour
    train["dep_hour"] = pd.to_datetime(
        train["actual_departure"], format="%H:%M", errors="coerce"
    ).dt.hour

    train = train[
        train["dep_hour"].between(5, 22)
        & train["arr_hour"].between(5, 22)
    ].copy()

    train["arrival_delayed"] = train["arr_delay"] > 0
    train["departure_delayed"] = train["dep_delay"] > 0
    train["arrival_delayed_3"] = train["arr_delay"] >= 3
    train["departure_delayed_3"] = train["dep_delay"] >= 3
    train["cancelled"] = (
        train["arr_cancelled"].fillna(False)
        | train["dep_cancelled"].fillna(False)
    )

    if filter_outliers:
        for col in ["arr_delay", "dep_delay"]:
            q1 = train[col].quantile(0.25)
            q3 = train[col].quantile(0.75)
            iqr = q3 - q1
            upper_limit = q3 + 1.5 * iqr

            train = train[
                train[col].isna() | (train[col] <= upper_limit)
            ].copy()

    total_vehicles = len(train)
    total_days = train["Date"].dt.normalize().nunique()

    if total_vehicles == 0 or total_days == 0:
        raise ValueError(
            f"No train observations remain for {start_period} to {end_period} "
            "after filtering. Check the dates and source data."
        )

    delayed_trains = (
        train["arrival_delayed"] | train["departure_delayed"]
    ).mean() * 100

    delayed_operations = (
        train["arrival_delayed_3"] | train["departure_delayed_3"]
    ).mean() * 100

    total_cancellations = int(train["cancelled"].sum())

    total_delay_minutes = (
        train["arr_delay"].clip(lower=0).fillna(0).sum()
        + train["dep_delay"].clip(lower=0).fillna(0).sum()
    )

    return {
        "Vehicles": total_vehicles,
        "Days": total_days,
        "Delayed vehicles (%)": delayed_trains,
        "Delayed operations (%)": delayed_operations,
        "Delay per vehicle (min)": total_delay_minutes / total_vehicles,
        "Delay per day (min)": total_delay_minutes / total_days,
        "Cancellations": total_cancellations,
        "Cancellations per 100 vehicles": total_cancellations / total_vehicles * 100,
        "Cancellations per day": total_cancellations / total_days,
    }


def compare_periods(
    data="data/traindata_total.csv",
    total_period=("2025-12-15", "2026-04-30"),
    selected_period=("2026-01-02", "2026-01-12"),
):
    """Compare the normal period with a selected period.

    Outliers are removed only from the total period.
    The selected period remains unfiltered.
    """
    total_results = _analyse_period(
        data, *total_period, filter_outliers=True
    )

    selected_results = _analyse_period(
        data, *selected_period, filter_outliers=False
    )

    return pd.DataFrame({
        "Total period": total_results,
        "Selected period": selected_results,
    })


if __name__ == "__main__":
    comparison = compare_periods()
    print(comparison.round(2))

import pandas as pd
from scipy import stats
import datetime as dt
import argparse as ap
import glob
import os
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ruptures as rpt


def find_anomalies(filename, window_size=7, epsilon=0.1, start_date=None):
    df = pd.read_csv(
        filename, delimiter=r"\t|\s{2,}", header=0, engine="python"
    )
    df.rename(columns={"# Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    if start_date is not None:
        df = df[df["date"] >= start_date]
    # average the duplicates
    df = df.groupby("date").mean().reset_index()
    # loop through all data columns and print the rolling slope
    for col in df.columns[1:]:
        windows = df[col].rolling(window_size)
        slopes = windows.apply(
            lambda x: stats.linregress(range(0, window_size), x)[0]
        )
        anomalies = slopes[abs(slopes) > epsilon]
        if len(anomalies) > 0:
            dates = df["date"][anomalies.index]
            dates = ", ".join([d.strftime("%m/%d/%y") for d in set(dates)])
            print(
                f"Anomalies found in column '{col}' for '{filename}' for these dates: {dates}"
            )


def find_anomalies2(filename, start_date=None):
    df = pd.read_csv(
        filename, delimiter=r"\t|\s{2,}", header=0, engine="python"
    )
    df.rename(columns={"# Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    if start_date is not None:
        df = df[df["date"] >= start_date]
    # average the duplicates
    df = df.groupby("date").mean().reset_index()

    df.set_index("date", inplace=True)

    for col in df.columns[1:]:
        stl = STL(df[col], period=7)
        result = stl.fit()
        print(result)
        # result.plot(["trend"])
        plt.figure(figsize=(10, 4))
        plt.plot(result.trend)
        plt.title(f"Trend Component for {col}")
        plt.xlabel("Date")
        plt.ylabel("Trend")
        plt.show()
        # only plot the trend

        # result.resid.plot.hist()
        # plt.show()


def find_anomalies3(filename, start_date=None):
    df = pd.read_csv(
        filename, delimiter=r"\t|\s{2,}", header=0, engine="python"
    )
    df.rename(columns={"# Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    if start_date is not None:
        df = df[df["date"] >= start_date]
    # average the duplicates
    df = df.groupby("date").mean().reset_index()

    df.set_index("date", inplace=True)

    scaler = StandardScaler()
    model = IsolationForest(contamination=0.05)

    for col in df.columns[1:]:
        # call fit_transform on the column and fit the model
        data = df[col].values.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data)
        result = model.fit(data_scaled)
        print(result)
        anomaly = model.predict(data_scaled)

        fix, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[col], label="Data")
        ax.scatter(
            df.index[anomaly == -1],
            df[col][anomaly == -1],
            color="red",
            label="Anomalies",
        )
        ax.set_title(f"Anomalies in {col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.legend()
        plt.show()


def find_anomalies4(filename, start_date=None):
    df = pd.read_csv(
        filename, delimiter=r"\t|\s{2,}", header=0, engine="python"
    )
    df.rename(columns={"# Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    if start_date is not None:
        df = df[df["date"] >= start_date]
    # average the duplicates
    df = df.groupby("date").mean().reset_index()

    df.set_index("date", inplace=True)

    scaler = StandardScaler()
    model = IsolationForest(contamination=0.05)

    for col in df.columns[1:]:
        # call fit_transform on the column and fit the model
        data = df[col].values.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data)

        algo = rpt.Pelt(model="l2").fit(data_scaled)
        result = algo.predict(pen=10)
        rpt.show.display(data_scaled, result, figsize=(10, 4))
        plt.title(f"Anomalies in {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.show()


def find_anomalies5(filename, start_date=None):
    df = pd.read_csv(
        filename, delimiter=r"\t|\s{2,}", header=0, engine="python"
    )
    df.rename(columns={"# Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    if start_date is not None:
        df = df[df["date"] >= start_date]
    # average the duplicates
    df = df.groupby("date").mean().reset_index()

    df.set_index("date", inplace=True)

    # resource
    # https://www.kaggle.com/code/yejining99/change-point-detection#Linearly-penalized-segmentation-(Pelt)
    # https://neptune.ai/blog/anomaly-detection-in-time-series
    # https://blog.jetbrains.com/pycharm/2025/01/anomaly-detection-in-time-series/

    scaler = StandardScaler()
    if_model = IsolationForest(contamination=0.05)
    pelt_model = rpt.Pelt(model="l2")

    for col in df.columns[1:]:
        data = df[col].values.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data)

        # find anomalies using Isolation Forest
        if_model.fit(data_scaled)
        anomalies = if_model.predict(data_scaled)

        # find trend changes using Pelt
        trend_changes = pelt_model.fit(data_scaled).predict(pen=10)

        fix, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[col], label="Data")
        ax.scatter(
            df.index[anomalies == -1],
            df[col][anomalies == -1],
            color="green",
            label="Anomalies",
        )

        for i in range(len(trend_changes) - 1):
            start_idx = trend_changes[i] if i == 0 else trend_changes[i - 1]
            end_idx = trend_changes[i]
            ax.axvspan(
                df.index[start_idx],
                df.index[min(end_idx - 1, len(df.index) - 1)],
                alpha=0.2,
                color="red",
                label="Trend Change" if i == 0 else None,
            )

        ax.set_title(f"Anomalies in {col}")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.legend()
        plt.show()


def main():
    a = ap.ArgumentParser(
        description="Find anomalies in a data file",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    a.add_argument(
        "files", nargs="*", help="Directory containing the data files"
    )
    a.add_argument(
        "--window-size", help="Size of the rolling window", type=int, default=7
    )
    a.add_argument(
        "--epsilon",
        help="Threshold for anomaly detection",
        type=float,
        default=0.1,
    )
    a.add_argument("--no-recurse", action="store_true", default=False)

    def valid_date(s):
        try:
            return dt.datetime.strptime(s, "%m/%d/%y")
        except ValueError:
            msg = f"Not a valid date: '{s}'."
            raise ap.ArgumentTypeError(msg)

    a.add_argument(
        "--start-date",
        help="Start date for analysis - 'MM/DD/YY'",
        type=valid_date,
        default=None,
    )
    args = a.parse_args()

    files = []
    for f in args.files:
        if os.path.isdir(f):
            if args.no_recurse:
                files.extend(glob.glob(f + "/*.dat", recursive=False))
            else:
                files.extend(glob.glob(f + "/**/*.dat", recursive=True))
        elif os.path.isfile(f):
            files.append(f)
        else:
            print(f"File {f} not found")

    for f in files:
        try:
            # find_anomalies(
            #     f,
            #     window_size=args.window_size,
            #     epsilon=args.epsilon,
            #     start_date=args.start_date,
            # )
            find_anomalies5(f, start_date=args.start_date)
        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == "__main__":
    main()

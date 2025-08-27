import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
from pathlib import Path

def find_anomalies(filename: Path, start_date=None):
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

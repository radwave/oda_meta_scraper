import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("valid_oda_headers.csv")

target = "DIAG_TRAPPIST1"
target_df = df[df["object"] == target]

for row in target_df.itertuples():
    # plot the row in time and frequency
    plt.plot(
        [
            row.center_frequency - row.bandwidth/2,
            row.center_frequency + row.bandwidth/2,
            row.center_frequency + row.bandwidth/2,
            row.center_frequency - row.bandwidth/2,
            row.center_frequency - row.bandwidth/2,
         ],
        [
            row.start_utc,
            row.start_utc,
            row.start_utc + row.duration,
            row.start_utc + row.duration,
            row.start_utc,
        ],
    )
plt.xlabel("Frequency (MHz)")
plt.ylabel("Time (UTC)")
plt.title(f"Observations of {target}")
plt.show()

breakpoint()
pass

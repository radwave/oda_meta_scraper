import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("valid_oda_headers.csv")

df["host"] = df["download_link"].apply(lambda x: x.split("/")[2])

host_groups = df.groupby("host")
host_sizes = host_groups.agg({"size_bytes": "sum"})
# convert bytes to TB
host_sizes["size_bytes"] /= 1e12
host_sizes.rename(columns={"size_bytes": "size_TB"}, inplace=True)
# plot a bar chart of the sizes, with the size written on top of each bar
plt.bar(host_sizes.index, host_sizes["size_TB"])
for i, size in enumerate(host_sizes["size_TB"]):
    plt.text(i, size, f"{size:.2f}", ha="center", va="bottom")
plt.xticks(rotation=20, ha="right")
plt.ylabel("Total size (TB)")
plt.title("Total size of data downloaded per host")
plt.tight_layout()
plt.show()
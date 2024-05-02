import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.signal import spectrogram, welch


def norm_to_readable(val):
    """Converts a number to a human-readable string"""
    aval = abs(val)
    if aval < 1e-6:
        return f"{val*1e9:.1f} n"
    if aval < 1e-3:
        return f"{val*1e6:.1f} $\mu$"
    if aval < 1:
        return f"{val*1e3:.1f} m"
    if aval < 1e3:
        return f"{val:.1f} "
    if aval < 1e6:
        return f"{val*1e-3:.1f} k"
    if aval < 1e9:
        return f"{val*1e-6:.1f} M"
    raise NotImplementedError("Value too large")


# Load the data
df = pd.read_csv("valid_oda_headers.csv")

target = "TIC159107668"
df = df[df["object"] == target]
freq_mask = np.isclose(df["center_frequency"],  845, atol=1, rtol=0)
time_mask = np.isclose(df["start_utc"], 1599800795.877907, atol=0.5, rtol=0)
rows = df[freq_mask & time_mask].reset_index(drop=True)
if len(rows) != 1:
    raise RuntimeError(f"{len(rows)} matching files, but was only expecting 1")
metadata = rows.loc[0]

# Determine where data starts
if metadata["DIRECTIO"]:
    # if DIRECTIO is enabled, data is aligned to blocks of 512 bytes.
    # round end of header up to next 512-byte boundary
    header_end = int(metadata["header_len"])
    start_byte = ((header_end + 512) // 512) * 512
else:
    # if DIRECTIO is disabled, data starts immediately after header
    start_byte = int(metadata["header_len"])

# The data is baseband GUPPI files is multi-channel and dual-polarization.
# There are OBSNCHAN channels. Each channel has two polarizations.
# Each sample of each polarization of each channel is a complex value, where
# the real and imaginary parts each have NBITS of resolution.
# There are BLOCSIZE bytes of samples between each header. Within a block,
# The samples are ordered with all the channel 0 samples first, then channel 1,
# etc. Within a channel, samples are ordered as
#   * real part of polarization 0
#   * imag part of polarization 0
#   * real part of polarization 1
#   * imag part of polarization 1
# and that pattern is repeated through that channel of that block
num_channels = int(metadata["OBSNCHAN"])
assert metadata["NPOL"] == 4  # NPOL == 4 means dual polarization
block_size = int(metadata["BLOCSIZE"])
bits_per_sample = int(metadata["NBITS"])
if bits_per_sample != 8:
    raise NotImplementedError("Only NBITS=8 is currently supported")
samp_rate = 1/metadata["TBIN"]
num_blocks = metadata["size_bytes"] / (start_byte + block_size)

# let's read one full block
print(f"Downloading block 1 of {num_blocks}. This is {block_size>>20} MB starting at byte {start_byte}")
url = metadata["download_link"]
response = requests.get(
    url,
    headers={
        "Range": f"bytes={start_byte}-{start_byte + block_size - 1}"
    }
)
if response.status_code != 206:
    raise RuntimeError(response.content)
data = np.frombuffer(response.content, dtype=np.int8)
# convert to complex64
data = data.astype(np.float32).view(np.complex64)
# reshape into a 3D matrix with dimensions
# channel x samples x polarizations
data = data.reshape((num_channels, -1, 2))
# Now let's look at the spectrogram
window_sizes = [2**8, 2**10, 2**16]
data_subset = 2**18
assert data_subset <= data.shape[1]
psd_fig, psd_axs = plt.subplots(len(window_sizes), 1, sharex=True)
spec_fig, spec_axs = plt.subplots(len(window_sizes), 1, sharex=True)
psd_fig.subplots_adjust(hspace=0.1)
spec_fig.subplots_adjust(hspace=0.1)
for ind, window_size in enumerate(window_sizes):
    Sxx = np.zeros((
        data_subset// window_size,
        window_size,
    ))
    Pxx = np.zeros((
        window_size,
    ))
    freq_bin_size = metadata["OBSBW"] / (num_channels * window_size)
    single_chan_freqs = np.arange(-window_size // 2, window_size // 2) * np.abs(freq_bin_size)
    freqs = np.zeros(num_channels * window_size)
    for chan in range(num_channels):
        freqs[chan*window_size:(chan+1)*window_size] = (
            metadata["OBSFREQ"]
            - (num_channels-1)/(2*num_channels)*metadata["OBSBW"]
            + chan * metadata["OBSBW"] / num_channels
            + single_chan_freqs
        )

    chan = 21
    for pol in range(2):
        f, t, sxx = spectrogram(
            data[chan, 0:data_subset, pol],
            fs=samp_rate,
            nperseg=window_size,
            noverlap=0,
            return_onesided=False,
            detrend=False,
        )
        time_bin_size = t[1] - t[0]
        sxx = 10*np.log10(np.fft.fftshift(sxx, axes=0).transpose())
        Sxx += sxx
        f, pxx = welch(
            data[chan, 0:data_subset, pol],
            fs=samp_rate,
            nperseg=window_size,
            noverlap=0,
            return_onesided=False,
            detrend=False,
        )
        pxx = 10*np.log10(np.fft.fftshift(pxx))
        Pxx += pxx
    sm = spec_axs[ind].imshow(
        Sxx,
        aspect="auto",
        interpolation="none",
        extent=[
            freqs[chan*window_size]-abs(freq_bin_size)*0.5,
            freqs[(chan+1)*window_size-1] + 0.5*abs(freq_bin_size),
            t[-1] + time_bin_size,
            t[0]
        ],
        vmin=-180,
        vmax=-100,
    )
    spec_fig.colorbar(sm, ax=spec_axs[ind], label="Power (dB)")
    spec_axs[ind].set_ylabel("Time (seconds)")

    text_str = (
        "Window Size=" + format(window_size, ",") + f" samples\n" +
        f"Frequency Resolution={norm_to_readable(abs(freq_bin_size))}Hz\n" +
        f"Time Resolution={norm_to_readable(time_bin_size)}secs"
    )
    spec_axs[ind].text(
        0.05, 0.95, text_str, 
        transform=spec_axs[ind].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor='white', alpha=0.5)
    )
    psd_axs[ind].plot(freqs[chan*window_size:(chan+1)*window_size], Pxx)
    psd_axs[ind].set_ylabel("Time (seconds)")
    psd_axs[ind].text(
        0.05, 0.95, text_str,
        transform=psd_axs[ind].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor='white', alpha=0.5)
    )

spec_fig.suptitle("Spectrogram")
psd_fig.suptitle("Power Spectrum")
psd_axs[-1].set_xlabel("Frequency (MHz)")
spec_axs[-1].set_xlabel("Frequency (MHz)")
plt.show()
    
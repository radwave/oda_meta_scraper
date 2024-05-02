import datetime
from tqdm import tqdm
import pandas as pd

MJD_EPOCH = datetime.datetime(1858, 11, 17, 0, 0, 0)
def mjd_to_utc_timestamp(mjd_day, mjd_sec, mjd_frac):
    mjd = mjd_day + mjd_sec / 86400 + mjd_frac / 86400
    delta = datetime.timedelta(days=mjd)
    dt = MJD_EPOCH + delta
    return dt.timestamp()

def validate_header(row):
    name = row["name"]
    # now we can validate the header
    if row["STTVALID"] == 0:
        raise ValueError(f"{name} has invalid start time")
    # Make sure we have 4 polarizations
    if row["NPOL"] != 4:
        raise ValueError(f"{name} has {row['NPOL']} polarizations")
    # Make sure we have 2 or 8 bits per sample
    if row["NBITS"] not in [2, 8]:
        raise ValueError(f"{name} has {row['NBITS']} bits per sample")
    # Make sure we have raw mode
    if row["OBS_MODE"] != "RAW":
        raise ValueError(f"{name} has {row['OBS_MODE']} observing mode")
    # make sure chan_bw and obsbw are the same sign
    if row["CHAN_BW"] * row["OBSBW"] < 0:
        raise ValueError(f"{name} has different signs for chan_bw and obsbw")
    # make sure tbin and chan_bw are equivalent
    chan_bw = abs(row["CHAN_BW"] * 1e6)
    if abs(row["TBIN"] - 1/chan_bw) > 1e-6:
        raise ValueError(f"{name} has different tbin and chan_bw: {row['TBIN']} != {chan_bw}")
    if row["OBS_MODE"] != "RAW":
        raise ValueError(f"{name} has {row['OBS_MODE']} observing mode")


def main():
    print("Loading headers.csv file...")
    df = pd.read_csv("oda_headers.csv")
    for index, row in tqdm(df.iterrows()):
        header_len = row["header_len"]
        if header_len <= 0:
            df.loc[index, "header_errs"] = "Invalid header length"
            continue
        validate_header(row)

        if row["DIRECTIO"] == 1:
            # when the DIRECTIO flag is set, the header is a multiple of 512 bytes
            # so we need to update header_end to be the next multiple of 512
            header_len = (header_len + 511) // 512 * 512
        block_size = row["BLOCSIZE"]
        block_header_size = header_len + block_size

        if row["size_bytes"] % block_header_size != 0:
            df.loc[index, "header_errs"] = "Invalid size"
            continue

        num_blocks = row["size_bytes"] // block_header_size
        sample_rate = abs(row["OBSBW"])*1e6
        samples_per_block = block_size / (row["NBITS"] / 8 * row["NPOL"])
        if samples_per_block % 1 != 0:
            df.loc[index, "header_errs"] = "Non-integer samples per block"
            continue
        samples_per_block = int(samples_per_block)
        block_duration = samples_per_block / sample_rate
        duration = block_duration * num_blocks
        df.loc[index, "duration"] = duration

        if row["PKTFMT"] != "1SFA":
            df.loc[index, "header_errs"] = "Invalid packet format"
            continue
        pkt_factor = 8 / row["NBITS"]
        start_byte_offset = row["PKTIDX"] * row["PKTSIZE"] / pkt_factor
        start_sample_offset = start_byte_offset / (
            row["NBITS"] / 8 * row["NPOL"])
        start_time_offset = start_sample_offset / sample_rate
        df.loc[index, "start_utc"] = mjd_to_utc_timestamp(
            row["STT_IMJD"],
            row["STT_SMJD"],
            row["STT_OFFS"]) + start_time_offset
        name = row["name"]
        tokens = name.split("_")
        try:
            last_tokens = tokens[-1].split(".")
            scan_num = last_tokens[0]
            df.loc[index, "bandwidth"] = abs(row["OBSBW"])
            df.loc[index, "channel_order"] = "ascending" if row["OBSBW"] > 0 else "descending"
            # verify that these keys exist
            row["AZ"]
            row["ZA"]
            row["PKTIDX"]
            row["PKTSIZE"]
            # blc2_2bit_guppi_57403_PSR_J2326+6113B_0001.0002 is missing scan
            df.loc[index, 'scan'] = int(scan_num) if pd.isna(row["SCAN"]) else row["SCAN"]
            row["SCANLEN"]
            # blc02_guppi_58060_20295_DIAG_3C123_0001.0000 is missing scan_remaining
            df.loc[index, 'scan_remaining'] = -1 if pd.isna(row["SCANREM"]) else row["SCANREM"]
            row["TBIN"]
            df.loc[index, "header_errs"] = ""
        except Exception as e:
            raise RuntimeError(f"Error processing {name}: {e}")

    df.to_csv("valid_oda_headers.csv", index=False)

if __name__ == '__main__':
    main()

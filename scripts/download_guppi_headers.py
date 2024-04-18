"""
This script downloads the headers of the GUPPI files from the BTL data csv file.
The headers appear to never exceed the first 6640 bytes of the file.
The headers are saved in the headers directory.
"""
import pandas as pd
import os
import requests
from tqdm import tqdm


def get_header_len(raw):
    # The header of GUPPI files is comprised of string fields.
    # There may be a variables number of fields, but the last
    # field is always 160 bytes long.
    # The last 80 bytes of the header is always:
    # END<77 spaces>
    # Let's verify that we have those last 80 bytes.
    footer = b"END" + b" " * 77
    assert footer in raw
    # remove the footer field from the header
    footer_pos = raw.find(footer)
    header_len = footer_pos + len(footer)
    return header_len


def get_fields(raw):
    # Now's let's split the header into fields
    fields = [raw[i:i+80] for i in range(0, len(raw), 80)]
    # The field delimiter is an equals sign followed by a space
    fields = [field.split("= ") for field in fields]
    # The field name may have spaces in it, so we need to
    # trim the field name
    field_names = [field[0].strip() for field in fields]
    # The field value may have spaces in it, so we need to
    # trim the field value
    field_values = [field[1].strip() for field in fields]
    return field_names, field_values

def get_header_dict(field_names, field_values, name, field_types):
    # now we can create a dictionary of field names and values
    header_dict = dict()
    for i, field in enumerate(field_values):
        # string field are enclosed by single quotes, so we need
        # to remove those
        if field[0] == "'" and field[-1] == "'":
            field_values[i] = field[1:-1]
            header_dict[field_names[i]] = field_values[i].strip()
        # float field have a . in them, so we need to convert
        # them to floats
        elif "." in field:
            field_values[i] = float(field)
            header_dict[field_names[i]] = field_values[i]
            # I don't know if fractional seconds are used. They docs
            # say that fractional seconds are a thing, and I would
            # imagine they would be floats, but I've only seen them
            # as 0 (integer, no '.')
            if field_names[i] == "STT_OFFS":
                if field_values[i] != 0:
                    print(f"{name} has non-zero fractional seconds")
        # integer fields are just integers
        else:
            field_values[i] = int(field)
            header_dict[field_names[i]] = field_values[i]

    # some files have strings for every field, so we need to convert
    # the strings to the appropriate type
    if len(field_types) > 0:
        for field_name, field_value in header_dict.items():
            if field_name in field_types:
                if field_types[field_name] == str:
                    continue
                elif field_types[field_name] == int:
                    header_dict[field_name] = int(float(field_value))
                elif field_types[field_name] == float:
                    header_dict[field_name] = float(field_value)
                else:
                    header_dict[field_name] = field_types[field_name](field_value)

    return header_dict

def main():
    print("Loading csv file...")
    df = pd.read_csv("oda.csv")
    os.makedirs("headers", exist_ok=True)
    df["header_len"] = 0
    df["name"] = df["download_link"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    field_types = dict()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        url = row["download_link"]
        name = row["name"]
        headers = {"Range": "bytes=0-6640"}
        response = requests.get(url, headers=headers)
        if response.status_code != 206:
            print(f"Error downloading {name}: {response.status_code}")
            continue
        content = response.content
        # get the header length
        header_len = get_header_len(response.content)
        df.loc[index, "header_len"] = header_len
        raw = response.content[:header_len-80].decode("utf-8")
        # get the field names and values
        field_names, field_values = get_fields(raw)

        # create a dictionary of field names and values
        header_dict = get_header_dict(field_names, field_values, name, field_types)
        # if the field_types dictionary is empty, then we need to
        # populate it
        if len(field_types) == 0:
            for field_name, field_value in header_dict.items():
                field_types[field_name] = type(field_value)

        # merge the header dictionary with the row
        for key, value in header_dict.items():
            df.loc[index, key] = value

    df.to_csv("oda_headers.csv", index=False) 


if __name__ == '__main__':
    main()

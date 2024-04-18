# Open Data Archive Metadata Scraper

This repo contains scripts to scrape metadata from GUPPI files from the
[Open Data Archive](https://breakthroughinitiatives.org/opendatasearch).

The intent is to better understand what data is available, and help resolve
ambiguities with the files available and the precise times that they correspond
to.

## Installation

You can create a virtual environment and
```
pip install -e .
```
this repo.

## Usage

Scripts should be run in the following order

1. `python scripts/scrape_btl_open_data.py` is hard-coded to scrape the result table of the Open Data Archive,
specifically for the Green Bank Telescope baseband (GUPPI) data. This will generate `oda.csv`.
2. `python scripts/download_guppi_headers.py` will read `oda.csv` and then `GET` the first portion of each file to
obtain the first GUPPI header. The header fields will be merged into the original `oda.csv` data
to output a combined `oda_headers.csv` file
3. `python scripts/validate_guppi.py` will read `oda_headers.csv`, validate the existance of critical fields,
and attempt to determine more precise start times for each GUPPI file. I'm not 100% sure if this is right, but
it's my best guess.
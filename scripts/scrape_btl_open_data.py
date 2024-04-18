"""Scrape the Breakthrough Listen Open Data Archive.

We need to scrape the Breakthrough Listen Open Data Archive to get the
metadata for the data we want to download. This script will scrape the
archive and return a list of dictionaries containing the metadata for
each data file.

At the time of this writing (May 18, 2023), the archive is located at:
https://breakthroughinitiatives.org/opendatasearch

We're currently going to limit ourselves to the data from the Green Bank
Telescope (GBT), and the baseband data. At a later date, we can expand
this to include the data from the Parkes Telescope, and the filterbank
and hdf5 data.

When we use the site directly, we can observe that the site shows a maximum
of 100 results per page, and a maximum of 50 pages. Each search is a GET
request, and the parameters are passed in the URL. When we click on a page,
we see the following URL
https://breakthroughinitiatives.org/opendatasearch?project=GBT&file_type=baseband%2Bdata&search=Search&page=18&perPage=100#results
If we manually change the page number, we can see that there are in fact
more than 50 pages of results. Doing a simple test, I found 455 pages
of results. This means that we need to do 455 GET requests to get all of
the results. We can do this by looping over the page number, and passing
the page number as a parameter in the URL.

Additionally, the source of the data is a table, and it isn't minimized
(there are newlines for each row). This makes it super easy to parse,
where we don't even need to use BeautifulSoup. We can just split the
text on the newlines and use regex to get the data we want.

Update
------
As of March 23, 2024, alpha users have discovered that some files are no longer
in the archive. I confirmed that many Voyager 1 files are no longer present.
I rechecked the archive, and found only 423 pages (100 per page) of files.
I'm going to re-attempt scraping the archive.
"""
import requests
import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, List


def parse_row(row: str) -> Dict:
    for n in range(len(row)):
        # remove the whitespace and <td> </td> tags
        row[n] = row[n].strip().replace('<td>', '').replace('</td>', '')
    # get the data
    data = dict(
        utc_time=row[0],
        mjd_time=row[1],
        telescope=row[2],
        object=row[3],
        right_ascension=row[4],
        declination=row[5],
        center_frequency=row[6],
        file_type=row[7],
        size_bytes=row[8],
        md5_sum=row[9],
        download_link=re.findall(r'href="(.*)"><img', row[10])[0]
    )
    return data

def parse_page(url: str, params: Dict) -> List[Dict]:
    response = requests.get(url, params=params)
    data = response.text.split("\n")
    # go down to the results
    for n, line in enumerate(data):
        if '<h2 id="results">Results</h2>' in line:
            break
    data = data[n+1:]
    start = None
    stop = None
    results = []
    for n, line in enumerate(data):
        if '<tr class="grey">' in line:
            start = n
        elif '</tr>' in line:
            stop = n
            if start is not None:
                results.append(parse_row(data[start+1:stop]))
    return results

def main():
    """Scrape the Breakthrough Listen Open Data Archive."""
    # Get the data from the first page
    url = 'https://breakthroughinitiatives.org/opendatasearch'
    all_results = []
    params = {
        'project': 'GBT',
        'file_type': 'baseband+data',
        'search': 'Search',
        'page': 1,
        'perPage': 100
    }
    pbar = tqdm()
    while True:
        pbar.update(1)
        results = parse_page(url, params)
        if len(results) == 0:
            print(f"Last page: {params['page']}")
            break
        all_results.extend(results)
        params['page'] += 1

    # Write the results to a file
    print(f'Writing {len(all_results)} results to oda.csv')
    df = pd.DataFrame(all_results)
    df.to_csv('oda.csv', index=False)

if __name__ == '__main__':
    main()

import os
import sys
import datetime
import requests
from tqdm import tqdm

API_KEY = "c090d86d"  # ← replace this with your actual key
BASE_URL = "http://www.omdbapi.com/"


def fetch_top50_omdb(start_year):
    current_year = datetime.datetime.now().year
    if start_year > current_year:
        print(f"No data yet for {start_year}—come back later! 😉")
        return

    # Prepare output directory
    dataset_dir = os.path.join(os.path.dirname(__file__), "DataSets")
    os.makedirs(dataset_dir, exist_ok=True)

    for year in tqdm(range(start_year, current_year + 1), desc="Years"):
        movies = []
        # OMDb returns 10 results per page; loop 5 pages max to get 50 titles
        for page in range(1, 6):
            params = {"apikey": API_KEY, "type": "movie", "y": year, "page": page}
            resp = requests.get(BASE_URL, params=params)
            data = resp.json()
            if resp.status_code != 200 or data.get("Response") == "False":
                # stop if no more results or error
                break
            movies.extend(item["Title"] for item in data.get("Search", []))
            if len(data.get("Search", [])) < 10:
                # fewer than 10 means last page
                break

        # Trim to top 50 if more, or whatever count we got
        movies = movies[:50]

        # Write to file
        out_path = os.path.join(dataset_dir, f"OMDb_Top50_{year}.txt")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(f"Top {len(movies)} OMDb Movies of {year}:\n\n")
            for idx, title in enumerate(movies, start=1):
                fout.write(f"{idx}. Movie: {title}\n")

    print("Done fetching via OMDb API! 🚀")


if __name__ == "__main__":
    try:
        start = int(input("Please enter start year (e.g., 2016): "))
    except ValueError:
        print("That’s not a valid year. Exiting!")
        sys.exit(1)
    fetch_top50_omdb(start)

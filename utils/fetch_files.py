import requests
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/hazourahh/big-data-course-2024-projects/refs/heads/master/imdb/"
files_to_fetch = ["train-1.csv", "train-2.csv", "train-3.csv", "train-4.csv", "train-5.csv", "train-6.csv", 
                  "train-7.csv", "train-8.csv", "writing.json", "validation_hidden.csv", "test_hidden.csv", "directing.json"]

def fetch_dataset(data_dir="data"):
    Path("data").mkdir(exist_ok=True)

    for file in files_to_fetch:
        url = f"{SOURCE_URL}/{file}"

        response = requests.get(url)
        response.raise_for_status()

        with open(f"{data_dir}/{file}", "wb") as f:
            f.write(response.content)

        print(f"Downloaded {file}")
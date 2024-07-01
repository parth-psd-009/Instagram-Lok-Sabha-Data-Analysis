import requests
from datetime import datetime, timedelta
import os
import json
import time


token = 'npc8czFQqSa9QjcgBbBcHrnbM5vHYxU3neI4pA4E'

startDate = "2023-10-01"
endDate = datetime.now().strftime("%Y-%m-%d")
count = 100

# Renaming the 'lists' variable to avoid conflict with built-in type
list_items = {
    "Lok_sabha_politicians": "1821293",
    # "Influential_politicians": "1826341",
    # "loksabha": "1826070",
    # "bjp": "1826069",
    # "indiannationalcongress": "1826363"
}

def store_json(new_data, filename):
    try:
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = []
    
    existing_data.extend(new_data)

    with open(filename, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
        print(f"Data appended to {filename} file.")

def fetch_lists(start, end, cnt, list_id, filename):
    url = f"https://api.crowdtangle.com/posts?token={token}&startDate={start}&endDate={end}&sortBy=date&count={cnt}&listIds={list_id}"
    start_time = time.time()
    while time.time() - start_time < 86400:  # Run for 24 hours (7200 seconds)
        response = requests.get(url)
        if response.status_code == 200:
            fetched_posts = response.json()
            print("API data fetched successfully")
            store_json(fetched_posts["result"]["posts"], filename)
            if "nextPage" in fetched_posts["result"]["pagination"]:
                url = fetched_posts["result"]["pagination"]["nextPage"]
        else:
            time.sleep(30)
            

def run_periodic():
    for name, list_id in list_items.items():
        filename = f"{name}.json"
        fetch_lists(startDate, endDate, count, list_id, filename)

run_periodic()


import schedule
import time
from datetime import datetime, timedelta
import os
import json
import requests

token = 'npc8czFQqSa9QjcgBbBcHrnbM5vHYxU3neI4pA4E'

# Define count and list_items
count = 100
list_items = {
    "Lok_sabha_politicians": "1821293",
    "Influential_politicians": "1826341",
    "loksabha": "1826070",
    "bjp": "1826069",
    "indiannationalcongress": "1826363"
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

def is_post_exists(post, filename):
    try:
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
            for existing_post in existing_data:
                if existing_post["id"] == post["id"]:
                    return True
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        pass
    return False

def fetch_lists():
    # Calculate yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Calculate today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    for name, list_id in list_items.items():
        filename = f"{name}.json"
        
        # API request URL
        url = f"https://api.crowdtangle.com/posts?token={token}&startDate={yesterday}&endDate={today}&sortBy=date&count={count}&listIds={list_id}"
        
        response = requests.get(url)
        if response.status_code == 200:
            fetched_posts = response.json()["result"]["posts"]
            print("API data fetched successfully")
            
            # Check if the post already exists in the JSON file
            if not is_post_exists(fetched_posts, filename):
                new_posts = fetched_posts
                store_json(new_posts, filename)
            
            
            # Check for next page
            if "nextPage" in response.json()["result"]["pagination"]:
                next_page_url = response.json()["result"]["pagination"]["nextPage"]
                fetch_lists()
        else:
            print("Failed to fetch data from API.")

# Schedule the function to run every 2 minutes
schedule.every(24).hours.do(fetch_lists)

# Continuously check and run pending tasks
while True:
    schedule.run_pending()
    time.sleep(1)

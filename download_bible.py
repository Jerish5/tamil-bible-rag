import os
import requests
import json

BASE_URL = "https://raw.githubusercontent.com/aruljohn/Bible-tamil/master/"
DATA_DIR = "data"

# List of books (standard English names as used in the repo)
# I'll fetch Books.json first to get the list if possible, or hardcode/discover them.
# The repo has files like Genesis.json, Exodus.json etc.
# Let's try to fetch Books.json first.

def download_file(filename):
    url = BASE_URL + filename
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
        return True
    else:
        print(f"Failed to download {filename}: {response.status_code}")
        return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Downloading Books.json...")
    if download_file("Books.json"):
        with open(os.path.join(DATA_DIR, "Books.json"), "r", encoding="utf-8") as f:
            books_data = json.load(f)
        
        print(f"Found {len(books_data)} books. Downloading individual book files...")
        
        for item in books_data:
            book_name = item.get("book", {}).get("english", "").strip()
            if not book_name:
                continue
                
            # Try different filename variations
            # 1. Exact match with .json
            # 2. Spaces removed
            # 3. Spaces replaced with %20 (requests handles this but the url string needs it? No, requests does encoding)
            
            # The repo seems to use "Genesis.json", "1 Samuel.json" (or "1Samuel.json")
            # Let's try "Name.json" first.
            if download_file(f"{book_name}.json"):
                continue
            
            # Try removing spaces
            book_name_nospace = book_name.replace(" ", "")
            if download_file(f"{book_name_nospace}.json"):
                continue
                
            print(f"Could not download {book_name}")

if __name__ == "__main__":
    main()

import tarfile
import requests
from tqdm import tqdm
import os

with open("links.txt", "r") as f:
    links = f.readlines()

for link in links:
    file_name = link.split(" ")[0]
    url = link.split(" ")[1]
    r = requests.get(url, stream=True)

    print(f"Downloading {file_name}...")
    with open(file_name, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

    print(f"Extracting {file_name}...")
    with tarfile.open(file_name, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            tar.extract(member, path="new_data")

    os.remove(file_name)

files = os.listdir("new_data")
for file in tqdm(files):
    # move images to images folder
    if file.endswith(".jpg"):
        os.rename(f"new_data/{file}", f"images/{file}")
    else:
        os.rename(f"new_data/{file}", f"data/{file}")

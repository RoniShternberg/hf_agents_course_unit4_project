from huggingface_hub import hf_hub_download

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path("env") / ".env")
# remove this later
load_dotenv()

# WRONG: Will likely 404
# hf_hub_download(repo_id="gaia-benchmark/GAIA", filename="7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx", repo_type="dataset")


# RIGHT: Use the full relative path from the dataset metadata
try:
    path_on_disk = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        filename="2023/validation/1f975693-876d-457b-a649-393859e79bf3.mp3", # Example file_path
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    print(f"Success!!! File is at: {path_on_disk}")
except Exception as e:
    print(f"Download failed: {e}")
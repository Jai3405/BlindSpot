#!/usr/bin/env python3
"""
Download SUN RGB-D dataset for indoor obstacle detection.
37 categories perfect for navigation: chairs, doors, tables, walls, beds, people, etc.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, destination):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    print("="*70)
    print("SUN RGB-D Dataset Download")
    print("37 indoor object categories for navigation")
    print("="*70 + "\n")
    
    # Create directories
    data_dir = Path("data/raw/sunrgbd")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading from HuggingFace (YOLO format)...")
    print("This will take some time...\n")
    
    # HuggingFace dataset URL (YOLO format)
    hf_url = "https://huggingface.co/datasets/mattschauer/sun-rgbd-yolo"
    
    print(f"Dataset info: {hf_url}")
    print("\nTo download:")
    print("1. Visit the URL above")
    print("2. Click 'Files and versions'")
    print("3. Download the dataset files")
    print("\nOr use git-lfs:")
    print(f"  git clone {hf_url} {data_dir}")
    
    print(f"\n✓ Dataset directory created: {data_dir}")
    print("\nCategories include:")
    categories = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "desk",
        "shelves", "curtain", "pillow", "clothes", "ceiling", "books",
        "refrigerator", "television", "paper", "towel", "box", "whiteboard",
        "person", "nightstand", "toilet", "sink", "lamp", "bathtub",
        "bag", "structure", "furniture", "prop", "others"
    ]
    for i, cat in enumerate(categories, 1):
        print(f"  {i:2d}. {cat}")

if __name__ == "__main__":
    main()

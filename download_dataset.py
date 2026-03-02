#!/usr/bin/env python3
"""
Script to download the CEDAR signature dataset
"""
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    import numpy as np
    from PIL import Image, ImageDraw
    
    print("\n📝 Creating sample dataset...")
    
    # Create directories
    sample_dir = Path("data/raw/sample_signatures")
    genuine_dir = sample_dir / "genuine"
    forged_dir = sample_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample signatures
    num_users = 5
    signatures_per_user = 10
    forgeries_per_user = 5
    
    for user_id in range(1, num_users + 1):
        # Create genuine signatures
        for sig_num in range(1, signatures_per_user + 1):
            img = Image.new('RGB', (220, 155), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple signature-like pattern
            draw.line([(20, 80), (200, 80)], fill='black', width=3)
            draw.ellipse([(50, 60), (100, 100)], outline='black', width=2)
            draw.text((120, 70), f"User{user_id}", fill='black')
            
            # Add some noise
            pixels = np.array(img)
            noise = np.random.randint(0, 50, pixels.shape, dtype=np.uint8)
            pixels = np.clip(pixels.astype(int) - noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)
            
            filename = genuine_dir / f"user_{user_id:03d}_genuine_{sig_num:03d}.png"
            img.save(filename)
        
        # Create forged signatures
        for sig_num in range(1, forgeries_per_user + 1):
            img = Image.new('RGB', (220, 155), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a slightly different pattern
            draw.line([(20, 85), (200, 75)], fill='black', width=3)
            draw.ellipse([(55, 65), (105, 105)], outline='black', width=2)
            draw.text((115, 75), f"User{user_id}", fill='black')
            
            # Add different noise
            pixels = np.array(img)
            noise = np.random.randint(0, 60, pixels.shape, dtype=np.uint8)
            pixels = np.clip(pixels.astype(int) - noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)
            
            filename = forged_dir / f"user_{user_id:03d}_forged_{sig_num:03d}.png"
            img.save(filename)
    
    print(f"✅ Created sample dataset:")
    print(f"   - {num_users} users")
    print(f"   - {num_users * signatures_per_user} genuine signatures")
    print(f"   - {num_users * forgeries_per_user} forged signatures")
    print(f"   - Location: {sample_dir}")

if __name__ == "__main__":
    create_sample_dataset()

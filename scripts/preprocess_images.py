import os
import cv2
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_single_image(args):
    """
    Function processing a single image:
    1. Loads
    2. Does Center Crop (crops to square)
    3. Resizes
    4. Saves in a new location
    """
    src_path, dst_path, img_size = args

    try:
        if dst_path.exists():
            return

        img = cv2.imread(str(src_path))
        if img is None:
            return 
            
        h, w, _ = img.shape
        shortest_side = min(h, w)
        
        start_y = (h - shortest_side) // 2
        start_x = (w - shortest_side) // 2
        
        img_cropped = img[start_y : start_y + shortest_side, 
                          start_x : start_x + shortest_side]
        
        img_resized = cv2.resize(img_cropped, (img_size, img_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(dst_path), img_resized)

    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing: Crop, Resize, Filter.")
    parser.add_argument("--input", required=True, help="Source directory (e.g. datasets/raw/train)")
    parser.add_argument("--output", required=True, help="Destination directory (e.g. datasets/processed/train)")
    parser.add_argument("--size", type=int, default=112, help="Target image size")
    parser.add_argument("--n_classes", type=int, default=0, help="Number of identities to select (0 = all)")
    parser.add_argument("--max_images", type=int, default=0, help="Max images per identity (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of threads")
    
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    
    print(f"--- CONFIGURATION ---\nInput: {src_root}\nOutput: {dst_root}\nSize: {args.size}x{args.size}")
    
    # 1. Get identities
    all_identities = sorted([d for d in os.listdir(src_root) if (src_root / d).is_dir()])
    print(f"Found {len(all_identities)} identities.")

    # 2. Filter identities
    if args.n_classes > 0 and args.n_classes < len(all_identities):
        random.seed(args.seed)
        selected_identities = random.sample(all_identities, args.n_classes)
        print(f"Selected {len(selected_identities)} identities (Seed: {args.seed})")
    else:
        selected_identities = all_identities
        print("Processing all identities.")

    # 3. Collect files
    tasks = []
    print("Indexing files...")

    for identity in tqdm(selected_identities):
        src_dir = src_root / identity
        dst_dir = dst_root / identity
        
        os.makedirs(dst_dir, exist_ok=True)
        
        images = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if args.max_images > 0:
            images = images[:args.max_images]
            
        for fname in images:
            tasks.append((
                src_dir / fname,
                dst_dir / fname,
                args.size
            ))

    print(f"Total images to process: {len(tasks)}")
    
    # 4. Execute
    if tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
    else:
        print("No images found to process.")

    print("--- DONE ---")

if __name__ == "__main__":
    main()
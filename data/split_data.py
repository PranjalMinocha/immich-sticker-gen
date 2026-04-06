import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_directories(base_output_dir, splits):
    """Creates the folder structure for images and masks."""
    for split in splits:
        os.makedirs(os.path.join(base_output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, split, 'annotations'), exist_ok=True)

def split_and_copy_data(source_dir, output_dir, train_pct=0.7, val_pct=0.15, prod_pct=0.15, seed=42):
    # 1. Identify all unique file base names
    print("Scanning source directory for image/annotation pairs...")
    all_files = os.listdir(source_dir)
    
    image_files = [f for f in all_files if f.endswith('.jpg')]
    valid_bases = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        # SA-1B requires both the image and the json mask
        if f"{base_name}.json" in all_files:
            valid_bases.append(base_name)
            
    print(f"Found {len(valid_bases)} valid image/mask pairs.")

    # 2. Perform the split using the fixed seed
    # First, split off the training set
    train_bases, temp_bases = train_test_split(
        valid_bases, 
        train_size=train_pct, 
        random_state=seed
    )
    
    # Second, split the remaining data into validation and the production reservoir
    relative_val_pct = val_pct / (val_pct + prod_pct)
    val_bases, prod_bases = train_test_split(
        temp_bases, 
        train_size=relative_val_pct, 
        random_state=seed
    )

    splits = {
        'train': train_bases,
        'val': val_bases,
        'production': prod_bases  # Strictly named for your data generator reservoir
    }

    # 3. Create the output folder structure
    create_directories(output_dir, splits.keys())

    # 4. Copy the files over
    for split_name, base_names in splits.items():
        print(f"\nCopying {len(base_names)} files to {split_name}...")
        
        for base in tqdm(base_names, desc=split_name):
            src_img = os.path.join(source_dir, f"{base}.jpg")
            src_mask = os.path.join(source_dir, f"{base}.json")
            
            dest_img = os.path.join(output_dir, split_name, 'images', f"{base}.jpg")
            dest_mask = os.path.join(output_dir, split_name, 'annotations', f"{base}.json")
            
            # Use shutil.move() instead of shutil.copy2() if you want to save disk space
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_mask, dest_mask)

    print("\nDataset split successfully completed!")
    print(f"Train: {len(train_bases)} | Val: {len(val_bases)} | Production Reservoir: {len(prod_bases)}")
    print(f"Seed used: {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split SA-1B data into train/val/production folders.")
    parser.add_argument("--source", required=True, help="Path to the directory with extracted raw SA-1B files")
    parser.add_argument("--output", required=True, help="Path where the split folders will be created")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    split_and_copy_data(args.source, args.output, train_pct=0.7, val_pct=0.15, prod_pct=0.15, seed=args.seed)
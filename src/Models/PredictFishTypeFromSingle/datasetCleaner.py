"""Code to clean the dataset by finding and removing duplicate images"""

import os
import glob
import hashlib
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
data_path = "../../../data/SingleFishes/SingleFishes/finalSFdataset/herring"

# =============================================================================
# Functions
# =============================================================================

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file to identify duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read file in chunks to handle large images efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates_in_directory(directory):
    """Find duplicate images in the specified directory based on file hash.
    
    Args:
        directory: Path to the directory to scan for duplicates
        
    Returns:
        dict: Dictionary mapping hash to list of file paths with that hash
    """
    hash_dict = defaultdict(list)
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    
    print(f"Scanning directory: {directory}")
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    print(f"Found {len(image_files)} image files")
    
    # Calculate hash for each image
    for filepath in image_files:
        try:
            file_hash = calculate_file_hash(filepath)
            hash_dict[file_hash].append(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Filter to only duplicates (hash with more than one file)
    duplicates = {h: files for h, files in hash_dict.items() if len(files) > 1}
    
    return duplicates

def remove_duplicates(duplicates, keep_first=True):
    """Remove duplicate files, keeping only one copy.
    
    Args:
        duplicates: Dictionary from find_duplicates_in_directory
        keep_first: If True, keeps the first file alphabetically; otherwise keeps the last
        
    Returns:
        tuple: (number of files removed, total space freed in bytes)
    """
    removed_count = 0
    space_freed = 0
    
    for file_hash, file_list in duplicates.items():
        # Sort files to have consistent behavior
        sorted_files = sorted(file_list)
        
        # Determine which file to keep
        if keep_first:
            files_to_remove = sorted_files[1:]
            file_to_keep = sorted_files[0]
        else:
            files_to_remove = sorted_files[:-1]
            file_to_keep = sorted_files[-1]
        
        print(f"\nKeeping: {file_to_keep}")
        
        # Remove duplicate files
        for filepath in files_to_remove:
            try:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                removed_count += 1
                space_freed += file_size
                print(f"  Removed: {filepath}")
            except Exception as e:
                print(f"  Error removing {filepath}: {e}")
    
    return removed_count, space_freed

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DUPLICATE IMAGE FINDER AND REMOVER")
    print("=" * 70)
    
    # Convert relative path to absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.normpath(os.path.join(script_dir, data_path))
    
    # Check if directory exists
    if not os.path.exists(absolute_data_path):
        print(f"Error: Directory does not exist: {absolute_data_path}")
        exit(1)
    
    # Find duplicates
    print("\nSearching for duplicates...")
    duplicates = find_duplicates_in_directory(absolute_data_path)
    
    if not duplicates:
        print("\n✓ No duplicates found!")
    else:
        # Display duplicate statistics
        total_duplicates = sum(len(files) - 1 for files in duplicates.values())
        print(f"\n⚠ Found {len(duplicates)} sets of duplicates")
        print(f"⚠ Total duplicate files: {total_duplicates}")
        
        # Show details
        print("\nDuplicate sets:")
        for i, (file_hash, file_list) in enumerate(duplicates.items(), 1):
            print(f"\n  Set {i} ({len(file_list)} identical files):")
            for filepath in sorted(file_list):
                print(f"    - {filepath}")
        
        # Ask for confirmation
        response = input("\nDo you want to remove duplicates? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            removed_count, space_freed = remove_duplicates(duplicates, keep_first=True)
            space_mb = space_freed / (1024 * 1024)
            print("\n" + "=" * 70)
            print(f"✓ Removed {removed_count} duplicate files")
            print(f"✓ Freed {space_mb:.2f} MB of space")
            print("=" * 70)
        else:
            print("\nOperation cancelled. No files were removed.")


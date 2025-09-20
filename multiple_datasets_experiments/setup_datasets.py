"""
Automated dataset setup script for CUB, AWA2, and Oxford Flowers
Handles downloading, extraction, and organization
"""
import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSetup:
    """Automated dataset setup"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.downloads_dir = self.base_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file with progress"""
        filepath = self.downloads_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        logger.info(f"Downloading {filename}...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print()  # New line after progress
            logger.info(f"✅ Downloaded: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def extract_archive(self, filepath: Path, extract_to: Path) -> bool:
        """Extract various archive formats"""
        extract_to.mkdir(exist_ok=True, parents=True)
        
        try:
            if filepath.suffix.lower() == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif filepath.suffix.lower() in ['.tar', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"Unsupported archive format: {filepath}")
                return False
            
            logger.info(f"✅ Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return False
    
    def setup_cub_200_2011(self):
        """Setup CUB-200-2011 dataset"""
        logger.info("🐦 Setting up CUB-200-2011 dataset...")
        
        # CUB dataset info
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        filename = "CUB_200_2011.tgz"
        target_dir = self.base_dir / "CUB_200_2011"
        
        if target_dir.exists() and (target_dir / "images").exists():
            logger.info("✅ CUB-200-2011 already set up")
            return True
        
        try:
            # Download
            archive_path = self.download_file(url, filename)
            
            # Extract
            temp_extract = self.downloads_dir / "CUB_temp"
            if self.extract_archive(archive_path, temp_extract):
                # Move to correct location
                extracted_cub = temp_extract / "CUB_200_2011"
                if extracted_cub.exists():
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.move(str(extracted_cub), str(target_dir))
                    
                    # Cleanup
                    shutil.rmtree(temp_extract)
                    
                    # Verify structure
                    if (target_dir / "images").exists():
                        logger.info("✅ CUB-200-2011 setup complete")
                        return True
                    else:
                        logger.error("❌ CUB dataset structure incorrect")
                        return False
            
        except Exception as e:
            logger.error(f"❌ CUB setup failed: {e}")
            return False
        
        return False
    
    def setup_awa2(self):
        """Setup AWA2 dataset"""
        logger.info("🦁 Setting up AWA2 dataset...")
        
        target_dir = self.base_dir / "AWA2"
        
        if target_dir.exists() and (target_dir / "JPEGImages").exists():
            logger.info("✅ AWA2 already set up")
            return True
        
        # AWA2 requires manual download due to registration
        logger.warning("⚠️  AWA2 requires manual download:")
        print("""
        🔗 Please manually download AWA2:
        1. Go to: https://cvml.ist.ac.at/AwA2/
        2. Register and download AwA2-data.zip
        3. Extract to AWA2/ directory
        4. Ensure structure: AWA2/JPEGImages/antelope/...
        
        Or if you have the file, place AwA2-data.zip in downloads/ and run:
        python setup_datasets.py --extract-awa2
        """)
        
        # Check if zip file exists in downloads
        awa2_zip = self.downloads_dir / "AwA2-data.zip"
        if awa2_zip.exists():
            return self.extract_awa2_manual(awa2_zip)
        
        return False
    
    def extract_awa2_manual(self, zip_path: Path):
        """Extract manually downloaded AWA2 zip"""
        target_dir = self.base_dir / "AWA2"
        
        try:
            temp_extract = self.downloads_dir / "AWA2_temp"
            if self.extract_archive(zip_path, temp_extract):
                # Find JPEGImages directory
                jpeg_dir = None
                for root, dirs, files in os.walk(temp_extract):
                    if "JPEGImages" in dirs:
                        jpeg_dir = Path(root) / "JPEGImages"
                        break
                
                if jpeg_dir and jpeg_dir.exists():
                    target_dir.mkdir(exist_ok=True)
                    target_jpeg = target_dir / "JPEGImages"
                    
                    if target_jpeg.exists():
                        shutil.rmtree(target_jpeg)
                    
                    shutil.move(str(jpeg_dir), str(target_jpeg))
                    shutil.rmtree(temp_extract)
                    
                    logger.info("✅ AWA2 setup complete")
                    return True
                else:
                    logger.error("❌ JPEGImages directory not found in AWA2 archive")
                    return False
        
        except Exception as e:
            logger.error(f"❌ AWA2 extraction failed: {e}")
            return False
        
        return False
    
    def setup_oxford_flowers(self):
        """Setup Oxford-102 Flowers dataset"""
        logger.info("🌸 Setting up Oxford-102 Flowers dataset...")
        
        target_dir = self.base_dir / "oxford-102-flowers"
        
        if target_dir.exists() and (target_dir / "jpg").exists():
            logger.info("✅ Oxford Flowers already set up")
            return True
        
        # Oxford Flowers URLs
        images_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
        
        try:
            target_dir.mkdir(exist_ok=True)
            
            # Download images
            images_archive = self.download_file(images_url, "102flowers.tgz")
            
            # Extract images
            temp_extract = self.downloads_dir / "flowers_temp"
            if self.extract_archive(images_archive, temp_extract):
                # Move jpg directory
                jpg_dir = temp_extract / "jpg"
                target_jpg = target_dir / "jpg"
                
                if jpg_dir.exists():
                    if target_jpg.exists():
                        shutil.rmtree(target_jpg)
                    shutil.move(str(jpg_dir), str(target_jpg))
                
                # Cleanup
                shutil.rmtree(temp_extract)
                
                # Download labels (optional, for organization)
                try:
                    labels_file = self.download_file(labels_url, "imagelabels.mat")
                    shutil.copy(labels_file, target_dir / "imagelabels.mat")
                except:
                    logger.warning("Could not download labels file")
                
                # Organize into class directories
                self.organize_flowers_by_class(target_dir)
                
                logger.info("✅ Oxford Flowers setup complete")
                return True
        
        except Exception as e:
            logger.error(f"❌ Oxford Flowers setup failed: {e}")
            return False
        
        return False
    
    def organize_flowers_by_class(self, flowers_dir: Path):
        """Organize flowers into class directories"""
        jpg_dir = flowers_dir / "jpg"
        
        if not jpg_dir.exists():
            return
        
        try:
            # Try to load labels if scipy is available
            try:
                from scipy.io import loadmat
                labels_file = flowers_dir / "imagelabels.mat"
                
                if labels_file.exists():
                    labels = loadmat(str(labels_file))['labels'][0]
                    
                    # Create class directories
                    for class_id in range(1, 103):  # 102 classes
                        class_dir = jpg_dir / f"class_{class_id:03d}"
                        class_dir.mkdir(exist_ok=True)
                    
                    # Move images to class directories
                    for i, label in enumerate(labels, 1):
                        old_path = jpg_dir / f"image_{i:05d}.jpg"
                        if old_path.exists():
                            new_path = jpg_dir / f"class_{label:03d}" / f"image_{i:05d}.jpg"
                            shutil.move(str(old_path), str(new_path))
                    
                    logger.info("✅ Organized flowers into class directories")
                    return
                    
            except ImportError:
                logger.warning("scipy not available, skipping class organization")
                
        except Exception as e:
            logger.warning(f"Could not organize flowers by class: {e}")
        
        # Fallback: create generic class directories
        for class_id in range(1, 103):
            class_dir = jpg_dir / f"class_{class_id:03d}"
            class_dir.mkdir(exist_ok=True)
        
        logger.info("Created generic class directories for flowers")
    
    def verify_setup(self):
        """Verify all datasets are properly set up"""
        datasets = {
            'CUB-200-2011': self.base_dir / "CUB_200_2011" / "images",
            'AWA2': self.base_dir / "AWA2" / "JPEGImages", 
            'Oxford Flowers': self.base_dir / "oxford-102-flowers" / "jpg"
        }
        
        print("\n🔍 Dataset Verification:")
        print("=" * 50)
        
        all_good = True
        for name, path in datasets.items():
            if path.exists():
                # Count classes
                classes = [d for d in path.iterdir() if d.is_dir()]
                print(f"✅ {name:<20}: {len(classes)} classes found")
            else:
                print(f"❌ {name:<20}: Not found at {path}")
                all_good = False
        
        if all_good:
            print("\n✅ All datasets are ready!")
            print("\n🚀 You can now run:")
            print("   python run_multi_dataset.py --datasets CUB AWA2 FLO")
        else:
            print("\n⚠️  Some datasets are missing. Please set them up manually or run individual setup commands.")
        
        return all_good


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup datasets for novelty detection experiments")
    parser.add_argument("--all", action="store_true", help="Setup all datasets")
    parser.add_argument("--cub", action="store_true", help="Setup CUB-200-2011")
    parser.add_argument("--awa2", action="store_true", help="Setup AWA2")
    parser.add_argument("--flowers", action="store_true", help="Setup Oxford Flowers")
    parser.add_argument("--extract-awa2", action="store_true", help="Extract manually downloaded AWA2")
    parser.add_argument("--verify", action="store_true", help="Verify dataset setup")
    parser.add_argument("--base-dir", default=".", help="Base directory for datasets")
    
    args = parser.parse_args()
    
    setup = DatasetSetup(args.base_dir)
    
    if args.verify or not any([args.all, args.cub, args.awa2, args.flowers, args.extract_awa2]):
        setup.verify_setup()
        return
    
    if args.all:
        print("🚀 Setting up all datasets...")
        setup.setup_cub_200_2011()
        setup.setup_awa2()
        setup.setup_oxford_flowers()
    else:
        if args.cub:
            setup.setup_cub_200_2011()
        if args.awa2 or args.extract_awa2:
            if args.extract_awa2:
                awa2_zip = setup.downloads_dir / "AwA2-data.zip"
                if awa2_zip.exists():
                    setup.extract_awa2_manual(awa2_zip)
                else:
                    print("❌ AwA2-data.zip not found in downloads/")
            else:
                setup.setup_awa2()
        if args.flowers:
            setup.setup_oxford_flowers()
    
    # Final verification
    setup.verify_setup()


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
from PIL import Image
from glob import glob
import random
import torchvision.transforms.functional as TF
import torch.utils.data as data

random.seed(1143)


class MultiExposureDynamicLoader(data.Dataset):
    """
    Load VARIABLE number of exposure images + ground truth
    """
    
    def __init__(self, dataset_path, mode='train', image_size=(1200, 900),
                 min_exposures=1, max_exposures=None, fixed_n=None):
        super(MultiExposureDynamicLoader, self).__init__()
        
        self.dataset_path = dataset_path
        self.mode = mode
        self.image_size = image_size
        self.min_exposures = min_exposures
        self.max_exposures = max_exposures
        self.fixed_n = fixed_n  # If set, always load this many
        
        # Find all image sets
        self.image_sets = self._find_image_sets()
        
        if mode == 'train':
            random.shuffle(self.image_sets)
        
        print(f"[{self.__class__.__name__}] Found {len(self.image_sets)} image sets")
        
        # Statistics
        num_exp_counts = {}
        for s in self.image_sets:
            n = s['num_exposures']
            num_exp_counts[n] = num_exp_counts.get(n, 0) + 1
        
        print(f"Exposure count distribution:")
        for n in sorted(num_exp_counts.keys()):
            print(f"  - {n} exposures: {num_exp_counts[n]} sets")
    
    def _find_image_sets(self):
        """
        Find all sets of exposures + GT
        Returns list of dicts with patterns:
        {
            'set_path': path,
            'exposures': [path1, path2, ...],   # Variable N
            'gt': path,
            'num_exposures': N
        }
        """
        image_sets = []
        
        # Look for subfolders (each is a set)
        subfolders = sorted([d for d in glob(os.path.join(self.dataset_path, '*'))
                            if os.path.isdir(d)])
        
        for subfolder in subfolders:
            # Find all images
            jpg_files = sorted(glob(os.path.join(subfolder, '*.jpg')))
            png_files = sorted(glob(os.path.join(subfolder, '*.png')))
            all_images = jpg_files + png_files
            
            # Separate GT and exposures
            gt_candidates = [f for f in all_images if 'gt' in f.lower()]
            exposure_candidates = [f for f in all_images if 'gt' not in f.lower()]
            
            if gt_candidates and exposure_candidates:
                gt_path = gt_candidates[0]
                exposures = exposure_candidates
                
                # Filter by exposure count range
                num_exps = len(exposures)
                if num_exps < self.min_exposures:
                    continue  # Skip sets with too few exposures
                
                if self.max_exposures is not None and num_exps > self.max_exposures:
                    exposures = exposures[:self.max_exposures]
                
                image_sets.append({
                    'set_path': subfolder,
                    'exposures': exposures,
                    'gt': gt_path,
                    'num_exposures': len(exposures)
                })
        
        return image_sets
    
    def _load_image(self, img_path):
        """Load and normalize image with consistent size"""
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize to exact target size (width, height)
            # Note: PIL resize(size) expects (width, height)
            target_size = self.image_size  # (width, height)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to tensor
            img_tensor = TF.to_tensor(img)  # (3, H, W), [0, 1]
            
            # Verify output shape
            expected_shape = (3, target_size[1], target_size[0])  # (C, H, W)
            if img_tensor.shape != expected_shape:
                print(f"Warning: Expected shape {expected_shape}, got {img_tensor.shape}")
                # Force reshape if needed
                img_tensor = TF.resize(img_tensor, (target_size[1], target_size[0]))
            
            return img_tensor
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback with correct size
            return torch.zeros(3, self.image_size[1], self.image_size[0])
    
    def _augment(self, tensors_list):
        """Apply same augmentation to all images"""
        if self.mode != 'train':
            return tensors_list
        
        # Random flip
        if random.random() > 0.5:
            tensors_list = [TF.hflip(t) for t in tensors_list]
        
        if random.random() > 0.5:
            tensors_list = [TF.vflip(t) for t in tensors_list]
        
        # Random rotation
        if random.random() > 0.7:
            k = random.randint(0, 3)
            tensors_list = [torch.rot90(t, k=k, dims=(1, 2)) for t in tensors_list]
        
        return tensors_list
    
    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, idx):
        """
        Return:
            exposures: List of variable-length exposure tensors
                      Each tensor: (3, H, W)
            gt: Ground truth tensor (3, H, W)
            num_exposures: Number of exposures in this set
        """
        image_set = self.image_sets[idx]
        
        # Decide how many exposures to use
        exposures_paths = image_set['exposures'].copy()
        
        if self.fixed_n is not None:
            # Use fixed number
            if len(exposures_paths) > self.fixed_n:
                # Random sample
                exposures_paths = random.sample(exposures_paths, self.fixed_n)
            elif len(exposures_paths) < self.fixed_n:
                # Pad with repetition
                while len(exposures_paths) < self.fixed_n:
                    exposures_paths.append(random.choice(exposures_paths))
        else:
            # Use all exposures (variable N)
            pass
        
        # Load exposures
        exposures = []
        for exp_path in exposures_paths:
            exp_tensor = self._load_image(exp_path)
            exposures.append(exp_tensor)
        
        # Load GT
        gt_tensor = self._load_image(image_set['gt'])
        
        # Augmentation (apply same to all)
        tensors_all = exposures + [gt_tensor]
        tensors_all = self._augment(tensors_all)
        
        exposures = tensors_all[:-1]
        gt_tensor = tensors_all[-1]
        
        return exposures, gt_tensor


class MultiExposureFixedNLoader(MultiExposureDynamicLoader):
    """
    Convenience wrapper: always load exactly N exposures
    """
    def __init__(self, dataset_path, num_exposures=3, mode='train', 
                 image_size=(1200, 900)):
        super().__init__(
            dataset_path=dataset_path,
            mode=mode,
            image_size=image_size,
            min_exposures=1,
            max_exposures=None,
            fixed_n=num_exposures
        )
        self.num_exposures = num_exposures


def collate_variable_exposures(batch, target_size=(3, 900, 1200)):
    """
    Custom collate function for DataLoader
    Handles variable N exposures with size normalization
    
    Input: List of (exposures, gt) tuples
           where exposures is list of variable length
    
    Output: Tuple of (exposures_list, gt_tensor)
            exposures_list: List[Tensor] - batch of images, each (B, 3, H, W)
            gt_tensor: (B, 3, H, W)
    """
    exposures_batch = []
    gt_batch = []
    
    # Resize function for consistency
    def normalize_tensor(t, target_shape):
        """Ensure tensor has target shape"""
        if t.shape != target_shape:
            # Resize spatially
            h, w = target_shape[1], target_shape[2]
            t = TF.resize(t, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
        return t
    
    for exposures, gt in batch:
        # Normalize each exposure
        norm_exposures = [normalize_tensor(exp, target_size) for exp in exposures]
        exposures_batch.append(norm_exposures)
        
        # Normalize GT
        norm_gt = normalize_tensor(gt, target_size)
        gt_batch.append(norm_gt)
    
    # Stack GTs
    gt_tensor = torch.stack(gt_batch, dim=0)
    # Stack GTs
    gt_tensor = torch.stack(gt_batch, dim=0)
    
    # Keep exposures as list (variable N)
    # Convert to parallel structure: List[Tensor] where each Tensor is (B, 3, H, W)
    max_n = max(len(e) for e in exposures_batch)
    
    exposures_parallel = []
    for i in range(max_n):
        exp_list = []
        for exposures in exposures_batch:
            if i < len(exposures):
                exp_list.append(exposures[i])
            else:
                # Pad with zeros if this set has fewer exposures
                exp_list.append(torch.zeros_like(exposures[0]))
        
        exposures_parallel.append(torch.stack(exp_list, dim=0))
    
    return exposures_parallel, gt_tensor


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultiExposureDynamicLoader")
    print("=" * 70)
    
    # Create dummy dataset for testing
    TEST_DIR = 'test_multi_exp_dataset'
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create test data
    for set_idx in range(3):
        set_folder = os.path.join(TEST_DIR, f'set_{set_idx+1}')
        os.makedirs(set_folder, exist_ok=True)
        
        # Random number of exposures (1-4)
        n_exp = random.randint(1, 4)
        
        for exp_idx in range(n_exp):
            # Create dummy image
            img = Image.new('RGB', (100, 100), color=(100 + 20*exp_idx, 100, 100))
            img.save(os.path.join(set_folder, f'exp_{exp_idx+1}.jpg'))
        
        # Create GT
        gt = Image.new('RGB', (100, 100), color=(150, 150, 150))
        gt.save(os.path.join(set_folder, 'gt.jpg'))
    
    print(f"\nCreated test dataset in: {TEST_DIR}")
    
    # Test 1: Dynamic loader
    print("\n" + "-" * 70)
    print("[Test 1] MultiExposureDynamicLoader - Variable N")
    print("-" * 70)
    
    loader = MultiExposureDynamicLoader(
        dataset_path=TEST_DIR,
        mode='train',
        image_size=(100, 100),
        min_exposures=1,
        max_exposures=None  # No limit
    )
    
    for i in range(len(loader)):
        exposures, gt = loader[i]
        print(f"Set {i}: {len(exposures)} exposures, GT shape: {gt.shape}")
    
    # Test 2: Fixed N loader
    print("\n" + "-" * 70)
    print("[Test 2] MultiExposureFixedNLoader - Fixed N=2")
    print("-" * 70)
    
    fixed_loader = MultiExposureFixedNLoader(
        dataset_path=TEST_DIR,
        num_exposures=2,
        mode='train'
    )
    
    for i in range(len(fixed_loader)):
        exposures, gt = fixed_loader[i]
        print(f"Set {i}: {len(exposures)} exposures (fixed=2), GT shape: {gt.shape}")
    
    # Test 3: DataLoader with collate function
    print("\n" + "-" * 70)
    print("[Test 3] DataLoader with custom collate")
    print("-" * 70)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        loader,
        batch_size=2,
        collate_fn=collate_variable_exposures
    )
    
    for batch_idx, (exposures_batch, gt_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  - Number of exposure layers: {len(exposures_batch)}")
        for i, exp in enumerate(exposures_batch):
            print(f"    - Layer {i}: {exp.shape}")
        print(f"  - GT shape: {gt_batch.shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree(TEST_DIR)
    
    print("\n" + "=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)

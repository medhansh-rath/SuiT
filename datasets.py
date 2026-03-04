# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from typing import Any, Callable, Optional
import os
import json
from pathlib import Path

import numpy as np
import torch

from skimage.transform import rescale
from skimage.segmentation import slic
from fast_slic.avx2 import SlicAvx2

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class SpixImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        n_segments=196,
        compactness=10,
        downsample=2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        spix_method = 'fastslic',
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.n_segments = n_segments
        self.denormalize = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.compactness = compactness
        self.downsample = downsample
        self.spix_method = spix_method

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, assignment, target) where assignment is a map of superpixel indices for each pixel, and target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # augmented sample
            sample = self.transform(sample)
            
            # temporarily convert to [0, 255] and resize to acquire superpixels.
            sample_for_spix = np.array(self.denormalize(sample) * 255).transpose(1, 2, 0)
            sample_for_spix = rescale(sample_for_spix, 1 / self.downsample, anti_aliasing=True, channel_axis=2).round().clip(0, 255).astype(np.uint8)
            if self.spix_method == 'fastslic':
                slic_ = SlicAvx2(num_components=self.n_segments, compactness=self.compactness)
                assignment = slic_.iterate(sample_for_spix)
            elif self.spix_method == 'slic':
                assignment = slic(sample_for_spix, n_segments=self.n_segments, compactness=self.compactness)
            else:
                raise NotImplementedError

            assignment = torch.tensor(assignment).unsqueeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, assignment, target


class Denormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor


class SUNRGBDGeolexelsDataset(torch.utils.data.Dataset):
    """SUNRGBD dataset with precomputed GeoLexels superpixels."""
    
    def __init__(
        self,
        root: str,
        geolexels_cache_dir: str,
        n_segments: int = 196,
        downsample: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Root directory of SUNRGBD dataset
            geolexels_cache_dir: Directory containing precomputed GeoLexels (.npy files)
            n_segments: Number of segments (for clustering if needed)
            downsample: Downsample factor
            transform: Image transforms
            target_transform: Target transforms
        """
        self.root = Path(root)
        self.cache_dir = Path(geolexels_cache_dir)
        self.n_segments = n_segments
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.denormalize = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        
        # Find all RGB images
        self.samples = []
        for rgb_path in self.root.rglob('image/*.jpg'):
            # Find corresponding GeoLexels cache
            scene_dir = rgb_path.parent.parent
            frame_name = rgb_path.stem
            relative_path = scene_dir.relative_to(self.root)
            cache_file = self.cache_dir / relative_path / f'{frame_name}.npy'
            
            if cache_file.exists():
                self.samples.append({
                    'rgb': str(rgb_path),
                    'cache': str(cache_file),
                    'scene': str(relative_path)
                })
        
        if not self.samples:
            raise FileNotFoundError(f"No cached GeoLexels found in {geolexels_cache_dir}")
        
        print(f"Found {len(self.samples)} images with GeoLexels cache")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """
        Returns:
            tuple: (image, assignment, target) where assignment is GeoLexels superpixel map
        """
        import cv2
        
        item = self.samples[index]
        
        # Load RGB image
        image = cv2.imread(item['rgb'], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        # Load precomputed GeoLexels data
        geolexels_data = np.load(item['cache'])  # Shape: (H, W, 7)
        
        # Generate superpixel assignments from GeoLexels
        # Use clustering on the features to create n_segments superpixels
        assignment = self._generate_assignment_from_geolexels(
            geolexels_data, 
            (image.shape[1], image.shape[2])  # (H, W) in tensor format
        )
        
        # Convert assignment to tensor
        assignment = torch.tensor(assignment, dtype=torch.long).unsqueeze(0)
        
        # For SUNRGBD, we use scene as a pseudo-target (0 for now)
        target = 0
        
        return image, assignment, target
    
    def _generate_assignment_from_geolexels(self, geolexels_data, target_size):
        """
        Generate superpixel assignments from GeoLexels data.
        Clusters the GeoLexels features to create ~n_segments superpixels.
        
        Args:
            geolexels_data: (H, W, 7) array with features [RGB, depth, nx, ny, nz]
            target_size: (H, W) target size after downsampling
        
        Returns:
            assignment: (H, W) array with superpixel IDs
        """
        from sklearn.cluster import KMeans
        
        H, W = geolexels_data.shape[:2]
        
        # Downsample for clustering
        downsampled_h = H // self.downsample
        downsampled_w = W // self.downsample
        
        # Resample features to downsampled size
        downsampled_data = geolexels_data[::self.downsample, ::self.downsample, :]  # (H', W', 7)
        
        # Reshape to (N, 7) for clustering
        features = downsampled_data.reshape(-1, 7)
        
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=self.n_segments, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Reshape back to (H', W')
        assignment = labels.reshape(downsampled_h, downsampled_w)
        
        # Upsample to original resolution
        assignment_full = np.repeat(np.repeat(assignment, self.downsample, axis=0), 
                                     self.downsample, axis=1)
        
        # Trim to exact original size (in case of rounding issues)
        assignment_full = assignment_full[:H, :W]
        
        return assignment_full


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if 'suit' in args.model:
            dataset = SpixImageFolder(root, transform=transform, n_segments=args.n_spix_segments, compactness=args.compactness, downsample=args.downsample, spix_method=args.spix_method)
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'SUNRGBD':
        # SUNRGBD with precomputed GeoLexels
        dataset = SUNRGBDGeolexelsDataset(
            root=args.data_path,
            geolexels_cache_dir=getattr(args, 'geolexels_cache_dir', os.path.join(args.data_path, '.geolexels_cache')),
            n_segments=args.n_spix_segments,
            downsample=args.downsample,
            transform=transform,
        )
        nb_classes = 1  # SUNRGBD doesn't have semantic classes in this context

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

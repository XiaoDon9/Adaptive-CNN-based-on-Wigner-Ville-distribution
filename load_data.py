import os
import random
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
import psutil

def stratified_split(labels, train_ratio=0.7, val_ratio=0.1, seed=42):
    """
    Stratified sampling of the data by label is conducted to ensure that the distribution of various samples is consistent in different datasets
    
    Parameter:
        labels: Corresponding list of labels (for layering)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    
    Return:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)
    random.seed(seed)
    
    label_to_indices = {}
    
    for i, label in enumerate(labels):
        if isinstance(label, list) or isinstance(label, np.ndarray):
            key = tuple(label)
        else:
            key = label
            
        if key not in label_to_indices:
            label_to_indices[key] = []
        label_to_indices[key].append(i)
    
    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        
        n_samples = len(indices)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices

def get_memory_usage():
    total_memory = psutil.virtual_memory().total
    used_memory = psutil.virtual_memory().used
    return used_memory/total_memory

class MTLDataset(IterableDataset):
    def __init__(self, config, batch_type='train', cache=True):
        super().__init__()
        self.config = config
        self.batch_type = batch_type  # 'train', 'val', or 'test'
        
        self.data_caching = cache
        self.data_cached = {} if cache else None
        self.cache_key_prefix = batch_type + "_"
        self.max_cache_items = config.max_cache_items if hasattr(config, 'max_cache_items') and cache else 1000 # Default max cache items

        self.dim_input = self.config.data_length
        train_ratio = self.config.train_ratio
        val_ratio = getattr(self.config, 'val_ratio', 0.1) # Default val_ratio if not in config

        base_path = self.config.data_path

        all_filepaths = []
        # Stores tuples of (fault_label_idx, condition_label_values)
        all_multitask_labels = [] 
        # Labels used for stratification, e.g., fault labels
        stratification_labels = [] 

        # --- 1. Collect fault label mapping ---
        condition_dirs_all = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        all_fault_types_set = set()
        for cond_dir_name in condition_dirs_all:
            cond_path_loop = os.path.join(base_path, cond_dir_name)
            fault_dirs_loop = [
                d for d in os.listdir(cond_path_loop)
                if os.path.isdir(os.path.join(cond_path_loop, d))
            ]
            all_fault_types_set.update(fault_dirs_loop)

        self.fault_names = sorted(list(all_fault_types_set))
        self.fault_label_mapping = {name: i for i, name in enumerate(self.fault_names)}

        if hasattr(self.config, 'num_classes') and len(self.fault_label_mapping) != self.config.num_classes:
            print(f"Warning: Discovered {len(self.fault_label_mapping)} fault classes, but config.num_classes is {self.config.num_classes}.")

        # --- 2. Collect condition label mapping for classification ---
        # Sort condition names and assign integer labels
        self.condition_names = sorted(condition_dirs_all)
        self.condition_label_mapping = {name: idx for idx, name in enumerate(self.condition_names)}

        
        # --- 3. Iterate through files, assign combined labels ---
        for cond_dir_name in self.condition_names:
            condition_path = os.path.join(base_path, cond_dir_name)
            current_cond_label = self.condition_label_mapping[cond_dir_name]
            
            fault_dirs_in_cond = [
                d for d in os.listdir(condition_path)
                if os.path.isdir(os.path.join(condition_path, d))
            ]

            for fault_dir_name in fault_dirs_in_cond:
                if fault_dir_name not in self.fault_label_mapping:
                    print(f"Warning: Fault directory '{fault_dir_name}' not in fault_label_mapping. Skipping.")
                    continue # Skip if fault type is unknown (e.g. not in initial scan)
                
                current_fault_idx = self.fault_label_mapping[fault_dir_name]
                fault_path = os.path.join(condition_path, fault_dir_name)
                
                data_files = [f for f in os.listdir(fault_path) if f.endswith(".npy")]
                data_files.sort(key=lambda x: int(x.split(".")[0]))

                for data_file_name in data_files:
                    file_path = os.path.join(fault_path, data_file_name)
                    all_filepaths.append(file_path)
                    all_multitask_labels.append((current_fault_idx, current_cond_label))
                    stratification_labels.append(current_fault_idx) # Stratify by fault

        if not all_filepaths:
            raise ValueError("No data files found. Check data_path and data structure.")

        # --- 4. Stratified Split ---
        train_indices, val_indices, test_indices = stratified_split(
            stratification_labels, # Stratify by fault label
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.config.seed
        )

        if batch_type == 'train':
            selected_indices = train_indices
        elif batch_type == 'val':
            selected_indices = val_indices
        elif batch_type == 'test':
            selected_indices = test_indices
        else:
            raise ValueError("Invalid batch_type. Choose 'train', 'val', or 'test'.")

        self.data = [all_filepaths[i] for i in selected_indices]
        self.labels = [all_multitask_labels[i] for i in selected_indices] # List of (fault_idx, cond_list)

        print(f"Created {batch_type} dataset with {len(self.data)} multi-task samples.")
        if not self.data:
             print(f"Warning: {batch_type} dataset is empty after splitting.")


    def _sample(self, index):
        file_path = self.data[index]
        # labels is a tuple: (fault_label_scalar, condition_label_list)
        fault_label_scalar, condition_label_list = self.labels[index]
        
        cache_key = self.cache_key_prefix + file_path
        
        if self.data_cached is not None and cache_key in self.data_cached:
            data_np = self.data_cached[cache_key]
            
            data = torch.tensor(data_np, dtype=torch.float32).view(1, -1)
            fault_label = torch.tensor(fault_label_scalar, dtype=torch.long)
            condition_label = torch.tensor(condition_label_list, dtype=torch.float32)
            return data, (fault_label, condition_label)

        else:
            try:
                data_np = np.load(file_path)
                data_np = data_np.reshape([self.dim_input]) # Ensure correct shape
                
                if self.config.normalize:
                    mean = data_np.mean()
                    std = data_np.std()
                    data_np = (data_np - mean) / (std + 1e-8)
                    
                memory_usage_percentage = get_memory_usage()
                if self.data_caching and memory_usage_percentage < 0.8: # Memory threshold
                    if len(self.data_cached) >= self.max_cache_items:
                        # Simple FIFO cache eviction
                        for _ in range(min(100, len(self.data_cached) // 10 + 1)): # Remove a small portion
                            if self.data_cached:
                                self.data_cached.pop(next(iter(self.data_cached)))
                                
                    self.data_cached[cache_key] = data_np # Cache the numpy array
                
                data = torch.tensor(data_np, dtype=torch.float32).view(1, -1)
                fault_label = torch.tensor(fault_label_scalar, dtype=torch.long)
                condition_label = torch.tensor(condition_label_list, dtype=torch.float32)
                
                return data, (fault_label, condition_label)
            
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                # Return None for data and a tuple of Nones for labels to be handled by __iter__
                return None, (None, None)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # Single-process loading
            indices = list(range(len(self.data)))
            if self.batch_type == 'train':
                random.shuffle(indices) # Shuffle for training
            
            for idx in indices:
                data, label_tuple = self._sample(idx)
                # Ensure both data and all parts of the label_tuple are valid
                if data is not None and label_tuple[0] is not None and label_tuple[1] is not None:
                    yield data, label_tuple
        else:  # Multi-process loading
            per_worker = int(math.ceil(len(self.data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))
            
            indices = list(range(iter_start, iter_end))
            
            # Shuffle for training, ensure consistent shuffling across epochs if seed is managed outside
            if self.batch_type == 'train': # Shuffle worker's part for training
                random.Random(self.config.seed + worker_id).shuffle(indices) # Seeded shuffle for reproducibility per worker

            for idx in indices:
                data, label_tuple = self._sample(idx)
                if data is not None and label_tuple[0] is not None and label_tuple[1] is not None:

                    yield data, label_tuple

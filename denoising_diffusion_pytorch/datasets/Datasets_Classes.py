import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from denoising_diffusion_pytorch.utils import *  # Import all utility functions


class DataProcessed(Dataset):
    def __init__(self, file_paths, config, mean_std_dict, min_max_dict):
        """
        Args:
            file_paths: List of paths to netCDF files
            config: Configuration dict with scaling and augmentation options
        """
        self.file_paths = file_paths
        self.config = config
        self.loaded_mean_std_dict = mean_std_dict
        self.loaded_min_max_dict = min_max_dict
        self.augmentation = config.get('augment', False)
        self.scaler = None
        if config['scaling'] == 'quantile':
            self.scaler = QuantileTransformer()

        # Initialize cache size (you can adjust it depending on memory constraints)
        self.cache_size = config.get('cache_size', 10)
        self.cached_data = {}  # Manual cache for storing loaded data

    def __len__(self):
        # Calculate total number of samples across all files
        total_len = 0
        for path in self.file_paths:
            with xr.open_dataset(path) as ds:
                total_len += ds.sizes['time']  # Assuming 'time' is the main dimension
        return total_len

    def _apply_scaling(self, ds):
        if self.config['scaling'] == 'std':
            ds['PS'][:,:,:] = (ds['PS'][:,:,:] - self.loaded_mean_std_dict['PS_mean']) / self.loaded_mean_std_dict['PS_std']
            ds['PRECT'][:,:,:] = (ds['PRECT'][:,:,:] - self.loaded_mean_std_dict['PRECT_mean']) / self.loaded_mean_std_dict['PRECT_std']
            ds['TREFHT'][:,:,:] = (ds['TREFHT'][:,:,:] - self.loaded_mean_std_dict['TREFHT_mean']) / self.loaded_mean_std_dict['TREFHT_std']
    
            vars = ['PS','PRECT','TREFHT']
    
            for vv in vars:
                mini = self.loaded_min_max_dict[f'{vv}_min']
                maxi = self.loaded_min_max_dict[f'{vv}_max']
                if maxi > 10:
                    maxi = 10
                if mini <-10:
                    mini = 10
                
                ds[vv][:,:,:] =  (ds[vv][:,:,:] - mini) / (maxi - mini)

            ds['month_expanded']=ds['month_expanded'].copy()
            ds['co2vmr_expanded']=ds['co2vmr_expanded'].copy()

            ds['month_expanded_scaled'] = (ds['month_expanded'][:,:,:] - 1)/(12-1)
            ds['co2vmr_expanded_scaled'] = (ds['co2vmr_expanded'][:,:,:] - 0.00039895)/(0.0008223-0.00039895)
            
        else:
            raise ValueError("Invalid scaling method specified.")
    
        return ds

    def _unapply_scaling(self, ds):
        if self.config['scaling'] == 'std':
            # Reverse the min-max scaling for 'PS', 'PRECT', and 'TREFHT'
            vars = ['PS', 'PRECT', 'TREFHT']
            
            for vv in vars:
                mini = self.loaded_min_max_dict[f'{vv}_min']
                maxi = self.loaded_min_max_dict[f'{vv}_max']
                
                if maxi > 10:
                    maxi = 10
                if mini < -10:
                    mini = -10
                
                # Reverse the min-max scaling
                ds[vv] = ds[vv] * (maxi - mini) + mini
    
            # Reverse the standardization (z-score normalization) for 'PS', 'PRECT', and 'TREFHT'
            ds['PS'] = ds['PS'] * self.loaded_mean_std_dict['PS_std'] + self.loaded_mean_std_dict['PS_mean']
            ds['PRECT'] = ds['PRECT'] * self.loaded_mean_std_dict['PRECT_std'] + self.loaded_mean_std_dict['PRECT_mean']
            ds['TREFHT'] = ds['TREFHT'] * self.loaded_mean_std_dict['TREFHT_std'] + self.loaded_mean_std_dict['TREFHT_mean']
    
        else:
            raise ValueError("Invalid scaling method specified.")
    
        return ds

    def _augment_data(self, data):
        # Apply rotations for augmentation
        data_rot90 = np.rot90(data, k=1, axes=(1, 2))
        data_rot180 = np.rot90(data, k=2, axes=(1, 2))
        data_rot270 = np.rot90(data, k=3, axes=(1, 2))
        
        # Concatenate all rotations along the first dimension (time)
        return np.concatenate([data, data_rot90, data_rot180, data_rot270], axis=0)

    @lru_cache(maxsize=2)  # Cache up to 5 file loads at once
    def _load_data_from_file(self, file_path):
        """
        Lazy load the data from file and preprocess it.
        """
        with xr.open_dataset(file_path) as ds:

            # Get the shape of the other variables (time, lat, lon)
            lat = ds['PS'].coords['lat']
            lon = ds['PS'].coords['lon']
            
            # Broadcast co2vmr across (time, lat, lon)
            co2vmr_broadcasted = ds['co2vmr'].broadcast_like(ds['PS'])
            
            # Alternatively, you can expand the dimensions manually
            co2vmr_expanded = ds['co2vmr'].expand_dims({'lat': lat, 'lon': lon}, axis=(1, 2))
            
            # Assign this expanded co2vmr back to the dataset with the correct dimensions
            ds['co2vmr_expanded'] = co2vmr_expanded
            
            ds['time.month'].broadcast_like(ds['PS'])
            # Alternatively, you can expand the dimensions manually
            month_expanded = ds['time.month'].expand_dims({'lat': lat, 'lon': lon}, axis=(1, 2))
            month_expanded = month_expanded-1
            month_corrected = month_expanded.where(month_expanded != 0, 12) # check this line... 
            ds['month_expanded'] = month_corrected
    
            ds = self._apply_scaling(ds)
            data = np.swapaxes(ds[['PS','PRECT','TREFHT']].to_array().values,0,1)  # Replace 'forecast' with the relevant key in your dataset
            cond = np.swapaxes(ds[['month_expanded_scaled','co2vmr_expanded_scaled']].to_array().values,0,1)  # Replace 'forecast' with the relevant key in your dataset
    
            # Apply augmentation if necessary
            if self.augmentation:
                data = self._augment_data(data)
    
        return torch.clamp(torch.tensor(data, dtype=torch.float32),min=0, max=1), torch.clamp(torch.tensor(cond, dtype=torch.float32),min=0, max=1)

    def __getitem__(self, idx):
        """
        Load data lazily and cache it, based on global index.
        """
        # Determine which file and sample this index belongs to
        cumulative_len = 0
        for file_idx, path in enumerate(self.file_paths):
            with xr.open_dataset(path) as ds:
                file_len = ds.sizes['time']  # Length along the 'time' dimension
                if cumulative_len + file_len > idx:
                    sample_idx = idx - cumulative_len
                    data, cond = self._load_data_from_file(path)  # Load data from cache or disk
                    return data[sample_idx], cond[sample_idx]
                cumulative_len += file_len

        raise IndexError(f"Index {idx} is out of bounds")
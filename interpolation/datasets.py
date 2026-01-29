import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split


PHYSIONET_PARAMS = [
    'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]
PHYSIONET_PARAMS_DICT = {k: i for i, k in enumerate(PHYSIONET_PARAMS)}


class PhysioNetDataset(Dataset):
    def __init__(self, root_dir='./data/physionet', split='train', download=True, quantization=0.1):
        self.root_dir = root_dir
        self.split = split
        self.quantization = quantization
        self.params = PHYSIONET_PARAMS
        self.params_dict = PHYSIONET_PARAMS_DICT
        
        processed_file = os.path.join(root_dir, f'norm_{split}.pt')
        if download and not os.path.exists(processed_file):
            self._download_and_process()
        
        self.data = torch.load(processed_file, map_location='cpu')
    
    def _download_and_process(self):
        os.makedirs(self.root_dir, exist_ok=True)
        raw_dir = os.path.join(self.root_dir, 'raw')
        processed_dir = os.path.join(self.root_dir, 'processed')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        url = 'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download'
        tar_path = os.path.join(raw_dir, 'set-a.tar.gz')
        
        if not os.path.exists(tar_path):
            print(f"Downloading PhysioNet dataset...")
            urllib.request.urlretrieve(url, tar_path)
        
        dirname = os.path.join(raw_dir, 'set-a')
        if not os.path.exists(dirname):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(raw_dir)
        
        print('Processing PhysioNet data...')
        patients = []
        for txtfile in sorted(os.listdir(dirname)):
            if not txtfile.endswith('.txt'):
                continue
            record_id = txtfile.split('.')[0]
            with open(os.path.join(dirname, txtfile)) as f:
                lines = f.readlines()
                prev_time = 0
                tt = [0.]
                vals = [torch.zeros(len(self.params))]
                mask = [torch.zeros(len(self.params))]
                nobs = [torch.zeros(len(self.params))]
                for l in lines[1:]:
                    parts = l.strip().split(',')
                    if len(parts) != 3:
                        continue
                    time_str, param, val = parts
                    try:
                        time = float(time_str.split(':')[0]) + float(time_str.split(':')[1]) / 60.
                    except:
                        continue
                    time = round(time / self.quantization) * self.quantization
                    
                    if time != prev_time:
                        tt.append(time)
                        vals.append(torch.zeros(len(self.params)))
                        mask.append(torch.zeros(len(self.params)))
                        nobs.append(torch.zeros(len(self.params)))
                        prev_time = time
                    
                    if param in self.params_dict:
                        try:
                            val_float = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + val_float) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = val_float
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        except:
                            continue
                
                tt = torch.tensor(tt)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                patients.append((record_id, tt, vals, mask, None))
        
        torch.save(patients, os.path.join(processed_dir, f'set-a_{self.quantization}.pt'))
        
        train, test = train_test_split(patients, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.25, random_state=0)
        
        torch.save(train, os.path.join(processed_dir, 'train.pt'))
        torch.save(val, os.path.join(processed_dir, 'val.pt'))
        torch.save(test, os.path.join(processed_dir, 'test.pt'))
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_tv = self._remove_time_invariant(data)
            data_norm = self._normalize_data(data_tv)
            torch.save(data_norm, os.path.join(self.root_dir, f'norm_{name}.pt'))
        
        print(f"Saved: {len(train)} train, {len(val)} val, {len(test)} test samples")
    
    def _remove_time_invariant(self, data):
        result = []
        for sample in data:
            obs = sample[2][:, 4:]
            mask = sample[3][:, 4:]
            result.append((sample[0], sample[1], obs, mask, sample[4]))
        return result
    
    def _normalize_data(self, data):
        all_obs = torch.cat([sample[2] for sample in data])
        all_mask = torch.cat([sample[3] for sample in data])
        
        min_vals = []
        max_vals = []
        for i in range(all_obs.shape[1]):
            valid_vals = all_obs[:, i][all_mask[:, i] == 1]
            if len(valid_vals) > 0:
                min_vals.append(valid_vals.min())
                max_vals.append(valid_vals.max())
            else:
                min_vals.append(torch.tensor(0.0))
                max_vals.append(torch.tensor(1.0))
        
        min_vals = torch.stack(min_vals)
        max_vals = torch.stack(max_vals)
        max_vals[max_vals == min_vals] = min_vals[max_vals == min_vals] + 1
        
        result = []
        for sample in data:
            obs = sample[2]
            mask = sample[3]
            obs_norm = (obs - min_vals) / (max_vals - min_vals)
            obs_norm[mask == 0] = 0
            result.append((sample[0], sample[1], obs_norm, mask, sample[4]))
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record_id, tt, vals, mask, label = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class USHCNDataset(Dataset):
    def __init__(self, root_dir='./data/ushcn', split='train', download=True, sample_rate=0.5):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        
        processed_file = os.path.join(root_dir, f'norm_{split}.pt')
        if download and not os.path.exists(processed_file):
            self._download_and_process()
        
        self.data = torch.load(processed_file, map_location='cpu')
    
    def _download_and_process(self):
        os.makedirs(self.root_dir, exist_ok=True)
        
        url = 'https://raw.githubusercontent.com/usail-hkust/t-PatchGNN/main/data/ushcn/raw/small_chunked_sporadic.csv'
        csv_path = os.path.join(self.root_dir, 'small_chunked_sporadic.csv')
        
        if not os.path.exists(csv_path):
            print(f"Downloading USHCN dataset...")
            urllib.request.urlretrieve(url, csv_path)
        
        print('Processing USHCN data...')
        df = pd.read_csv(csv_path)
        
        value_cols = [c for c in df.columns if c.startswith('Value_')]
        mask_cols = [c for c in df.columns if c.startswith('Mask_')]
        
        grouped = df.groupby('ID')
        all_data = []
        
        for station_id, group in grouped:
            group = group.sort_values('Time')
            times = torch.tensor(group['Time'].values, dtype=torch.float32)
            values = torch.tensor(group[value_cols].values, dtype=torch.float32)
            mask = torch.tensor(group[mask_cols].values, dtype=torch.float32)
            values = torch.nan_to_num(values, nan=0.0)
            all_data.append((str(station_id), times, values, mask))
        
        train, test = train_test_split(all_data, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.25, random_state=0)
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_norm = self._normalize_data(data)
            torch.save(data_norm, os.path.join(self.root_dir, f'norm_{name}.pt'))
        
        print(f"Saved: {len(train)} train, {len(val)} val, {len(test)} test samples")
    
    def _normalize_data(self, data):
        all_obs = torch.cat([sample[2] for sample in data])
        all_mask = torch.cat([sample[3] for sample in data])
        
        min_vals = []
        max_vals = []
        for i in range(all_obs.shape[1]):
            valid_vals = all_obs[:, i][all_mask[:, i] == 1]
            if len(valid_vals) > 0:
                min_vals.append(valid_vals.min())
                max_vals.append(valid_vals.max())
            else:
                min_vals.append(torch.tensor(0.0))
                max_vals.append(torch.tensor(1.0))
        
        min_vals = torch.stack(min_vals)
        max_vals = torch.stack(max_vals)
        max_vals[max_vals == min_vals] = min_vals[max_vals == min_vals] + 1
        
        result = []
        for sample in data:
            obs = sample[2]
            mask = sample[3]
            obs_norm = (obs - min_vals) / (max_vals - min_vals)
            obs_norm[mask == 0] = 0
            result.append((sample[0], sample[1], obs_norm, mask))
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        station_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


def create_interpolation_task(times, values, mask, sample_rate=0.5):
    observed_mask = mask.clone().bool()
    n_total = observed_mask.sum().item()
    n_observed = max(1, int(n_total * sample_rate))
    
    observed_indices = torch.where(observed_mask)
    if len(observed_indices[0]) == 0:
        return observed_mask, torch.zeros_like(observed_mask, dtype=torch.bool)
    
    perm = torch.randperm(len(observed_indices[0]))[:n_observed]
    
    new_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    new_mask[observed_indices[0][perm], observed_indices[1][perm]] = True
    
    target_mask = observed_mask & (~new_mask)
    
    return new_mask, target_mask

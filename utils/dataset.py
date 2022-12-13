"""
1. Clean the dataset
2. Dataset
3. Dataloader
"""

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import pandas as pd
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, config):

        self.gen_new_data = config['gen_new_data']
        self.clean_data_dir = config['clean_data_dir']
        self.raw_data_dir = config['raw_data_dir']
        self.sigma = config['sigma']
        self.scaling_column = [
            # 'Load (mW)',
                            'Pressure_kpa', 
                            'Cloud Cover (%)', 
                            'Humidity (%)', 
                            'Temperature (C)', 
                            'Wind Direction (deg)', 
                            'Wind Speed (kmh)']
        
        self.is_scale = config['is_scale']
        self.scale_type = config['scale_type']
        self.one_year = config['one_year']
        
        # load data
        if self.gen_new_data or not os.path.exists(self.clean_data_dir):
            print('Generating new data...')
            self.dataframe = self.data_clean()
        else:
            print('Loading data...')
            self.dataframe = pd.read_csv(self.clean_data_dir)
        
        # scale the entire dataset
        if self.is_scale:
            self.scale()
            
        self.target = self.dataframe['Load (mW)'].to_numpy()
        self.dataframe.drop(columns = ['Load (mW)'], inplace = True)
        self.data = self.dataframe.to_numpy()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.target[index]).float()
    
    def scale(self):
        
        # construct the mean and std for normalization
        # train_frame = self.dataframe[:self.train_size]
        
        if self.scale_type == 'minmax':
            self.min = {}
            self.max = {}    
            
            for column_name in self.scaling_column:
                # find the min max on train set
                self.min[column_name] = self.dataframe[column_name].min()
                self.max[column_name] = self.dataframe[column_name].max()
                # apply on the entire dataset
                self.dataframe[column_name] = (self.dataframe[column_name] - self.min[column_name])/(self.max[column_name] - self.min[column_name])
        
        elif self.scale_type == 'standard':
            self.mean = {}
            self.std = {}
            
            for column_name in self.scaling_column:
                self.mean[column_name] = self.dataframe[column_name].mean()
                self.std[column_name] = self.dataframe[column_name].std()
                self.dataframe[column_name] = (self.dataframe[column_name] - self.mean[column_name])/self.std[column_name]
        
        else:
            raise ValueError('Wrong scale type')
            
    def data_clean(self):
        
        dataframe = pd.read_excel(self.raw_data_dir)
        dataframe.rename(columns = {'Temperature (C) ': 'Temperature (C)'}, inplace = True)
        # select one year
        if self.one_year:
            index = (dataframe['Time'] >= "2017-03-18 00:00:00") & (dataframe['Time'] <= "2018-03-18 23:00:00")
            # index = (dataframe['Time'] >= "2018-06-01 00:00:00") & (dataframe['Time'] <= "2019-06-01 23:00:00")
            dataframe = dataframe[index]
        
        # kW -> mW
        dataframe['Load (kW)'] = dataframe['Load (kW)']/1e6
        dataframe.rename({'Load (kW)': 'Load (mW)'}, axis = 1, inplace = True)  
        
        # outlier scaling
        dataframe = self.outlier_scaling(dataframe)
        
        # periodical encoding
        dataframe['Year'] = dataframe.Time.dt.year
        dataframe['Month_sin'] = np.sin(dataframe.Time.dt.month/12*2*np.pi)
        dataframe['Month_cos'] = np.cos(dataframe.Time.dt.month/12*2*np.pi)
        dataframe['Day_sin'] = np.sin(dataframe.Time.dt.day/30*2*np.pi)
        dataframe['Day_cos'] = np.cos(dataframe.Time.dt.day/30*2*np.pi)
        dataframe['Hour_sin'] = np.sin(dataframe.Time.dt.hour/24*2*np.pi)
        dataframe['Hour_cos'] = np.cos(dataframe.Time.dt.hour/24*2*np.pi)
        dataframe.drop(columns = ['Time', 'Year'], inplace=True)

        dataframe.to_csv(self.clean_data_dir, index = False)
        
        return dataframe
    
    def outlier_scaling(self, dataframe):
        
        for column_name in self.scaling_column:

            # remove outliers
            mean = dataframe[column_name].mean()
            std = dataframe[column_name].std()
            outlier_max = mean + self.sigma*std
            outlier_min = mean - self.sigma*std

            dataframe[column_name] = dataframe[column_name].apply(lambda x: x if x < outlier_max else mean)
            dataframe[column_name] = dataframe[column_name].apply(lambda x: x if x > outlier_min else mean)
        
        return dataframe

def return_dataloader(config, args):
    
    batch_size = args.batch_size
    dataset = MyDataset(config)
    train_size = int(config['train_prop'] * len(dataset))
    test_size = len(dataset) - train_size
    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], 
                                                        generator=torch.Generator().manual_seed(config['random_seed']))
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    return train_dataloader, test_dataloader
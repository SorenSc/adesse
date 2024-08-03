import copy
import pickle
import random
from datetime import timedelta, datetime
from typing import Type, List, Iterator
import dacite

import numpy as np
import pandas as pd
import toml
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from config.config import RepositioningConfiguration, WholeCityDemandPredictionConfiguration

from matrix_generation.matrix_generator import MatrixGenerator
from matrix_generation.pseudo_grid_creator.PseudoGridCreator import PseudoSquareGrid
from pickup_demand_prediction.models import HaliemetalCNNNet, SingleCellFCNN


class RandomBatchSampler(Sampler[List[int]]):
    replacement: bool

    def __init__(self, data_source) -> None:
        self.data_source = data_source

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        yield from [self.data_source[i] for i in torch.randperm(n, generator=generator).tolist()]

    def __len__(self) -> int:
        return self.num_samples


class RepositioningDataset(Dataset):
    """
    Loads the data needed for an upcoming episode and also generates the #request prediction.
    """

    def __init__(self, idx, idx_all, config_request_n, config_taxi, single_cell=True):
        self.single_cell = single_cell
        self.idx = idx
        self.dev = config_request_n.dev
        self.episode_length = config_request_n.episode_length
        self.ds_requests_n = NormalizedAdditionalFeatureDataset(config_request_n, idx_all)
        config_request = copy.copy(config_request_n)
        config_request.normalized = False
        self.ds_requests = AdditionalFeatureDataset(config_request, idx_all, True, self.ds_requests_n)
        self.ds_taxis = CityCenterDataset(config_taxi, idx_all)
        self.request_predictor = self.create_request_predictor()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, id):
        # `r_X_n` refers to the Normalized number of Requests 
        r_X_n, r_pois_n, r_X_add_n, r_y = self.ds_requests_n.__getitem__(id)
        r_X, r_pois, r_X_add, _ = self.ds_requests.__getitem__(id)
        t_X, t_y = self.ds_taxis.__getitem__(id)

        # Predict #requests per cell 
        if self.single_cell:  # via single cell model
            requests_X_single_cell, _ = DatasetCommons.prepare_data_fcnn(
                r_X_n.reshape(1, 4, 20, 20),
                r_pois_n.reshape(1, 20, 20),
                r_X_add_n.reshape(1, 13),
                r_y.reshape(1, 20, 20),
                self.dev, False, 0, True)
            r_y_p = self.request_predictor.predict(requests_X_single_cell).reshape(20, 20)
        else:  # Via city-wide model
            r_y_p = torch.squeeze(self.request_predictor(r_X_n))
        return r_X_n, r_pois_n, r_X_add_n, r_X, r_pois, r_X_add, r_y, r_y_p, t_X, t_y

    def create_request_predictor(self):
        if self.single_cell:
            request_predictor = SingleCellFCNN()
            request_predictor.load_state_dict(
                torch.load(f'./pickup_demand_prediction/models/SingleCellFCNNv3_1_128_0.001',
                        map_location=torch.device(self.dev))['model_state_dict'])
        else:  # self.single_cell = False or the city-wide case
            config = toml.load(r'./config/whole_city_demand_predictor.toml')
            config = dacite.from_dict(data_class=WholeCityDemandPredictionConfiguration, data=config)
            request_predictor = HaliemetalCNNNet(config)
            request_predictor.load_state_dict(
                torch.load(f'./pickup_demand_prediction/models/WholeCityCNNv3',
                        map_location=torch.device(self.dev))['model_state_dict'])
        request_predictor.to(torch.device(self.dev))
        # Make sure the model is not optimized. Otherwise, the training is slowed down dramatically (and 
        # the code does not work :))
        for param in request_predictor.parameters():
            param.requires_grad = False
        return request_predictor


class DemandDataset(Dataset):
    """
    Standard dataset that extracts demand matrices from a given dictionary. The dataset is mostly used for pickup
    demand, but can theoretically also deal with dropoff data.

    For doing so, in the __getitem__ method, we first identify the corresponding indices from self.idx and get the
    keys from the dictionary demand_dict via keys_at_t, which makes the method faster. When all keys for each of the
    identified indices is selected, we build and fill a matrix. From this matrix, we can extract X and y.
    """

    def __init__(self, config, idx):
        self.config, self.idx = config, idx
        self.demand_dict = DatasetCommons.load_demand_dictionary(self.config.demand_dict_file)
        self.square_grid, self.matrix_gen = DatasetCommons.create_matrix_gen(config)
        self.transforms = transforms.Normalize([self.config.X_mean] * config.nof_input_maps,
                                                [self.config.X_std] * config.nof_input_maps)
        self.keys_at_t = DatasetCommons.get_keys_for_all_t(self.config, self.idx, self.demand_dict)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, id, normalized=True):
        index = self.idx[id]
        time_idx = [index + i for i in range(self.config.nof_input_maps + 1)]
        keys_of_interest = []
        for key in time_idx: keys_of_interest.extend(self.keys_at_t[key])
        demand_subset = {key: self.demand_dict[key] for key in keys_of_interest}
        demand_mat = np.zeros(shape=(len(time_idx), self.matrix_gen.grid.nof_y_cells, self.matrix_gen.grid.nof_x_cells), dtype=np.uint32)
        for key in demand_subset: demand_mat[time_idx.index(key[2]), key[1], key[0]] = demand_subset[key]
        demand_mat = np.einsum('abcd->acd', demand_mat.reshape(
            self.config.nof_input_maps + 1, 1, self.matrix_gen.grid.nof_y_cells, self.matrix_gen.grid.nof_x_cells))
        if normalized:  # Default
            X = self.transforms(torch.tensor(demand_mat[0:-1, :, :].astype('float32')))
        else:  # For instance used for explaining the predicted demand
            X = torch.tensor(demand_mat[0:-1, :, :].astype('float32'))
        y = torch.from_numpy(demand_mat[-1, :, :, ].astype('float32'))
        return X, y


class CityCenterDataset(DemandDataset):
    """
    Reduces the considered area to the one of Manhattan. Depending on the grid cell size, the values used in
    __getitem__ vary for config['red_t'], config['red_b'], config['red_l'], and config['red_r'].
    `red` refers to reduction and `_<>` to the direction from which the original dataset is reduced.
    """

    def __init__(self, config, idx, from_dataset=False, ds=None):
        if not from_dataset:
            super(CityCenterDataset, self).__init__(config, idx)
        else:
            self.config, self.idx = config, idx
            self.demand_dict = ds.demand_dict
            self.square_grid, self.matrix_gen = ds.square_grid, ds.matrix_gen
            self.transforms = ds.transforms
            self.keys_at_t = ds.keys_at_t

    def __getitem__(self, index):
        X, y = super().__getitem__(index, self.config.normalized)
        # Reduce considered area to roughly Manhattan
        X = X[:, self.config.red_t:-self.config.red_b, self.config.red_l:-self.config.red_r]
        y = y[self.config.red_t:-self.config.red_b, self.config.red_l:-self.config.red_r]
        return X, y


class AdditionalFeatureDataset(CityCenterDataset):

    def __init__(self, config, idx, from_dataset=False, ds=None):
        if not from_dataset:
            super(AdditionalFeatureDataset, self).__init__(config, idx)
            self.weather_dict = self.create_weather_dict(self.config.weather_file)
            self.holiday_dict = self.create_holiday_dict()
            self.nof_pois = self.create_nof_pois()  # Idea from https://doi.org/10.3390/app11209675
        else:
            self.config, self.idx = config, idx
            self.demand_dict = ds.demand_dict
            self.square_grid, self.matrix_gen = ds.square_grid, ds.matrix_gen
            self.transforms = ds.transforms
            self.keys_at_t = ds.keys_at_t
            self.weather_dict = ds.weather_dict
            self.holiday_dict = ds.holiday_dict
            self.nof_pois = ds.nof_pois

    def create_nof_pois(self):
        nof_pois = pickle.load(open(f'./data/additional_data_nof_pois.npy', 'rb'))
        nof_pois = nof_pois[self.config.red_t:-self.config.red_b, self.config.red_l:-self.config.red_r]
        return nof_pois

    def create_holiday_dict(self, create=False):
        if create:  # Works only when self.idx = ds.idx
            holidays = {}
            self.config.holidays = [datetime.strptime(d, "%d.%m.%Y") for d in self.config.holidays]
            for i in self.idx:
                datetime_at_index = DatasetCommons.idx_to_datetime(self.idx[i], self.config)
                day = datetime(datetime_at_index.year, datetime_at_index.month, datetime_at_index.day, 0, 0)
                holiday_ohe = 1.0 if day in self.config.holidays else 0.0  # One hot encoded
                holidays[i] = holiday_ohe
            pickle.dump(holidays, open(f'./data/holiday_dict', 'wb'))
        else:  # load
            holidays = pickle.load(open(f'./data/additional_data_holiday_dict', 'rb'))
        return holidays

    def create_weather_dict(self, file, create=False):
        if create:
            weather_df = pd.read_csv(
                file,
                parse_dates=['reported_date_time', 'start_date_time'],
                date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

            # Create dictionary with idx as key and weather data as value
            weather = {}
            for i in self.idx:
                datetime_at_index = DatasetCommons.idx_to_datetime(self.idx[i], self.config)
                try:
                    weather_row = weather_df[(weather_df.start_date_time < datetime_at_index) &
                                             (datetime_at_index <= weather_df.reported_date_time)].iloc[0]
                    weather[i] = np.asarray((weather_row.temp, weather_row.wind, weather_row.humidity, weather_row.barometer,
                                             weather_row['view'], weather_row.snow, weather_row.precipitation,
                                             weather_row.cloudy))
                except IndexError:
                    print(i)
            weather[0] = weather[1]
            pickle.dump(weather, open(f'./data/additional_data_weather_dict', 'wb'))
        else:  # load
            weather = pickle.load(open(f'./data/additional_data_weather_dict', 'rb'))
        return weather

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)

        # Collect additional features
        datetime_at_index = DatasetCommons.idx_to_datetime(self.idx[idx], self.config)
        hour = datetime_at_index.hour
        minute = datetime_at_index.minute
        weekday = datetime_at_index.weekday()
        month = datetime_at_index.month
        weather = self.weather_dict[self.idx[idx]]
        holiday = self.holiday_dict[self.idx[idx]]

        # Transform additional features
        time_features = torch.tensor((hour, minute, weekday, month), dtype=torch.float32)
        weather_features = torch.tensor(weather, dtype=torch.float32)
        holiday_feature = torch.tensor([holiday], dtype=torch.float32)
        X_add = torch.cat((time_features, weather_features, holiday_feature))
        return X, torch.tensor(self.nof_pois), X_add, y


class NormalizedAdditionalFeatureDataset(AdditionalFeatureDataset):

    def __init__(self, config, idx):
        super(NormalizedAdditionalFeatureDataset, self).__init__(config, idx)
        self.X_add_means = torch.tensor([11.5, 29.5, 3.0, 5.5, 12.59857542610023, 9.03183668277792, 56.25228949376749, 1017.0959425082676, 14.523289239379293, 0.01469091834138896, 0.05737725769524294, 0.35105571101500893, 0.02930552022386161])
        self.X_add_std = torch.tensor([6.922186552431729, 17.318102282486574, 2.0, 3.452052529534663, 10.289749748716927, 6.506661862899048, 18.279020770525968, 7.802860124425204, 3.356261603888417, 0.1203124900402099, 0.23256205192295804, 0.47730032346391243, 0.16866151519617759])
        self.transform_pois = transforms.Normalize([12.42], [13.6322])
        self.pois_n = self.transform_pois(torch.tensor(self.nof_pois).reshape(1, 20, 20)).reshape(20, 20)

    def __getitem__(self, idx):
        X, _, X_add, y = super().__getitem__(idx)
        X_add_n = (X_add - self.X_add_means) / self.X_add_std
        return X, self.pois_n, X_add_n, y


class PUandDODemandDataset(DemandDataset):
    """
    Returns X and y for both, the pickup and dropoff.
    """
    def __init__(self, idx, pickup_ds: DemandDataset, dropoff_ds: DemandDataset):
        super(PUandDODemandDataset, self).__init__(pickup_ds.config, idx)
        self.idx = idx
        self.pickup_ds = pickup_ds
        self.dropoff_ds = dropoff_ds
        self.pu_demand_predictor = SingleCellFCNN()
        self.pu_demand_predictor.load_state_dict(
            torch.load(f'./pickup_demand_prediction/models/SingleCellFCNNv3_1_128_0.001',
                       map_location=torch.device('cpu'))['model_state_dict'])
        

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        pu_X, pu_X_pois, pu_X_add, pu_y = self.pickup_ds.__getitem__(idx)
        do_X, do_y = self.dropoff_ds.__getitem__(idx)

        # Create pick-up prediction
        pu_X = pu_X.reshape(1, 4, 20, 20)
        pu_X_pois = pu_X_pois.reshape(1, 20, 20)
        pu_X_add = pu_X_add.reshape(1, 13)
        pu_y = pu_y.reshape(1, 20, 20)
        pu_X, _ = DatasetCommons.prepare_data_fcnn(pu_X, pu_X_pois, pu_X_add, pu_y, 'cpu', False, 0, True)
        pu_y_p = self.pu_demand_predictor.predict(pu_X).reshape(20, 20)
        pu_y_p = pu_y_p.round().abs()  # Make sure the values are positive ints or zero.

        return pu_X.reshape(20, 20, 20), pu_y, do_X, do_y, pu_y_p


class DatasetCommons:

    @staticmethod
    def create_data_loader(config_requests, config_taxis, load=False, single_cell=True):
        """
        This method creates a data loader that loads one episode at every call. Therefore, a pretrained demand
        prediction model is used directly.
        :return:
        """
        if load:
            # pickle.dump(dl, open('./data/dl.pickle', 'wb'))
            return pickle.load(open('./data/dl.pickle', 'rb'))
        else:
            idx_all, _, _, _, idx_train_rp, _, _ = \
                DatasetCommons.get_train_validation_test_indices_for_repositioning(config_requests)
            # +3 as we need one per default, and additional two as the data is accessed before done is checked
            idx_list = [[i + j for j in list(range(config_requests.episode_length + 3))] for i in idx_train_rp]
            ds = RepositioningDataset(idx_train_rp, idx_all, config_requests, config_taxis, single_cell)
            return DataLoader(ds, batch_sampler=RandomBatchSampler(idx_list))        


    @staticmethod
    def get_train_validation_test_indices_for_repositioning(config: RepositioningConfiguration):
        """
        This method separates all indices available - all 10 minute time steps from January 2015 till June 2016 - into
        six lists. For each model - demand prediction and repositioning - three lists are created - train, validation, and
        test. The separation is done as follows:
            - The last two month are used for testing of both models.
            - The month before are randomly divided into train and validation with a ratio of around 80/20.
            - Each month is split after its 15ths - the first half is used for demand prediction, the second for
                repositioning.
        We ensure that no indice is used twice by not including those indices that fall into the overlapping area - for
        instance the index corresponding to the 16th of January 2016 at 00:30 am is not included as the demand prediction
        model uses four input maps and the fifth one - 16th of January 2016 at 00:40 am - as y.

        78764 overall with 78763 as the highest index which refers to datetime.datetime(2016, 6, 30, 23, 10)
        Length of all training and validation sets and one test set is 78548
        """
        validation_months = [(2015, 3), (2015, 9),
                             (2016, 1)]  # Randomly selected via [random.random() > 0.8 for i in range(16)]
        test_months = [(2016, 5), (2016, 6)]
        idx_all = list(range(DatasetCommons.get_dataset_length(config)))
        idx_train_dp, idx_valid_dp, idx_test_dp, idx_train_rp, idx_valid_rp, idx_test_rp = [], [], [], [], [], []
        for i in idx_all:
            current_datetime = DatasetCommons.transform_idx_to_time(i, config)
            demand_prediction = current_datetime.day <= 15
            validation_month = (current_datetime.year, current_datetime.month) in validation_months
            overlapping = DatasetCommons.overlapped_idx(current_datetime, config.nof_input_maps, config.time_bin_size)
            test = (current_datetime.year, current_datetime.month) in test_months
            if not overlapping:
                if test:
                    idx_test_dp.append(i), idx_test_rp.append(i)
                else:
                    if demand_prediction:
                        idx_valid_dp.append(i) if validation_month else idx_train_dp.append(i)
                    else:
                        idx_valid_rp.append(i) if validation_month else idx_train_rp.append(i)
        return idx_all, idx_train_dp, idx_valid_dp, idx_test_dp, idx_train_rp, idx_valid_rp, idx_test_rp
        # Length of the returned idx lists: 78764, 28002, 6462, 8756, 28578, 6750, 8756

    @staticmethod
    def transform_idx_to_time(idx: int, config) -> datetime:
        return config.start + idx * timedelta(minutes=config.time_bin_size)

    @staticmethod
    def overlapped_idx(dt, nof_input_maps=4, time_bin_size=10):
        if dt.day not in [1, 16]:  # Beginning or middle of month?
            return False
        else:
            if dt.hour != 0:  # Beginning of the day?
                return False
            else:
                return dt.minute <= (nof_input_maps + 1) * time_bin_size  # First part of the hour?

    @staticmethod
    def create_set_of_random_dropoff_locations(do_y=None, size=1000, load=True):
        if not load:
            x_dist, y_dist = do_y.sum(axis=(0, 1)).numpy(), do_y.sum(axis=(0, 2)).numpy()
            potential_x_dest, potential_y_dest = list(range(len(x_dist))), list(range(len(y_dist)))
            random_x = torch.tensor(random.choices(potential_x_dest, weights=x_dist, k=size))
            random_y = torch.tensor(random.choices(potential_y_dest, weights=y_dist, k=size))
            random_dropoff_locations = torch.stack([random_x, random_y], dim=1)
            pickle.dump(random_dropoff_locations, open('./data/random_dropoff_locations.pickle', 'wb'))
        else:
            random_dropoff_locations = pickle.load(open('./data/random_dropoff_locations.pickle', 'rb'))
        return random_dropoff_locations

    @staticmethod
    def transform_to_single_cell_data(X, X_pois, X_add, y, normalized=False, size=20):
        X, X_pois, X_add, y = X.numpy(), X_pois.numpy(), X_add.numpy(), y.numpy()
        result = []
        for sample in range(X.shape[0]):
            for y_index in range(size):
                for x_index in range(size):
                    if normalized:
                        x_i_n = (x_index - 9.5) / 5.766281297335398
                        y_i_n = (y_index - 9.5) / 5.766281297335398
                    else:
                        x_i_n, y_i_n = x_index, y_index
                    result.append(np.concatenate((
                        x_i_n, y_i_n,  # Index features
                        X[sample, :, y_index, x_index].flatten(),  # Previous pickup demand
                        X_pois[sample, y_index, x_index],  #
                        X_add[sample],  # time, weather etc. features
                        y[sample, y_index, x_index]  #
                    ), axis=None))
        return np.asarray(result)

    @classmethod
    def prepare_data_fcnn(cls, X, X_pois, X_add, y, dev, randomized=False, size=1, normalized=True):
        data = cls.transform_to_single_cell_data(X, X_pois, X_add, y, normalized)
        if randomized:
            indices = np.random.choice(data.shape[0], size, replace=False)
            data = data[indices]
        return torch.tensor(data[:, :20]).to(dev), torch.tensor(data[:, 20].reshape(data.shape[0], 1)).to(dev)


    @staticmethod
    def get_dataset_length(config):
        duration_in_minutes = (config.end - config.start).total_seconds() // 60
        length_with_last_input_maps = int(duration_in_minutes // config.time_bin_size)
        return length_with_last_input_maps - config.nof_input_maps

    @staticmethod
    def idx_to_datetime(idx, config):
        if idx <= config.dataset_len:
            return config.start + timedelta(minutes=int(idx * config.time_bin_size // 1))
        else:
            raise Exception

    @staticmethod
    def get_train_validation_test_indices_for_demand_prediction(config):
        dataset_indices = list(range(0, config.dataset_len))

        # Take test indices from the end of the dataset
        test_indices = dataset_indices[-(config.test_size + config.nof_input_maps):-config.nof_input_maps]

        len_train_valid_dataset = len(dataset_indices) - len(test_indices) - 2 * config.nof_input_maps
        train_valid_indices = range(len_train_valid_dataset)

        # Get train indices
        nof_indices_for_train = int((1 - config.validation_size) * len_train_valid_dataset)
        train_indices = random.sample(train_valid_indices, nof_indices_for_train)
        train_indices.sort()

        # Get validation indices
        validation_indices = list(np.setdiff1d(train_valid_indices, train_indices))

        return dataset_indices, train_indices, validation_indices, test_indices

    @staticmethod
    def create_data_loaders(config, dataset_class: Type[DemandDataset]):
        ds_idx, train_idx, valid_idx, test_idx = DatasetCommons.get_train_validation_test_indices_for_demand_prediction(config)
        train_dl = DataLoader(dataset_class(config, train_idx),
                              shuffle=True, batch_size=config.batch_size, drop_last=True)
        valid_dl = DataLoader(dataset_class(config, valid_idx),
                              shuffle=True, batch_size=config.batch_size, drop_last=True)
        test_dl = DataLoader(dataset_class(config, test_idx),
                             shuffle=True, batch_size=config.batch_size, drop_last=True)
        return train_dl, valid_dl, test_dl

    @staticmethod
    def create_data_loaders_given_lengths(config, dataset_class: Type[DemandDataset], n_samples):
        ds_idx, train_idx, valid_idx, test_idx = DatasetCommons.get_train_validation_test_indices_for_demand_prediction(config)
        train_dl = DataLoader(dataset_class(config, train_idx),
                              shuffle=True, batch_size=n_samples[0], drop_last=True)
        valid_dl = DataLoader(dataset_class(config, valid_idx),
                              shuffle=True, batch_size=n_samples[1], drop_last=True)
        test_dl = DataLoader(dataset_class(config, test_idx),
                             shuffle=True, batch_size=n_samples[2], drop_last=True)
        return train_dl, valid_dl, test_dl

    @staticmethod
    def create_matrix_gen(config):
        square_grid = PseudoSquareGrid(None, config.grid_cell_area,
                                            config.pgc_grid_bl_lat, config.pgc_grid_bl_lon,
                                            config.pgc_grid_tr_lat, config.pgc_grid_tr_lon)
        matrix_gen = MatrixGenerator('square', square_grid, config)
        return square_grid, matrix_gen

    @staticmethod
    def load_demand_dictionary(demand_dict_file: str):
        return np.load(demand_dict_file, allow_pickle=True).item()

    @staticmethod
    def get_keys_for_all_t(config, idx, demand_dict):
        # Enhance list of idx by idx + 1, idx + 2, ..., idx + nof_input_maps + 1 so that all indices necessary for
        # creating X and y are included.
        idx_plus_nof_input_maps = set()
        for idx in idx:
            for i in range(config.nof_input_maps + 1):  # + 1 to also cover y
                idx_plus_nof_input_maps.add(idx + i)

        # For each idx of the previously generated enhanced list, add all keys from the demand_dict so that accessing
        # it during __getitem__() is faster. It is much fast doing it this way instead of getting the keys at each call
        # of __getitem__().
        keys_at_t = {}
        for t_id in idx_plus_nof_input_maps: keys_at_t[t_id] = []
        for key in demand_dict.keys():
            if key[2] in idx_plus_nof_input_maps:
                keys_at_t[key[2]].append(key)
        return keys_at_t

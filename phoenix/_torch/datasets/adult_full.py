from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets.abstract_dataset import AbstractDataset


class AdultFullDataset(AbstractDataset):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    train_labels_map = {'<=50K': 0, '>50K': 1}
    test_labels_map = {'<=50K.': 0, '>50K.': 1}

    def __init__(self, split, args=None, normalize=True):
        super().__init__('adult', split)

        train_data_file = path.join(self.data_dir, 'adult.data')
        test_data_file = path.join(self.data_dir, 'adult.test')

        if not path.exists(train_data_file):
            request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', train_data_file
            )
        if not path.exists(test_data_file):
            request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', test_data_file
            )

        train_dataset = pd.read_csv(train_data_file, sep=',', header=None, names=AdultFullDataset.column_names)
        test_dataset = pd.read_csv(test_data_file, sep=',', header=0, names=AdultFullDataset.column_names)

        # preprocess strings
        train_dataset = train_dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        test_dataset = test_dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # drop missing values
        train_dataset.replace(to_replace='?', value=np.nan, inplace=True)
        test_dataset.replace(to_replace='?', value=np.nan, inplace=True)
        train_dataset.dropna(axis=0, inplace=True)
        test_dataset.dropna(axis=0, inplace=True)

        # encode labels
        train_dataset.replace(AdultFullDataset.train_labels_map, inplace=True)
        test_dataset.replace(AdultFullDataset.test_labels_map, inplace=True)

        # split features and labels
        train_features, train_labels = train_dataset.drop('income', axis=1), train_dataset['income']
        test_features, test_labels = test_dataset.drop('income', axis=1), test_dataset['income']

        continuous_vars = []
        self.categorical_columns = []
        for col in train_features.columns:
            if train_features[col].isnull().sum() > 0:
                train_features.drop(col, axis=1, inplace=True)
            else:
                if train_features[col].dtype == np.object:
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]

        if args is not None and args.protected_att is not None:
            protected_att = args.protected_att
        else:
            protected_att = 'sex'

        self.protected_unique = train_features[protected_att].nunique()
        protected_train = np.logical_not(pd.Categorical(train_features[protected_att]).codes)
        protected_test = np.logical_not(pd.Categorical(test_features[protected_att]).codes)

        # one-hot encode categorical data
        train_features = pd.get_dummies(train_features, columns=self.categorical_columns, prefix_sep='=')
        test_features = pd.get_dummies(test_features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [train_features.columns.get_loc(var) for var in continuous_vars]

        # add missing column to test dataset
        test_features.insert(
            loc=train_features.columns.get_loc('native_country=Holand-Netherlands'),
            column='native_country=Holand-Netherlands', value=0
        )

        self.one_hot_columns = {}
        for column_name in self.categorical_columns:
            ids = [i for i, col in enumerate(train_features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        print('categorical features: ', self.one_hot_columns.keys())

        train_features = torch.tensor(train_features.values.astype(np.float32), device=self.device)
        train_labels = torch.tensor(train_labels.values.astype(np.int64), device=self.device)
        train_protected = torch.tensor(protected_train.astype(np.bool), device=self.device)

        test_features = torch.tensor(test_features.values.astype(np.float32), device=self.device)
        test_labels = torch.tensor(test_labels.values.astype(np.int64), device=self.device)
        test_protected = torch.tensor(protected_test.astype(np.bool), device=self.device)

        # filter

        train_features = train_features[:, self.continuous_columns]
        test_features = test_features[:, self.continuous_columns]

        self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
            train_features, train_labels, train_protected, test_size=0.2, random_state=0
        )

        self.X_test = test_features
        self.y_test = test_labels
        self.protected_test = test_protected

        if normalize:
            self._normalize(self.continuous_columns)
        else:
            assert(0) #

        self._assign_split()
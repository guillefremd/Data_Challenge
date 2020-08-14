import numpy as np
import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
np.warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, filename, delimiter='\t'):
        self.filename = filename
        self.np_array = np.genfromtxt(
            filename, delimiter=delimiter, skip_header=1, dtype=float
        )
        with open(filename) as f:
            self.columns = f.readline().strip().split(delimiter)
        self.cleaned = self.clean(self.np_array)
        self.no_outliers = self.remove_outliers(self.np_array)
        self.scaled = scale(self.cleaned)

    def clean(self, np_array):
        np_array = np.copy(np_array)
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.fit(np_array)
        np_array = imp.transform(np_array)
        return np_array

    def remove_outliers(self, np_array, iqr_param=1.5):
        np_array = np.copy(np_array)
        q1 = np.nanpercentile(np_array, 25, axis=0)
        q3 = np.nanpercentile(np_array, 75, axis=0)
        iqr = q3 - q1
        cut_off = iqr * iqr_param
        lower = q1 - cut_off
        upper = q3 + cut_off
        np_array[np_array < lower] = np.nan
        np_array[np_array > upper] = np.nan
        return self.clean(np_array)

    def get_variable_idx(self, variable_name):
        return self.columns.index(variable_name)

    def get_variable_name(self, variable_idx):
        return self.columns[variable_idx]

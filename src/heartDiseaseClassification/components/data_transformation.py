from heartDiseaseClassification import logger
from sklearn.model_selection import train_test_split
from heartDiseaseClassification.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import wfdb 
import ast
import pywt
import random


class Datatransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_ecg_data(self, df, freq_rate, path):
        """loadding ECG data with 100 frequency

        Args:
            df (pandas): (pbtxl_database.csv) dataframe containing location of ecg data
            freq_rate (int): frequency of ECG signal
            path (Path): root dir

        Returns:
            numpy.ndarry : ECG signal in form of numpy array
        """
        logger.info(f"loading the ecg data with freq_rate {freq_rate}")
        data = []
        if freq_rate == 100:
            # data = df['filename_lr'].apply(lambda x:wfdb.rdsamp(os.path.join(path,x)) )
            # data = [wfdb.rdsamp(os.path.join(path,f)) for f in df.filename_lr]
            for filename in df["filename_lr"]:
                try:
                    sample = wfdb.rdsamp(os.path.join(os.path.join(path, filename)))
                    data.append(sample)
                except Exception as e:
                    logger.error(f"error processing file {filename}: {e}")
        else:
            for filename in df["filename_hr"]:
                try:
                    sample = wfdb.rdsamp(os.path.join(os.path.join(path, filename)))
                    data.append(sample)
                except Exception as e:
                    logger.error(f"error processing file {filename}: {e}")
        logger.info(f"ecg data loaded!! converting to numpy.")
        data = np.array([signal for signal, meta in data])
        return data

    def load_ptbxl_csv_file(self):
        """
        loadding ptbxl_database.csv file
        """
        logger.info(f"loading csv file !! location {self.config.ptbxl_csv_path}")
        Y = pd.read_csv(self.config.ptbxl_csv_path, index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        X = self.load_ecg_data(Y, self.config.freq_rate, self.config.data_path)
        np.save(self.config.ecg_data_path, X)
        logger.debug(f"The shape of ecg data is: {X.shape}")
        logger.debug(f"The shape of ptbxl file data is: {Y.shape}")

    def aggregate_diagnostic(self, dit_superclass_dict):
        tmp = []
        Y = pd.read_csv(self.config.ptbxl_csv_path)
        for key in dit_superclass_dict.keys():
            if key in Y.index:
                tmp.append(Y.loc[key].diagnostic_class)
        return list(set(tmp))

    def train_test_split(self):
        scp_statement = pd.read_csv(self.config.scp_csv_path, index_col=0)
        Y = pd.read_csv(self.config.ptbxl_csv_path)
        X = np.load(self.config.ecg_data_path)
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y["diagnostic_superclass"] = Y.scp_codes.apply(self.aggregate_diagnostic)
        ## location all super classes total 5
        ## maksing with the length of the list =1
        mask = Y["diagnostic_superclass"].apply(lambda x: len(x) == 1)
        Y_filterted = Y[mask]
        X_filterted = X[mask]

        Y_filterted.apply(lambda x: x[0])
        test_fold, val_fold = 10, 9

        logger.info(f"maskng bet testing and validation set")
        # Train
        X_train = X_filterted[
            np.where(
                (Y_filterted.strat_fold != test_fold)
                & (Y_filterted.strat_fold != val_fold)
            )
        ]
        y_train = Y_filterted[
            (
                (Y_filterted.strat_fold != test_fold)
                & (Y_filterted.strat_fold != val_fold)
            )
        ].diagnostic_superclass

        # Test
        X_test = X_filterted[np.where(Y_filterted.strat_fold == test_fold)]
        y_test = Y_filterted[
            (Y_filterted.strat_fold == test_fold)
        ].diagnostic_superclass

        # Valdiation
        X_val = X_filterted[np.where((Y_filterted.strat_fold == val_fold))]
        y_val = Y_filterted[(Y_filterted.strat_fold == val_fold)].diagnostic_superclass

        # logger.info(f"sacinvg hte training testing ddata_split/ata")
        np.save(os.path.join(self.config.root_dir, "data_split/X_train.npy"), X_train)
        np.save(os.path.join(self.config.root_dir, "data_split/X_test.npy"), X_test)
        np.save(os.path.join(self.config.root_dir, "data_split/X_val.npy"), X_val)
        np.save(os.path.join(self.config.root_dir, "data_split/y_train.npy"), y_train)
        np.save(os.path.join(self.config.root_dir, "data_split/y_test.npy"), y_test)
        np.save(os.path.join(self.config.root_dir, "data_split/y_val.npy"), y_val)

        print(f"y_train shape {y_train.shape}, X_train shape {X_train.shape}")
        print(f"y_test shape {y_test.shape}, X_test shape {X_test.shape}")
        print(f"y_val shape {y_val.shape}, X_val shape {X_val.shape}")

    def swt_augment(self,data, wavelet='db4', levels=2, augment_params=None):
        """
        Augment data using Stationary Wavelet Transform (SWT).
        
        Args:
            data (numpy.ndarray): Input data of shape (num_samples, num_features).
            wavelet (str, optional): Wavelet family to use for SWT. Default is 'db4'.
            levels (int, optional): Number of decomposition levels for SWT. Default is 3.
            augment_params (dict, optional): Dictionary containing augmentation parameters.
                'scale': Scale factor for scaling wavelet coefficients.
                'noise': Standard deviation of Gaussian noise to add to wavelet coefficients.
                'permute': Boolean indicating whether to permute wavelet coefficients.
                'warping': Maximum time warping factor.
        
        Returns:
            numpy.ndarray: Augmented data of the same shape as input data.
        """
        augmented_data = []
        
        for sample in data:
            coeffs = pywt.swt(sample, wavelet, level=levels, start_level=0)
            augmented_sample = sample.copy()
            
            if augment_params is not None:
                for coeff in coeffs:
                    if augment_params.get('scale', None) is not None:
                        coeff *= np.multiply(coeff, augment_params['scale'])
                    if augment_params.get('noise', None) is not None:
                        coeff += np.random.normal(0, augment_params['noise'], coeff.shape)
                    if augment_params.get('permute', False):
                        np.random.shuffle(coeff)
                    if augment_params.get('warping', None) is not None:
                        warping_factor = np.random.uniform(-augment_params['warping'], augment_params['warping'])
                        coeff = coeff.reshape(-1)
                        coeff = np.roll(coeff, int(len(coeff) * warping_factor))
                        coeff = coeff.reshape(coeff.shape[0], -1)
                
                augmented_sample = pywt.iswt(coeffs, wavelet)
            
            augmented_data.append(augmented_sample)
        
        return np.array(augmented_data)
    
    def applying_SWT(self):
        y_train = np.load(f"{self.config.root_dir}/data_split/y_train.npy")
        X_train = np.load(f"{self.config.root_dir}/data_split/X_train.npy")
        logger.info(f" -----------the sze sna shape of {y_train.shape, X_train.shape}")
        counts = np.unique(y_train, return_counts=True)[1]
        augment_params = {
            "scale": random.uniform(0.8, 1.2),
            "noise": random.uniform(0.0, 0.2),
            "permute": random.choice([True, False]),
        }
        for i, value in enumerate(counts):
            if i == 0:
                # print("passing statemetn", i)
                continue
            aug_data_size1   = 4000 - value
            aug_data_size = value + aug_data_size1
            # print(i, aug_data_size)
            X_temp = np.zeros(shape=(aug_data_size, 1000, 12))
            y_temp = np.zeros(shape=(aug_data_size,), dtype="int")

            y_train_index = np.where(y_train == i)[0]

            for j in range(aug_data_size):
                ran_index = np.random.choice(y_train_index)
                ecg_data = X_train[ran_index].T
                aug_data = self.swt_augment(ecg_data,augment_params=augment_params)
                # print(j)
                X_temp[j] = aug_data.T
                y_temp[j] = i

            # print(
            #     "class no and original shape and new data shape", i, X_train.shape, X_temp.shape
            # )
            X_train = np.concatenate((X_train, X_temp))
            y_train = np.concatenate((y_train, y_temp))
        # logger.info(f"shape is ------- {np.unique(y_train,return_counts=True)}")
            
        np.save(os.path.join(self.config.aug_data_path,"X_train.npy"),X_train)
        np.save(os.path.join(self.config.aug_data_path,"y_train.npy"),y_train)
        
        print("tstet")
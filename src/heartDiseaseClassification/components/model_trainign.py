import numpy as np
from heartDiseaseClassification.entity.config_entity import ModelTrainerConfig
import tensorflow as tf
from heartDiseaseClassification import logger
import pickle
from keras import backend as K

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Precision, Recall, AUC, CategoricalAccuracy
from tensorflow.keras.utils import to_categorical


class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config
        
    def recall_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self,y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def build_model(self,input_shape,num_classes):
        model = Sequential()

        # Increased model capacity and depth
        model.add(Conv1D(filters=128, kernel_size=15, padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=256, kernel_size=11, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=512, kernel_size=7, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=1024, kernel_size=5, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GlobalAveragePooling1D())

        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation="softmax"))
        return model

    def train_model(self):
        input_shape = (self.config.input_size0,self.config.input_size1)
        num_classes = self.config.output_size
        model = self.build_model(input_shape=input_shape,num_classes=num_classes)     
        metrics = [
            Precision(),
            Recall(),
            AUC(multi_label=True, num_labels=num_classes),
            CategoricalAccuracy(),
            "accuracy",
            self.f1_m
        ]
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate=0.001),
            metrics=metrics,
        )   
        X_train = np.load(f"{self.config.train_data_path}/X_train.npy")
        y_train = np.load(f"{self.config.train_data_path}/y_train.npy")
        X_val = np.load(f"{self.config.test_data_path}/X_val.npy")
        y_val = np.load(f"{self.config.test_data_path}/y_val.npy")
        y_val = to_categorical(y_val,num_classes=5)
        y_train = to_categorical(y_train,num_classes=5)
        
        history = model.fit(
            X_train,y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[
                ModelCheckpoint(
                    filepath="./artifacts/model_training/model/ECG-Classifier/{epoch:02d}-{val_categorical_accuracy:.2f}.keras",
                    monitor="val_categorical_accuracy",
                    save_best_only=True,
                ),
                # TensorBoard("./artifacts/model_training/model/pre-final/logs", update_freq=1),
                # EarlyStopping(monitor="val_categorical_accuracy", patience=10, restore_best_weights=True),
            ],
        )
        with open('./artifacts/model_training/model/history-model/history-best-model.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        model.save("./artifacts/model_training/model/final.keras")
    # def train(self):
    #     train_data = pd.read_csv(self.config.train_data_path)
    #     test_data = pd.read_csv(self.config.test_data_path)
        
    #     print(test_data.columns)
        
    #     train_x = train_data.drop([self.config.target_column],axis=1)
    #     test_x = test_data.drop([self.config.target_column],axis=1)
    #     train_y = train_data[[self.config.target_column]]
    #     print(train_y.columns)
    #     test_y = test_data[[self.config.target_column]]
        
        

    #     lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=43)
    #     lr.fit(train_x,train_y)
        
    #     joblib.dump(lr,os.path.join(self.config.root_dir,self.config.model_name))
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers, utils
from tensorflow.keras.layers.experimental import preprocessing


def split(dataframe):
    train, val, test = np.split(
        dataframe.sample(frac=1),
        [int(0.75*len(dataframe)), int(0.9*len(dataframe))])
    print(len(train), 'training examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    return (train, val, test)


@tf.autograph.experimental.do_not_convert
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = preprocessing.Normalization()
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    # Learn the statistics of the data
    normalizer.adapt(feature_ds)
    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


class Estimator:
    def __init__(self, dataframe, target_name):
        dataframe = dataframe.rename(columns={target_name: 'target'})
        cols  = ['home', 'away', 'date', 'home_score', 'away_score', 'winner']
        dataframe = dataframe.drop(columns=cols , errors='ignore')
        df_train, df_val, df_test = split(dataframe)
        self.ds_train = self._dataframe_to_dataset(df_train)
        self.ds_val = self._dataframe_to_dataset(df_val, shuffle=False)
        self.ds_test = self._dataframe_to_dataset(df_test, shuffle=False)
        self.feature_names = list(dataframe.drop(columns=['target']).columns)
        self.metric = None  # Set in subclass
        self.model = None
        self.history = None

    def run(self, epochs=5):
        # self._check()
        print("Build model")
        self.build_model()
        # self.model.summary()
        print("Train and evaluate model")
        self.fit(epochs)
        self.model.evaluate(self.ds_test)
        return self

    def build_model(self):
        """Override in subclass"""
        pass

    def fit(self, epochs=5):
        callbacks = [
            # keras.callbacks.ModelCheckpoint(
            #     "best_model.h5", save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=50, verbose=1,
                restore_best_weights=True)
        ]
        self.history = self.model.fit(
            self.ds_train, epochs=epochs, callbacks=callbacks,
            validation_data=self.ds_val)

    def plot_validation_curve(self):
        plt.figure()
        plt.plot(self.history.history[self.metric])
        plt.plot(self.history.history["val_" + self.metric])
        plt.title("model " + self.metric)
        plt.ylabel(self.metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.show()
        plt.close()

    @staticmethod
    def _dataframe_to_dataset(dataframe, shuffle=True, batch_size=64):
        df = dataframe.copy()
        labels = df.pop('target')
        ds = Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    def _inputs_and_encoded_features(self):
        all_inputs = []
        encoded_features = []
        # Numerical features
        for name in self.feature_names:
            feature = keras.Input(shape=(1,), name=name)
            feature_en = encode_numerical_feature(feature, name, self.ds_train)
            all_inputs.append(feature)
            encoded_features.append(feature_en)
        return (all_inputs, encoded_features)

    def _check(self):
        [(train_features, label_batch)] = self.ds_train.take(1)
        print('Every feature:', list(train_features.keys()))
        print('A batch of goals_diff:', train_features['goals_diff'])
        print('A batch of targets:', label_batch )


class Classifier(Estimator):
    def __init__(self, dataframe, target_name='winner'):
        super().__init__(dataframe, target_name)
        self.metric = 'accuracy'

    def build_model(self):
        # Deep Feed Forward
        all_inputs, encoded_features = self._inputs_and_encoded_features()
        all_features = layers.concatenate(encoded_features)
        x = layers.Dense(128, activation='relu')(all_features)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        # Multiclass Classification -> Softmax
        output = layers.Dense(3, activation='softmax')(x)
        model = keras.Model(all_inputs, output)
        model.compile(
            'adam', 'categorical_crossentropy', metrics=[self.metric])
        self.model = model

    @staticmethod
    def _dataframe_to_dataset(dataframe, shuffle=True, batch_size=64):
        df = dataframe.copy()
        labels = utils.to_categorical(df.pop('target'), 3)
        ds = Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds


class Regressor(Estimator):
    def __init__(self, dataframe, target_name='home_score'):
        super().__init__(dataframe, target_name)
        self.metric = 'mae'
 
    def build_model(self):
        # Deep Feed Forward
        all_inputs, encoded_features = self._inputs_and_encoded_features()
        all_features = layers.concatenate(encoded_features)
        x = layers.Dense(128, activation='relu')(all_features)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        # Regression -> Linear
        output = layers.Dense(1, activation='linear')(x)
        model = keras.Model(all_inputs, output)
        model.compile('adam', 'mse', metrics=[self.metric])
        self.model = model

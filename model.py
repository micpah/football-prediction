import matplotlib.pyplot as plt

from numpy import split
from tensorflow import convert_to_tensor
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def train_val_test_split(dataframe):
    train, val, test = split(
        dataframe.sample(frac=1),
        [int(0.75*len(dataframe)), int(0.9*len(dataframe))])
    print(len(train), 'training examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    return (train, val, test)

def standardize(df_train, df_val, df_test):
    mean = df_train.mean()
    stddev = df_train.std()
    train = (df_train - mean) / stddev
    val = (df_val - mean) / stddev
    test = (df_test - mean) / stddev
    return (train, val, test)


class Data:
    def __init__(self, dataframe, target_name):
        df = self.relevant_columns(dataframe, target_name)
        df_trn, df_val, df_tst = train_val_test_split(df)
        x_trn, y_trn = self.x_y_split(df_trn)
        x_val, y_val = self.x_y_split(df_val)
        x_tst, y_tst = self.x_y_split(df_tst)
        x_trn, x_val, x_tst = standardize(x_trn, x_val, x_tst)
        if target_name == 'winner':
            y_trn = to_categorical(y_trn, 3)
            y_val = to_categorical(y_val, 3)
            y_tst = to_categorical(y_tst, 3)
        self.x_trn = convert_to_tensor(x_trn)
        self.y_trn = convert_to_tensor(y_trn)
        self.x_val = convert_to_tensor(x_val)
        self.y_val = convert_to_tensor(y_val)
        self.x_tst = convert_to_tensor(x_tst)
        self.y_tst = convert_to_tensor(y_tst)
        self.feature_names = list(df.drop(columns=['target']).columns)

    @staticmethod
    def relevant_columns(dataframe, target_name):
        df = dataframe.rename(columns={target_name: 'target'})
        irrelevant_cols  = [
            'home', 'away', 'date',
            'home_score', 'away_score', 'winner']
        return df.drop(columns=irrelevant_cols , errors='ignore')

    @staticmethod
    def x_y_split(dataframe):
        y = dataframe['target']
        X = dataframe.drop(columns=['target']).astype('float32')
        return (X, y)


class Estimator:
    def __init__(self, data):
        self.data = data
        self.n_features = len(data.feature_names)
        self.loss = None  # Init in subclass
        self.metric = None  # Init in subclass
        self.last_layer = None  # Init in subclass
        self.model = None
        self.history = None

    def run(self, epochs=5):
        print("Build model")
        self.build_model()
        # self.model.summary()
        print("Train and evaluate model")
        self.fit(epochs)
        self.model.evaluate(self.data.x_tst, self.data.y_tst)
        return self


    def build_model(self):
        # Deep Feed Forward
        model = Sequential(
            [
                InputLayer(input_shape=(self.n_features,)),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.5),
                self.last_layer
            ]
        )
        model.compile('adam', self.loss, metrics=[self.metric])
        self.model = model

    def fit(self, epochs=5, batch_size=64):
        callbacks = [
            # ModelCheckpoint(
            #     "best_model.h5", save_best_only=True, monitor="val_loss"),
            EarlyStopping(
                monitor="val_loss", patience=50, verbose=1,
                restore_best_weights=True)
        ]
        self.history = self.model.fit(
            self.data.x_trn, self.data.y_trn,
            batch_size, epochs,
            callbacks=callbacks,
            validation_data=(self.data.x_val, self.data.y_val))

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


class Classifier(Estimator):
    def __init__(self, data):
        super().__init__(data)
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.last_layer = Dense(3, activation='softmax')


class Regressor(Estimator):
    def __init__(self, data):
        super().__init__(data)
        self.loss = 'mse'
        self.metric = 'mae'
        self.last_layer = Dense(1, activation='linear')

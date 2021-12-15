import matplotlib.pyplot as plt

from numpy import split
from tensorflow import convert_to_tensor
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from keras_tuner import Hyperband


def train_val_test_split(dataframe):
    train, val, test = split(
        dataframe.sample(frac=1),
        [int(0.7*len(dataframe)), int(0.85*len(dataframe))])
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
        self.output_units = None  # Init in subclass
        self.output_activation = None  # Init in subclass
        self.model = None
        self.history = None
        self.tuner = None

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
                Dense(self.output_units, self.output_activation)
            ]
        )
        model.compile('adam', self.loss, metrics=[self.metric])
        self.model = model

    def fit(self, epochs=5, batch_size=64):
        self.history = self.model.fit(
            self.data.x_trn, self.data.y_trn,
            batch_size, epochs,
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
            validation_data=(self.data.x_val, self.data.y_val))

    def hypertune(self, project_name, overwrite, max_epochs=80, batch_size=64):
        tuner = Hyperband(
            self.build_hypermodel, 'val_loss', max_epochs,
            directory='logs', project_name=project_name, overwrite=overwrite,
        )
        tuner.search_space_summary()
        tuner.search(
            self.data.x_trn, self.data.y_trn,
            batch_size=batch_size,
            callbacks = [
                EarlyStopping(patience=20),
                TensorBoard('logs/tb/'+project_name)],
            validation_data=(self.data.x_val, self.data.y_val),
        )
        tuner.results_summary()
        self.tuner = tuner

    def build_hypermodel(self, hp):
        n_layers, acti, w_init, dr, optimizer, lr = self.hyperparameters(hp)
        model = Sequential([InputLayer(input_shape=(self.n_features,))])
        # Tune the number of layers
        for i in range(n_layers):
            model.add(Dense(
                # Tune number of units separately
                hp.Choice(f"units_{i}", [32, 64, 128, 256, 512]),
                activation=acti, kernel_initializer=w_init))
            model.add(Dropout(dr))
        model.add(Dense(self.output_units, self.output_activation))
        model.compile(eval(optimizer)(lr), self.loss, metrics=[self.metric])
        return model

    @staticmethod
    def hyperparameters(hp):
        n_hidden_layers = hp.Int('n_hidden_layers', min_value=1, max_value=3)
        activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
        kernel_initializer = hp.Choice(
            'kernel_initializer',['he_uniform', 'glorot_uniform'])
        dropout_rate = hp.Float('rate', min_value=0.2, max_value=0.5, step=0.1)
        optimizer = hp.Choice('optimizer', ['Adam', 'RMSprop'])
        lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        params = (
            n_hidden_layers, activation, kernel_initializer,
            dropout_rate, optimizer, lr
        )
        return params


    def fit_and_evaluate_best_model(self, epochs=200, batch_size=64):
        best_hps = self.tuner.get_best_hyperparameters(1)
        print("\nTrain and evaluate best model (with following parameters):")
        print(best_hps[0].values)
        self.model = self.build_hypermodel(best_hps[0])
        self.fit(epochs, batch_size)
        self.model.evaluate(self.data.x_tst, self.data.y_tst)

    def load_tuner(self, project_name, max_epochs=80):
        tuner = Hyperband(
            self.build_hypermodel, 'val_loss', max_epochs,
            directory='logs', project_name=project_name, overwrite=False,
        )
        return tuner

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
        self.output_units = 3
        self.output_activation = 'softmax'


class Regressor(Estimator):
    def __init__(self, data):
        super().__init__(data)
        self.loss = 'mse'
        self.metric = 'mae'
        self.output_units = 1
        self.output_activation = 'linear'



if __name__ == "__main__":
    from time import gmtime, strftime
    from data import prepared_data

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    df = prepared_data(n_trend=8)

    print("\nTune hyperparameters for multiclass classification")
    clf = Classifier(Data(df, target_name='winner'))
    clf.hypertune(project_name='clf', overwrite=True, max_epochs=200)
    clf.fit_and_evaluate_best_model()

    print("\nTune hyperparameters for regression (of home_score)")
    rgr1 = Regressor(Data(df, target_name='home_score'))
    rgr1.hypertune(project_name='rgr_home', overwrite=True, max_epochs=200)
    rgr1.fit_and_evaluate_best_model()

    print("\nTune hyperparameters for regression (of away_score)")
    rgr2 = Regressor(Data(df, target_name='away_score'))
    rgr2.hypertune(project_name='rgr_away', overwrite=True, max_epochs=200)
    rgr2.fit_and_evaluate_best_model()

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

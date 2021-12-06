# from tensorflow.keras import utils

from data import read, preprocess, transform
from model import Classifier, Regressor


def run_data_preparation(path, n_trend=5):
    print("Read data")
    df = read(path)
    print("Preprocess data")
    df = preprocess(df)
    print("Transform data")
    df = transform(df, n_trend)
    # print(df.shape)
    # print(df.winner.value_counts() / df.shape[0])
    return df


if __name__ == "__main__":
    df = run_data_preparation('./data/matches.csv')

    print("\nRun multiclass classification")
    clf = Classifier(df)
    clf.run(epochs=100)

    print("\nRun regression (for home_score)")
    rgr1 = Regressor(df, target_name='home_score')
    rgr1.run(epochs=100)

    print("\nRun regression (for away_score)")
    rgr2 = Regressor(df, target_name='away_score')
    rgr2.run(epochs=100)

    # Plot and save model graphs
    # utils.plot_model(clf.model, to_file='./doc/gfx/model_clf.png', show_shapes=True, rankdir="LR")
    # utils.plot_model(rgr1.model, to_file='./doc/gfx/model_rgr_home.png', show_shapes=True, rankdir="LR")
    # utils.plot_model(rgr2.model, to_file='./doc/gfx/model_rgr_away.png', show_shapes=True, rankdir="LR")

    # Plot validation curves
    clf.plot_validation_curve()
    rgr1.plot_validation_curve()
    rgr2.plot_validation_curve()

    # TODO: Pickle clf, rgr1, rgr2
    # TODO: Predict upcoming match day (of e.g. Bundesliga)
    # TODO: Proper Hyperparameter Tuning

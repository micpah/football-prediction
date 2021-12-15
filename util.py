import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data import prepared_data

def correlation_heatmap(dataframe):
    # Compute the correlation matrix
    corr = dataframe.corr().round(2)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot = True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

if __name__ == "__main__":
    sns.set_theme(style="white")
    df = prepared_data(n_trend=8)
    correlation_heatmap(df)

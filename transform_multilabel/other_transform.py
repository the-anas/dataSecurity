import numpy as np
import pandas as pd
dataframe = pd.read_csv("Other.csv")
dataframe.drop('category', axis=1, inplace=True)
dataframe = dataframe[~dataframe.duplicated(subset=['segment', 'Other Type'])]
print((dataframe['Other Type'].iloc[0]))


def label_to_vector(label, labels, count):
    """

    Returns a vector representing the label passed as an input.

    Args:
        label: string, label that we want to transform into a vector.
        labels: dictionary, dictionary with the labels as the keys and indexes as the values.
    Returns:
        vector: np.array, 1-D array of lenght 12.

    """

    vector = np.zeros((count), dtype=np.int64)
    try:

        index = labels[label]

        vector[index] = 1

    except KeyError:

        vector = np.zeros((count), dtype=np.int64)

    return vector

labels = {'Introductory/Generic': 0,'Practice not covered': 1,'Other': 2,'Privacy contact information': 3}


dataframe['Other Type'] = dataframe['Other Type'].apply(lambda x: label_to_vector(x, labels, 4)) # returns one hot encoding in a 1D-12D vector

labels_data = dataframe[[ 'segment', 'Other Type']]
labels = labels_data.groupby("segment").sum() # since segments can have many labels, it sums label vectors of the same segment, resuling in a vector with all the labels of the segment marked as 1, other values as 0 (summing together one hot encodings)

labels = labels.reset_index()
labels.head() # reindex this
labels.shape
# print(labels.columns)
labels['Other Type'] = labels['Other Type'].apply(lambda x: x.tolist()) # convert to list to be able to work with it later


labels.to_csv("../updated_multilabel_data/Other2.csv", index=False)
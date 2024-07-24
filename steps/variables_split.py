import numpy as np
from sklearn.model_selection import train_test_split


def train_test(dataset):

    '''Defining X, y variables and splitting data into train and test is needed to achieve better evaluation results'''

    #Defining variable X, y

    X = dataset.drop(['Date', 'RainTomorrow'], axis=1)
    y = dataset['RainTomorrow']

    #Spliting the dataset in Train and Test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    return X_train, X_test, y_train, y_test
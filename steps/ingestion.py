#Importing Libraries

import pandas as pd

#Creating a function in order to import the data to train the model

def call_df():

    '''This function will call the dataset and return a dataframe in order to foster the next steps'''


    'It will be made to call a CSV file using the pandas read_csv function as below'

    data = pd.read_csv('./weatherAUS.csv')

        #Drop na rows

    'dropna is a function that is used to delete all null rows in the target column'
    'In  this case ''RainTomorrow'' is our target'

    df = data.dropna(subset=['RainTomorrow'])

    return df

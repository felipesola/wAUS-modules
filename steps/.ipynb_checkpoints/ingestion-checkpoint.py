#Importing Libraries

import pandas as pd

#Import dataset function

def import_dataset():

    data = pd.read_csv('weatherAUS.csv')

    #Drop na rows

    df = data.dropna(subset=['RainTomorrow'])

    return df
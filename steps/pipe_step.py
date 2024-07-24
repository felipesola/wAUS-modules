from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def pipe():
    
    '''Pipe function is made on sklearn pipeline structure and basically all the additional data transformation needded will run here.'''
    '''The data is splited in numerical and categorical so it will be possible to provide proper destination as explained bellow'''
    
    ### NUMERICAL ###

    #Numerical Transfomation Pipelines
    num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',\
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',\
                'Temp9am', 'Temp3pm']

    #Using method SimpleImputer to fill the empty numerical spaces with the median
    num_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    ### CATEGORICAL ###

    #Categorical Transformation Pipelines
    cat_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    #Use OneHotEncoder to transform categorical data
    cat_transform = Pipeline(steps=[('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))])

    ### PREPROCESSOR ###

    #Creating a preprocessor with the pipelines above to be used on trainig step
    pipe_preprocessor = ColumnTransformer(transformers=[('num', num_transform, num_features), ('cat', cat_transform, cat_features)], remainder='passthrough')

    return pipe_preprocessor
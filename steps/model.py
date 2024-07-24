import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, \
                            recall_score, \
                            f1_score, \
                            precision_score, \
                            confusion_matrix, \
                            RocCurveDisplay, \
                            PrecisionRecallDisplay, \
                            ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn


def model_run(preprocessor, X_train_f, X_test_f, y_train_f, y_test_f):
    
    '''This is the function responsible for the model.'''
    '''It will run the model using sklearn library, start a 'MLflow run' and save all the data for evaluation purposes on MLflow'''

    #Setting MLflow experiment
    mlflow.set_experiment('weatherAUS')

    #Model parameters
    max_depth = 15
    random_state = 42
    min_sample_split = 150
    min_sample_leaf = 100
    criterion = 'gini'

    ######---------MLflow start run-------------##########

    with mlflow.start_run():
        
        #Model
        model_pipe = Pipeline(steps=[('preprocessor', preprocessor), \
                                ('tree', DecisionTreeClassifier(max_depth=max_depth, \
                                                            random_state=random_state, \
                                                            min_samples_split=min_sample_split, \
                                                            min_samples_leaf=min_sample_leaf, \
                                                            criterion=criterion
                                                            ))])
        #Creating parameters for MLflow
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('min_sample_split', min_sample_split)
        mlflow.log_param('min_sample_leaf', min_sample_leaf)
        mlflow.log_param('criterion', criterion)
        


        #Fitting model and predictor
        model_pipe.fit(X_train_f, y_train_f)
        predictor = model_pipe.predict(X_test_f)
        

        #Evaluation metrics
        model_score = model_pipe.score(X_train_f, y_train_f)
        acc_score = accuracy_score(y_test_f, predictor)
        prec_score = precision_score(y_test_f, predictor, average='macro')
        rec_score = recall_score(y_test_f, predictor, average='macro')
        f1_sco = f1_score(y_test_f, predictor, average='macro')
    
        
        #Metrics for MLflow
        mlflow.log_metric('model_score', model_score)
        mlflow.log_metric('accuracy_score', acc_score)
        mlflow.log_metric('precision_score', prec_score)
        mlflow.log_metric('recall_score', rec_score)
        mlflow.log_metric('f1_score', f1_sco)
        
        
        #Confusion Matrix
        
        #conf_matrix = confusion_matrix(y_test, predictor)
        conf_matrix_chart = ConfusionMatrixDisplay.from_estimator(model_pipe, X_test_f, y_test_f)
        conf_matrix_chart.figure_.savefig('./evaluation_images/Conf_Matrix.png') #save as png
        
        #Confusion Matrix MLflow artifact
        mlflow.log_artifact('./evaluation_images/Conf_Matrix.png')
        
        #Roc-auc curve chart
        roc = RocCurveDisplay.from_estimator(model_pipe, X_test_f, y_test_f)
        roc.figure_.savefig('./evaluation_images/ROC-Curve.png') #save chart as png
        
        #Roc-auc curve MLflow artifact
        mlflow.log_artifact("./evaluation_images/ROC-Curve.png")
        
        #Precision Recall chart
        precision_recall = PrecisionRecallDisplay.from_estimator(model_pipe, X_test_f, y_test_f, name="DecisionTree")
        precision_recall.figure_.savefig('./evaluation_images/Precision-Recall.png') #save chart as png
            
        #Precision Recall MLflow artifact
        mlflow.log_artifact("./evaluation_images/Precision-Recall.png")
        
        #Logging model to MLFlow
        mlflow.sklearn.log_model(model_pipe, 'model')

        return None
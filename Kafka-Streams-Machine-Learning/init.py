import pandas as pd;
from pathlib import Path

path = Path('C:/*/*/Machine-Learing-python-kafka-bigdata')

def download_data():
    print("downloading training data...");
    columns = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]
    # Train data
    df_train = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
                           names = columns, skipinitialspace=True, index_col=0)
    df_train.to_csv(path/'data/train/adult.data')
    df_train.to_csv(path/'data/train/train.csv')


    # Test data
    df_test = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
                           names=columns, skipinitialspace=True, index_col=0)
    df_test.to_csv(path/'data/test/test.data')
    df_test.to_csv(path/'data/test/test.csv')

download_data();
print('Exit.')
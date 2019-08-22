import pandas as pd;
from pathlib import Path;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.preprocessing import LabelEncoder;

path = Path('C:/*/Machine-Learing-python-kafka-bigdata')

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

# Pre process data
def pre_process_data(data_src):
    target = 'income_label'
    df = pd.read_csv(data_src)

    # Setting the target to 0 or 1 based on >50K or <=50K
    df[target] = (df['income_bracket'].apply(lambda x: '>50K' in x)).astype(int);

    # Droping the income bracket
    df.drop('income_bracket', axis=1, inplace=True)

    # Categorical columns
    cat_columns = list(df.select_dtypes(include=['object']).columns)

    # Numerical columns
    num_columns = [c for c in df.columns if c not in cat_columns] + [target];

    df_scaled_encoded = df.copy();

    # Scale numerical columns
    sc = MinMaxScaler();
    df_scaled_encoded = num_scale(df_scaled_encoded, num_columns, sc)

    le = LabelEncoder();
    # Encode categorical columns
    df_scaled_encoded = encode_cat_cols(df_scaled_encoded, cat_columns, le);

def num_scale(df_scaled, num_cols, sc):
    df_scaled[num_cols] = sc.fit_transform(df_scaled[num_cols])
    return df_scaled;

def encode_cat_cols(df, cat_cols, le):
    for col in cat_cols:
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df;

# download_data();
pre_process_data(path/'data/train/train.csv')
print('Exit.')
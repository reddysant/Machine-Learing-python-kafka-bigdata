import pandas as pd;
from pathlib import Path;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


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
    print('Pre processing data..')
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
    return df_scaled_encoded;

def num_scale(df_scaled, num_cols, sc):
    df_scaled[num_cols] = sc.fit_transform(df_scaled[num_cols])
    return df_scaled;

def encode_cat_cols(df, cat_cols, le):
    for col in cat_cols:
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df;

def create_model(df):
    X = df[['age', 'workclass', 'fnlwgt', 'education', 'education_num','marital_status', 'occupation', 'relationship', 'race', 'gender','capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]
    Y = df['income_label']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

    model = Sequential()
    model.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, activation='sigmoid'))

    lr = 0.01
    # _opt = SGD(lr)
    _opt = 'adam'
    _loss = 'binary_crossentropy'

    model.compile(loss=_loss, optimizer=_opt, metrics=['accuracy'])

    # Early stopping
    from keras.callbacks import EarlyStopping
    _es_monitor = 'val_loss'
    _es_patience = 50
    es = EarlyStopping(monitor=_es_monitor, mode='min', verbose=1, patience=_es_patience)

    # Model check point
    from keras.callbacks import ModelCheckpoint
    _mc_model_location = 'v1_model.h5'
    _mc_monitor = 'val_acc'
    mc = ModelCheckpoint(_mc_model_location, monitor=_mc_monitor, mode='max', verbose=1, save_best_only=True)
    _epochs = 1000
    history = model.fit(X_train.values, y_train.values, validation_split=0.20, epochs=_epochs,batch_size=32, verbose=1, callbacks=[es, mc])
    metrics = model.evaluate(X_train.values, y_train.values)
    print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1] * 100))


# download_data();
df = pre_process_data(path/'data/train/train.csv')
create_model(df)



print('Exit.')
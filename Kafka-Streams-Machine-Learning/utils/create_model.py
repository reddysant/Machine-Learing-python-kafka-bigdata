from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

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
    _mc_model_location = 'models/model.h5'
    _mc_monitor = 'val_acc'
    mc = ModelCheckpoint(_mc_model_location, monitor=_mc_monitor, mode='max', verbose=1, save_best_only=True)
    _epochs = 1000
    history = model.fit(X_train.values, y_train.values, validation_split=0.20, epochs=_epochs,batch_size=32, verbose=1, callbacks=[es, mc])
    metrics = model.evaluate(X_train.values, y_train.values)
    print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1] * 100))

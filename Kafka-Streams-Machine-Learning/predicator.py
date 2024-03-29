from pathlib import Path
from kafka import KafkaProducer
from keras.models import load_model
from utils.pre_processor import pre_process_data
import numpy as np
from time import sleep
from json import dumps

topic = 'GET_PREDECTIONS'
KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages', 'retrain_topic']
PATH = Path('C:/Santhosh/AIML/Github/Machine-Learing-python-kafka-bigdata/')
MODEL_PATH = PATH/'Kafka-Streams-Machine-Learning/models/model.h5'

def reload_model(path):
	print(path)
	return load_model(path)

def predict(model, row):
	# Convert to 2 dimentional if it 1D
	if (row.ndim == 1):
		row = np.array([row])

	# predict(model,single_row);
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	result = model.predict(row)
	return result;


model = reload_model(str(MODEL_PATH));
df_test = pre_process_data(PATH/'data/test/test.csv')

df_train_x = df_test.drop(['income_label'], axis=1);
df_train_y = df_test[['income_label']];

test_data_size = df_train_x.shape[0]
print(test_data_size)

for i in range(test_data_size):
	row = np.array(list(df_train_x.iloc[i, 0:14].values))
	result = predict(model, row)
	print("Predicted value: ", result);
	producer = KafkaProducer(bootstrap_servers='localhost:9092')
	producer.send('PREDICTED_VALUE', bytes(str(result[0][0]), 'utf-8'))
	sleep(2)

print('Exit')
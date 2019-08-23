import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer
from keras.models import load_model


KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages', 'retrain_topic']
PATH = Path('C:/Santhosh/AIML/Github/Machine-Learing-python-kafka-bigdata/')
MODEL_PATH = PATH/'Kafka-Streams-Machine-Learning/models/model.h5'

def reload_model(path):
	print(path)
	return load_model(path)


def predict(m, row):

	print('exit')

model = reload_model(str(MODEL_PATH));
testData = pd.read_csv(PATH/'data/test/test.csv');



# predict(model, row);
print('exit')
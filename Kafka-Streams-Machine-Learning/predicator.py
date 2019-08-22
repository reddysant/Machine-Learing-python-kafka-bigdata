import json
import pandas as pd
import pickle

from pathlib import Path
from kafka import KafkaConsumer
from keras.models import load_model


KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages', 'retrain_topic']
PATH = Path('/')
MODEL_PATH = PATH/'models/model.h5'

def reload_model(path):
	print(path)
	return load_model(path)

model = reload_model(str(MODEL_PATH));

print('exit')
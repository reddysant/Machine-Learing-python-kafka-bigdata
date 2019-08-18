from kafka import KafkaConsumer
# from pymongo import MongoClient
from json import loads

# Create Kafka consumer
# Connecting server
# subscribing to 'numtest' topic
# auto_offset_rest - Where consumer restarts after break down or shutdown (earlist or latest)
# value_deserializer - deserializer

consumer = KafkaConsumer(
    'numtest',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='counters',value_deserializer=lambda x: loads(x.decode('utf-8')))
for message in consumer:
    message = message.value
    print('Received : ', message)


print('exit.')
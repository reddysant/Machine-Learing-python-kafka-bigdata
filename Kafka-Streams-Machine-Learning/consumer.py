from kafka import KafkaConsumer
from json import loads

# topic = 'GET_PREDECTIONS'
# KAFKA_HOST = 'localhost:9092'
# consumer = KafkaConsumer('numtest',bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: loads(x.decode('utf-8')))
# for message in consumer:
#     message = message.value
#     print('Received : ', message);
# print('exit.')

consumer = KafkaConsumer(
    'numtest',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='counters',value_deserializer=lambda x: loads(x.decode('utf-8')))
for message in consumer:
    print('bduhrfeveguybr')
    message = message.value
    print('Received : ', message)


print('exit.')
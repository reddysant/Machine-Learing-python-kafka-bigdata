from time import sleep
from json import dumps
from kafka import KafkaProducer

# Create kafka producer and connect to server
# value_serializer - Will serialize before senidng the data
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

# Send data
for e in range(1000):
    data = {'number' : e}
    producer.send('numtest', value=data)
    print(e, ' sent.')
    sleep(5)

print('Exit')
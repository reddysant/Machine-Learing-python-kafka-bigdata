from kafka import KafkaConsumer

KAFKA_HOST = 'localhost:9092'

consumer = KafkaConsumer('PREDICTED_VALUE', bootstrap_servers=[KAFKA_HOST])
for msg in consumer:
    print((msg.value).decode("utf-8"))
print('exit.')
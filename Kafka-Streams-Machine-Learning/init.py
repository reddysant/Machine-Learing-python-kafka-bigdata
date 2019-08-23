
from pathlib import Path;

from utils.pre_processor import pre_process_data
from utils.create_model import create_model
from utils.download_data import download_data


path = Path('C:/Santhosh/AIML/Github/Machine-Learing-python-kafka-bigdata')

download_data(path);
df = pre_process_data(path/'data/train/train.csv')
create_model(df)
print('Exit.')
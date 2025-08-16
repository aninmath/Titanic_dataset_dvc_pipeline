import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
import yaml



log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)


def load_data (data_url:str ) -> pd.DataFrame :
    """load the data from URL"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df

    except  pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise 
    
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data (df:pd.DataFrame ) -> pd.DataFrame :
    try:
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace = True)
        logger.debug('Data preprocessing completed')
        return df

    except  KeyError as e:
        logger.error('Missing column in dataframe: %s', e)
        raise 
    
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing %s', e)
        raise


def save_data(train_data: np.ndarray,test_data:np.ndarray ,data_path: str) -> None:
    """save data"""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"), index = False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"), index = False)
        logger.debug("train test file saved")

    except Exception as e:
        logger.error("error happened %s",e)
        raise




def main():
    try:
        data_path = 'https://raw.githubusercontent.com/aninmath/Titanic_dataset_dvc_pipeline/refs/heads/main/experiments/Titanic-Dataset.csv'
        df = load_data(data_path)
        logger.debug('Data loaded from from main %s', data_path)
        final_df = preprocess_data(df)

        param = load_params('params.yaml')
        test_size = param['data load']['test_size']

        train_data, test_data = train_test_split(final_df, test_size= 0.2, random_state= 42)
        save_data(train_data,test_data, './data')

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        

if __name__ == '__main__':
    main()



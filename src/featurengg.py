import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_csv(path:str):

    try:
        df_train = pd.read_csv(os.path.join(path,'train.csv'))
        df_test = pd.read_csv(os.path.join(path,'test.csv'))
        logger.debug ('data loaded from train and test')
       
        return (df_train, df_test)

    except Exception as e:
        logger.error('Error during reading csv %s',e)
        raise




def create_pipeline(df_train, df_test):

    try:

        categorical_features = ['Sex', 'Embarked', 'Pclass']
        numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

        numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()) 
            ])
    # Combine transformers into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features), 
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any other columns not specified
        )

        x_train = df_train.iloc[:,1:]
        x_test  = df_test.iloc[:,1:]
        

        preprocessor.fit(x_train)

        train = pd.DataFrame(preprocessor.transform(x_train))
        train['label'] = df_train.Survived.values


        test = pd.DataFrame(preprocessor.transform(x_test))
        test['label'] = df_test.Survived.values

        logger.debug ('train and test with feature engg is done')


        return (train,test)


    except KeyError as e:
        logger.error("error during train test split %s",e)
        raise





def save_data(train_data: np.ndarray,test_data:np.ndarray ,data_path: str) -> None:
    """save data"""
    try:
        interim_data_path = os.path.join(data_path,'interim')
        os.makedirs(interim_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path,"train.csv"), index = False)
        test_data.to_csv(os.path.join(interim_data_path,"test.csv"), index = False)
        logger.debug("train test file with feature engg is saved")

    except Exception as e:
        logger.error("error happened %s",e)
        raise    


def main():

    try: 
        data_path = os.path.join('data','raw')
        df_train, df_test = load_csv(data_path)
        train , test = create_pipeline (df_train,df_test)
        save_data(train,test, './data')
        
    except Exception as e:
        logger.error("problem in feature engg %s",e)
        raise
    


if __name__ == '__main__':
    main()

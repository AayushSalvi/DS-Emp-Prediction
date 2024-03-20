import os
import sys
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    '''
    this fucntion is to reduce the number of countries in the data 
    they are mapped to others
    '''
    def shorten_category(self,categories, cutoff):
        self.categorical_map = {}
        for i in range(len(categories)):
            if categories.values[i] >= cutoff:
                self.categorical_map[categories.index[i]] = categories.index[i]
            else:
                self.categorical_map[categories.index[i]] = 'Other'
        return self.categorical_map

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\ds_sal_data.csv')
            logging.info('Read the dataset as dataframe')

            country_map = self.shorten_category(df.company_location.value_counts() , 60)
            df['company_location'] = df['company_location'].map(country_map)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df = df[['job_title','job_category','employee_residence','experience_level','employment_type','company_location','company_size','salary_in_usd']]
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



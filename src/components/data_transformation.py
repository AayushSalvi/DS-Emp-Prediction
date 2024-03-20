import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
import os

from src.pipeline.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config = DataTransformationConfig()

    

    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation 
        """
        try:
            cat_columns = ['job_title','job_category','employee_residence','experience_level','employment_type','company_location','company_size']
            label_transformer = OrdinalEncoder()

            logging.info(f"Categorical columns: {cat_columns}")
            preprocessor=ColumnTransformer(
                [
                ("OrdinalEncoder",label_transformer,cat_columns)
                ],remainder='passthrough'
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="salary_in_usd"
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


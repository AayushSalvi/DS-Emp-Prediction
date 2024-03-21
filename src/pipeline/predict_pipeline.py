import sys
import pandas as pd
from exception import CustomException
from utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


['job_title','job_category','employee_residence','experience_level','employment_type','company_location','company_size']
class CustomData:
    def __init__(  self,
        job_title: str,
        job_category: str,
        employee_residence:str,
        experience_level: str,
        employment_type: str,
        company_location: str,
        company_size: str):

        self.job_title = job_title

        self.job_category = job_category

        self.employee_residence = employee_residence

        self.experience_level = experience_level

        self.employment_type = employment_type

        self.company_location = company_location

        self.company_size = company_size

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

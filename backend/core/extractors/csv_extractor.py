import pandas as pd
from pathlib import Path
from core.models import ExtractedPage, PageMetadata
from core.extractors.base_extractor import BaseExtractor


class CSVExtractor:


    def _load_file(self, file_path:str):
        try:
            df = pd.read_csv(file_path,low_memory=False)
            return df
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding = "latin-1",low_memory=False)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

    def _generate_schema(self , df, file_path):
        meta = {
            "file_name":file_path,
            "total_row_count": len(df),
            "total_column_count":df.shape[1],
            "data":[]
        }

        
        for col_name in df:
            column_meta = {}
            column_meta["column_name"]= col_name
            column_meta["dtype"]= str(df[col_name].dtype)
            column_meta["null_count"]= int(df[col_name].isna().sum())
            column_meta["first_3_columns"]=df[col_name].dropna().head(3).tolist()

            if pd.api.types.is_numeric_dtype(df[col_name]):
                column_meta["minimum_value"]=float=(df[col_name].min())
                column_meta["maximum_value"]=float(df[col_name].max())
                column_meta["mean"]=float(df[col_name].mean())
            else:
                column_meta["unique_values_count"]=df[col_name].nunique()
                if column_meta["unique_values_count"] < 10:
                    column_meta["unique_values"]=df[col_name].unique().tolist()
            meta["data"].append(column_meta)
        return meta 
                    

            

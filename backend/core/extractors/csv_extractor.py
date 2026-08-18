import pandas as pd
from pathlib import Path
from core.models import ExtractedPage, PageMetadata
from core.extractors.base_extractor import BaseExtractor


class CSVExtractor:

    def extract(self, file_path:str) ->list[ExtractedPage]:
        df = self._load(file_path)

        schema_dict = self._generate_schema(df, file_path)
        schema_text = self._format_schema_to_text(schema_dict)
        metadata = PageMetadata(
            source=file_path,
            file_type="csv"
        )
        page = ExtractedPage(
            text=schema_text,
            metadata = metadata
        )

        return [page]


    def _load(self, file_path:str):
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
                column_meta["minimum_value"]=float(df[col_name].min())
                column_meta["maximum_value"]=float(df[col_name].max())
                column_meta["mean"]=float(df[col_name].mean())
            else:
                column_meta["unique_values_counts"]=df[col_name].nunique()
                if column_meta["unique_values_counts"] < 10:
                    column_meta["unique_values"]=df[col_name].unique().tolist()
            meta["data"].append(column_meta)
        return meta 
                    
    
    def _format_schema_to_text(self, meta):
        lines = []
       
        lines.append(f"File: {meta['file_name']} | Rows:{meta['total_row_count']} | Cols:{meta['total_column_count']}")
        for data in meta["data"]:
            lines.append(f" Col_name: {data['column_name']} | dtype: {data['dtype']} | Null_count: {data['null_count']} | First 3 Columns:{data['first_3_columns']}")
            if "minimum_value" in data:
                lines.append(f" Minimum Value: {data['minimum_value']} | Maximum Value: {data['maximum_value']} | Mean: {data['mean']}")
            else:
                lines.append(f"Unique Values Count:{data['unique_values_counts']}")
        return "\n".join(lines)
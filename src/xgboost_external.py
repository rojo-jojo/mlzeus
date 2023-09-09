import xgboost
import boto3
from typing import List
import pandas as pd
from sklearn.datasets import dump_svmlight_file
import pyarrow.parquet as pq

s3 = boto3.client("s3")

def get_s3_filepaths(bucket:str, prefix:str, extension:str='.csv') -> List[str]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix =prefix)
    return [obj['Key'] for obj in response['Contents'] if extension in obj['Key']]



def df_to_libsvm(df:pd.DataFrame, outfilename:str):
    x = df.drop('label', axis=1)
    y = df['label']
    dump_svmlight_file(X=x, y=y, f=f'{outfilename}.dat', zero_based=True)
import xgboost
import boto3
import os
import logging
import glob
import pandas as pd
import pyarrow.parquet as pq
from sklearn.datasets import dump_svmlight_file
from src.xgboost_extend.xgboost_iterator import Iterator
from typing import List
from tqdm import tqdm

logging.basicConfig()
s3 = boto3.client("s3")
extension = '.parquet'


def get_s3_files(bucket:str, prefix:str, extension:str='.csv', download_path:str) -> List[str]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix =prefix)
    list_of_keys = [obj['Key'] for obj in response['Contents'] if extension in obj['Key']]
    logging.info("Downloading files from S3")
    for key in tqdm(list_of_keys):
       fname = key.split('/')[-1]
       write_path = os.path.join(download_path,fname)
       s3.download_file(bucket, key, write_path)

def df_to_libsvm(df:pd.DataFrame, **kwargs):
    outfilename = kwargs['outfilename']
    label = kwargs['label']
    feature_names = kwargs['feature_names']
    x = df.drop(label, axis=1)
    x = x[feature_names]
    y = df[label]
    dump_svmlight_file(X=x, y=y, f=f'{outfilename}', zero_based=True)

def load_data(data_paths:List[str], output_path:str, label:str, feature_names:str) -> List[str]:
    '''Loads csv/parquet data from disk writes them as libsvm and returns
    list of paths for libsvm files'''
    logging.info(f"Total training files: {str(len(data_paths))}")
    libsvm_paths = []
    logging.info(f"Converting {extension} to libsvm...")
    for i,f in tqdm(enumerate(data_paths)):
        libsvm_outfile = os.path.join(output_path, f'file_{i}.dat')
        #if not os.path.isfile(libsvm_outfile):
        df_iter = pd.read_parquet(f, engine='pyarrow')
        df_to_libsvm(
        df_iter, 
        outfilename=libsvm_outfile,
        label=label,
        feature_names=feature_names)
        libsvm_paths.append(libsvm_outfile)
    return libsvm_paths



if __name__ == '__main__':
    # User inputs
    #
    bucket = 'bucketname'
    prefix = 'prefix/to/csv/parquet/'
    input_path = 'inputs'
    output_path = 'outputs'
    feature_names = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 
                'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 
                'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 
                'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 
                'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 
                'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 
                'f_49', 'f_50']
    xgb_hyperparameter = {"tree_method": "approx"}


    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    get_s3_files(bucket, prefix, input_path)
    data_paths = glob.glob(os.path.join(input_path,f'*{extension}'))
    libsvm_paths = load_data(data_paths, output_path, 'target', feature_names)
    it = Iterator(libsvm_paths,feature_names)
    Xy = xgboost.DMatrix(it)
    # Other tree methods including ``hist`` and ``gpu_hist`` also work, but has some caveats
    # as noted in following sections.
    logging.info(f"Starting iterative model training...")
    booster = xgboost.train(xgb_hyperparameter, Xy)
    saved_model = os.path.join(output_path, 'modelfile.json')
    booster.save_model(saved_model)
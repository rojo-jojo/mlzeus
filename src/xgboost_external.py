import xgboost
import boto3
import os
import logging
import glob
import pandas as pd
import pyarrow.parquet as pq
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from typing import List, Callable
from tqdm import tqdm

logging.basicConfig()
training_cols = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 
                     'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 
                     'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 
                     'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 
                     'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 
                     'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 
                     'f_49', 'f_50']
s3 = boto3.client("s3")
extension = '.parquet'
def get_s3_filepaths(bucket:str, prefix:str, extension:str='.csv') -> List[str]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix =prefix)
    return [obj['Key'] for obj in response['Contents'] if extension in obj['Key']]



def df_to_libsvm(df:pd.DataFrame, **kwargs):
    outfilename = kwargs['outfilename']
    label = kwargs['label']
    training_cols = kwargs['training_cols']
    x = df.drop(label, axis=1)
    x = x[training_cols]
    y = df[label]
    dump_svmlight_file(X=x, y=y, f=f'{outfilename}', zero_based=True)



class Iterator(xgboost.DataIter):
  def __init__(self, svm_file_paths: List[str]):
    self._file_paths = svm_file_paths
    self._it = 0
    # XGBoost will generate some cache files under current directory with the prefix
    # "cache"
    super().__init__(cache_prefix=os.path.join(".", "cache"))

  def next(self, input_data: Callable):
    """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
    called by XGBoost during the construction of ``DMatrix``

    """
    if self._it == len(self._file_paths):
      # return 0 to let XGBoost know this is the end of iteration
      return 0

    # input_data is a function passed in by XGBoost who has the exact same signature of
    # ``DMatrix``
    X, y = load_svmlight_file(self._file_paths[self._it])
    kwargs = {'feature_names': training_cols}
    input_data(X, y, **kwargs)
    self._it += 1
    # Return 1 to let XGBoost know we haven't seen all the files yet.
    return 1

  def reset(self):
    """Reset the iterator to its beginning"""
    self._it = 0

if __name__ == '__main__':
    
    input_path = 'inputs'
    output_path = 'outputs'
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_paths = glob.glob(os.path.join(input_path,f'*{extension}'))
    logging.info(f"Total training files: {str(len(data_paths))}")
    libsvm_paths = []
    logging.info(f"Converting {extension} to libsvm...")
    for i,f in tqdm(enumerate(data_paths)):
        libsvm_outfile = os.path.join(output_path, f'file_{i}.dat')
        if not os.path.isfile(libsvm_outfile):
            df_iter = pd.read_parquet(f, engine='pyarrow')
            df_to_libsvm(
            df_iter, 
            outfilename=libsvm_outfile,
            label='target',
            training_cols=training_cols)
        libsvm_paths.append(libsvm_outfile)
    it = Iterator(libsvm_paths)
    Xy = xgboost.DMatrix(data=it,feature_names=training_cols)

    # Other tree methods including ``hist`` and ``gpu_hist`` also work, but has some caveats
    # as noted in following sections.
    logging.info(f"Starting iterative model training...")
    booster = xgboost.train({"tree_method": "approx"}, Xy)
    saved_model = os.path.join(output_path, 'modelfile.json')
    booster.save_model('tmp/model-object.json')
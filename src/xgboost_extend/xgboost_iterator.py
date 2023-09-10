import xgboost
from typing import List, Callable
from sklearn.datasets import load_svmlight_file

class Iterator(xgboost.DataIter):
  def __init__(self, svm_file_paths: List[str], feature_names: List[str]):
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
    # kwargs = {'feature_names': training_cols}
    input_data(data=X, label=y, feature_names=self.feature_names)
    self._it += 1
    # Return 1 to let XGBoost know we haven't seen all the files yet.
    return 1

  def reset(self):
    """Reset the iterator to its beginning"""
    self._it = 0
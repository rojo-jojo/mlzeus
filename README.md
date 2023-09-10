# mlzeus
XGBoost came up with external memory training in version 1.7.2. This allows you to load data from disk in batches and train iteratively on your ram.

mlzeus extends the functionality to use pandas friendly tabular file types like parquet and csv for training. The original code on xgboost documentation only supports libsvm file type.

It also has a function to get batch of files AWS S3 and train on them

To run on a unix based system

```
git clone https://github.com/rojo-jojo/mlzeus.git
cd mlzeus
sh buildenv.sh
python3 src/xgboost_extend/xgboost_extend.py
```
Saved model json available in `inputs` folder

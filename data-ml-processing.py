# processing needed to handle categorical variables in ML
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data import - note data was saved locally via kaggle API download
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

# permanently remove id col
data_train.drop('id', inplace=True, axis=1)
data_test.drop('id', inplace=True, axis=1)

# select categorical col to encode
categorical_cols = data_train.select_dtypes(include='object')

# review all categories inform which type of coding to use
for col in categorical_cols:
    categories = sorted(set(data_train[col]))
    print(categories)

# none of the categorical variables are dichotomous 

# grade-subgrade can be treated as ordinal
# create dictionary of categories and numeric values
grade_subgrades = sorted(set(data_train['grade_subgrade']))
grade_subgrades_order = np.arange(0,len(grade_subgrades)).tolist()
grade_subgrades_dict = dict(zip(grade_subgrades, grade_subgrades_order))
# use dictionary to execute ordinal recode in train data
data_train_encoded = data_train.copy()
data_train_encoded['grade_subgrade'] = data_train_encoded[
    'grade_subgrade'].replace(grade_subgrades_dict)
# repeat for test data
data_test_encoded = data_test.copy()
data_test_encoded['grade_subgrade'] = data_test_encoded[
    'grade_subgrade'].replace(grade_subgrades_dict)

# the other categorical cols will undergo one-hot encoding  
one_hot_encoding_cols = categorical_cols.drop('grade_subgrade', axis=1)

# function to execute one-hot encoding
def one_hot_encode(data_ordinal_encoded, columns_to_encode):
    # df to save results form loop below
    data_one_hot_encoded = data_ordinal_encoded.copy()
    
    for col in columns_to_encode:
        # # identify variable to encode
        col_multi_categorical = data_ordinal_encoded[col]
        # use get_dummies method for one-hot encoding
        cols_coded = pd.get_dummies(col_multi_categorical, prefix=col)
        # join encoded data to initial data frame
        data_one_hot_encoded = pd.concat([data_one_hot_encoded, cols_coded], 
                                         axis=1)
        # drop multi categorical col, for we have new individual cols for every category
        data_one_hot_encoded.drop([col], axis=1, inplace=True)
    
    return data_one_hot_encoded

# deploy
data_train_encoded = one_hot_encode(data_train_encoded, one_hot_encoding_cols)
data_test_encoded = one_hot_encode(data_test_encoded, one_hot_encoding_cols)

# write data
data_train_encoded.to_csv('./data/data_train_processed.csv', index=False)
data_test_encoded.to_csv('./data/data_test_processed.csv', index=False)

# move on to modelling
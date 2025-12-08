# set up
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ml methods
from sklearn.model_selection import train_test_split, cross_validate, \
    cross_val_predict, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve, RocCurveDisplay, f1_score, \
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, precision_recall_fscore_support

data_train_encoded = pd.read_csv('./data/data_train_processed.csv')
# the data test provided by kaggle doesn't have labels
# so data for splitting, model training, and testing will come from 'data_train'
data = data_train_encoded.copy()
# check encoded version of data is what has been uploaded
sorted(data.columns)

# separate features and labels (outcome) 
X = data_train_encoded.drop('loan_paid_back', axis=1)
y = data_train_encoded['loan_paid_back']
# what's the overall split of labels in train data
y.value_counts(normalize=True) # 79.9% were payers vs 20.1% non-payers
#TODO: account for imbalanced classes

# data splits: training, validation, and testing 
# (80% for training/validation, 20% for testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=17)
# (80% for training, 20% for validation)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val,
                                                            test_size=0.2,
                                                            random_state=17)

# function to standardise data (mean = 0, stdev = 1)
def standardise_data(X_train, X_validate, X_test):
    # intialise scaling object 
    sc = StandardScaler()
    # set on the training set
    sc.fit(X_train)
    # apply scaler
    train_std = sc.fit_transform(X_train)
    validate_std = sc.fit_transform(X_validate)
    test_std = sc.fit_transform(X_test)
    return train_std, validate_std, test_std

X_train_std, X_validate_std, X_test_std = standardise_data(X_train, 
                                                           X_validate, X_test)

# start modelling
# approach 1 - logistic regressions
model = LogisticRegression()

# defaulted to stratified cross validation (cv) - i.e., percentage of sample 
# in each label is kept consistent as possible
cv_results = cross_validate(
    model, X_train_std, y_train,
    scoring=['accuracy', 'f1', 'roc_auc', 'precision_macro', 'recall_macro'],
    n_jobs=-1,
    # number of folds
    cv=10,
    # returns trained models for each fold
    # but increases running time a little
    return_estimator=True
    )

# extract coefficients from models generated in cross val
co_eff_df = pd.DataFrame()
co_eff_df['feature'] = list(X_train_val)

coefficients = []
for model in cv_results['estimator']:
    # append array of coefficients
    coefficients.append(model.coef_[0])

# calculate mean coefficient for every features
mean_coefficients = np.mean(coefficients, axis = 0)
# save into coeff df
co_eff_df['mean_coefficient_cv10'] = mean_coefficients
# sort by absolute value (derived from mean coeff to understand magnitude)
co_eff_df['co_efficient_abs'] = np.abs(co_eff_df['mean_coefficient_cv10'])
co_eff_df.sort_values(by='co_efficient_abs', ascending=False, inplace=True)
co_eff_df

# generate classification report
# we'll be plotting as well 
pd.DataFrame(
    classification_report(
        y_true=y_train,
        y_pred=predictions, 
        target_names=['Non-Payer', 'Payer'],
        output_dict=True)
        )

# model was trained on 9 folds and tested on 1
performance_metrics = pd.DataFrame(cv_results).drop(
    columns=['fit_time', 'score_time', 'estimator'])

# plot metrics obtained via stratified kfolds
# the test_ metrics refer to the the 10th fold 
plt.subplots(figsize=(8,6))
plt.boxplot(performance_metrics, labels=performance_metrics.columns)
plt.tight_layout()
plt.show()

# predictions and confusion matrices
predictions = cross_val_predict(
    model,X_train_std,y_train,
    cv=10
)

cm = confusion_matrix(y_true=y_train, 
                      y_pred=predictions,
                      normalize='true')

cv_confusion_matrix = ConfusionMatrixDisplay(cm, 
                                             display_labels=['Non-payer','Payer'])

cv_confusion_matrix.plot()
ax = cv_confusion_matrix.ax_
ax.set_title('10-fold cross validation')
plt.show()

#############
#############




# receiver operator characteristic curve
# plot specificity (false positive rate) vs recall (aka sensitivity; true positive rate) 
# HMMM BUT CHAT IF I WANT TO PLOT BASED ON NEGATIVE RATES?
roc_curve = RocCurveDisplay.from_estimator(model, X_train_std, y_train_val)
fig = roc_curve.figure_
ax = roc_curve.ax_
# plot chance
ax.plot([0,1], [0,1], color='darkblue', linestyle=':')

# this is deploying the model on test data, asusming we're happy with model performance
# here are the predicted probabilites of the model
# on the standaridised, test data
y_pred_test = model.predict(X_test_std)
pred_probabilities = model.predict_proba(X_test_std)
submission_probabilities = pd.DataFrame()
data_test = pd.read_csv('data/test.csv')
submission_probabilities['id'] = data_test['id']
# prob of paying (label = 1)
submission_probabilities['pred_prob_payback'] = pred_probabilities[:, 1]

# set up
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

# data import - note data was saved locally via kaggle API download
data_train = pd.read_csv('data/train.csv')

# data exploration (train data)
data_train.columns.to_list()
data_train.info() # no missing values in any columns 
# permanently remove id col
data_train.drop('id', inplace=True, axis=1)

# exploratory data analysis - numeric variables
# drop dependent variable (as it's been coded as 1 and 0)
data_train.describe().drop('loan_paid_back', axis=1)

# though description above is useful, 
# it would be more informative to see description by outcome group
# even better if we could visualise it

# create df's by payback group 
filter = data_train['loan_paid_back'] == 1
data_train_payers = data_train[filter]
filter = data_train['loan_paid_back'] == 0
data_train_non_payers = data_train[filter]

# TODO: SEARCH if there's a package that already does below

# save description objects of numeric variables (by outcome group)
payers_desc = data_train_payers.describe().drop('loan_paid_back', axis=1)
non_payers_desc = data_train_non_payers.describe().drop('loan_paid_back', axis=1)

# visualisation of numeric variables

# define cols to loop through in plot
numeric_cols = data_train_payers.select_dtypes(
    include=np.number).columns.drop('loan_paid_back')
# determine grid size
num_cols = len(numeric_cols)
ncols = 2
nrows = int(np.ceil(num_cols/2)) # in case of single row/col

# define plot objects
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*2))
axes = axes.flatten()

# make boxplots
for i, col in enumerate(numeric_cols):
    ax = axes[i]

    # create boxplot for col
    bp = ax.boxplot(
        [data_train_payers[col], data_train_non_payers[col]],
        labels=['Payers', 'Non-Payers'],
        # allow custom colours 
        patch_artist=True,
        # horizontal orientation of bars
        vert=False,
        # make whiskers represent min and max values
        whis = (0, 100)
        )
    
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp['medians']:
        median.set(color='black')
    
    ax.set_title(f"Distribution of {col} by outcome group")

#hide unused subplots
for i in range(num_cols, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# alternatively, these variables can also be visualised as histograms

fig, axes = plt.subplots(nrows,ncols, figsize=(12, nrows*2))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]

    # create histograms for each group
    ax.hist(
        data_train_payers[col], bins=50, #alpha=0.5, 
        color='blue',
        label='Payers', 
        histtype = 'step'
        )
        
    ax.hist(
        data_train_non_payers[col], bins=20, #alpha=0.5, 
        color='red',
        label='Non-payers', histtype = 'step'
        )
    
    ax.set_title(f"Distribution of {col} by outcome group")
    ax.legend()

#hide unused subplots
for i in range(num_cols, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# visualisation of categorical variables

# recode outcome variable
recode_dict = {1:'Payers', 0:'Non-Payers'}
data_train['loan_paid_back_recoded'] =  data_train['loan_paid_back'].map(recode_dict)

# generate frequency tables by outcome group
def generate_freq_table(data, col_name):
    # frequencies of categorical vol by outcome 
    count_table = pd.crosstab(data[col_name], data['loan_paid_back_recoded'])
    # reset index so categorical variable is not kept as index
    count_table = count_table.reset_index()
    # calculate percentages by outcome group
    percent_payers = (
        count_table['Payers']/count_table['Payers'].sum()*100).round(1)
    percent_non_payers = (
        count_table['Non-Payers']/count_table['Non-Payers'].sum()*100).round(1)
    count_table['percent_payers'] = percent_payers
    count_table['percent_non_payers'] = percent_non_payers
    # sort values by count of payers 
    frequency_table = count_table.sort_values(by='Payers', ascending=False)
    return frequency_table

# function to plot frequencies by group
def plot_categorical_cols(freq_table, col_name):
    # drop count columns, as the plot is for percentages
    freq_table_tidy = freq_table.drop(['Payers', 'Non-Payers'], axis=1)
    freq_table_tidy.plot(x = col_name, kind='barh', color=['blue', 'red'])
    # axis labels
    plt.xlabel('Percent')
    plt.gca().xaxis.set_major_formatter(
        # data already scalled to percent, 100
        PercentFormatter(100))
    plt.ylabel('')
    # show bars in desc order
    plt.gca().invert_yaxis()
    plt.title(f"Breakdown of outcome group by \n{col_name}")
    custom_labels = ['Payers', 'Non-Payers']
    plt.legend(loc='lower right', frameon = False, labels=custom_labels)
    plot = plt.show()
    return plot

#plot_categorical_cols(freq_table, 'grade_subgrade')

# loop over to see freq tables and plots
categorical_cols = data_train.select_dtypes(include='object').drop(
    'loan_paid_back_recoded', axis=1)

for col in categorical_cols:
    print(f"\nBreakdown of outcome group by \n{col}")
    freq_table = generate_freq_table(data_train, col)
    print(freq_table)
    plot = plot_categorical_cols(freq_table, col)
    print(plot)

# move on to data pre-processing
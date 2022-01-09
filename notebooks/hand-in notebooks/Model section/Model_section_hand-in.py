# %% [markdown]
# # Considerations for this section
# 
# - Before building a classification model, let’s build a Dummy Classifier to determine the ‘baseline’ performance. This answers the question — ‘What would be the success rate of the model, if one were simply guessing?’ The dummy classifier we are using will simply predict the majority class.
# - Determine differences between oversampling and undersampling
# - Explain the evaluation metric we chose (A false negative is more serious as a disease has been ignored, which can lead to the death of the patient. At the same time, a false positive would lead to an unnecessary treatment — incurring additional cost.)
# - Report the results and measurements in a good way
# - Explain the model tuning we did (GridSearch e.g.)
# - Compare model before and after tuning

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
df = pd.read_csv("..\\..\\data\\healthcare-dataset-stroke-data.csv")

# %%
# Fill missing values
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DecisionTreePip = Pipeline(steps=[ 
                               ('Scale',StandardScaler()),
                               ('DecisionTreeReg',DecisionTreeRegressor(random_state = 42))
                              ])

X = df[['age','gender','bmi']]
X.gender = X.gender.replace({'Male' : 0, 'Female' : 1 , 'Other' : -1}).astype(np.uint8)

# create a dataframe containing the missing values of X
missing = X[X.bmi.isna()]

# remove the missing values from X 
X = X.dropna()

# creates Y by removing bmi from X
Y = X.pop('bmi')

# fit the pipeline
DecisionTreePip.fit(X,Y)

# make the prediction 
predict_bmi = pd.Series(DecisionTreePip.predict(missing[['age', 'gender']]), index = missing.index)
df.loc[missing.index, 'bmi'] = predict_bmi

# %%
# Generate the data samples: Original, Oversampled, Undersampled
y = df[["stroke"]].copy()
X = df.copy()

# %%
# ALWAYS SPLIT THE DATA IN TRAIN AND TEST AND THEN OVERSAMPLE/DOWNSAMPLE
# see here: https://stackoverflow.com/questions/48805063/balance-classes-in-cross-validation/48810493#48810493 
from sklearn.model_selection import train_test_split
# Split data
X_raw_train, X_test, y_raw_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3, shuffle=True) # Set shuffle to true to have data of both labels in test set

# %%
# The oversampled/undersampled data will be used for training ONLY! not for testing

from src.Resample import undersample_kmeans
from src.Resample import oversample

X_raw_train.drop(["id", "stroke"], inplace=True, axis=1)
X_test.drop(["id", "stroke"], inplace=True, axis=1)

df_input = X_raw_train.copy()
df_input["stroke"] = y_raw_train

# Get over and undersampled data
undersampled = undersample_kmeans(df_input)
oversampled = oversample(df_input)

y_over_train = oversampled[["stroke"]].copy()
y_under_train = undersampled[["stroke"]].copy()

oversampled.drop(columns=["stroke"], inplace=True)
undersampled.drop(columns=["stroke"], inplace=True)

X_over_train = oversampled.copy()
X_under_train = undersampled.copy()
X_under_train = X_under_train[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]] # Only do this because column order must be same as X_test column order!

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


num_pipe = Pipeline([
    #('impute', SimpleImputer(strategy='median', copy=False)),
    #('minmax_scaler', MinMaxScaler(copy=False))
    #('standard_scaler', StandardScaler(copy=False))
    ('robust_scaler', RobustScaler(copy=False))
])

ordinal_pipe = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

binary_pipe = Pipeline([
    ('label_encoder', OrdinalEncoder()),
])

# two pipelines combined in the column transformer
full_transform = ColumnTransformer([
    ("num", num_pipe, ["age", "avg_glucose_level", "bmi"]),
    ("ord", ordinal_pipe, ["gender", "work_type", "smoking_status"]),
    ("binary", binary_pipe, ["ever_married", "Residence_type"]),
])

full_pipeline_with_model = Pipeline([
    ('trf', full_transform),
    ('svm', SVC(C= 25, class_weight= 'balanced', kernel= 'linear'))
    #('svm', SVC(C= 25, class_weight= None, kernel= 'rbf', gamma=1))
])

full_pipeline_no_model = Pipeline([
    ('trf', full_transform),
])

# %%
# Evaluate model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# %% [markdown]
# ## Grid Search

# %% [markdown]
# ### Evaluation Metric Discussion
# 
# If your dataset is heavily imbalanced and/or you mostly care about the positive class, I’d consider using F1 score, or Precision-Recall curve and PR AUC.  The additional reason to go with F1 (or Fbeta) is that these metrics are easier to interpret and communicate to business stakeholders.
# (https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)
# 
# As a rule of thumb, if the cost of having False negative is high, we want to increase the model sensitivity and recall (which are the exact same in regard to their formula)!.
# For instance, in fraud detection or sick patient detection, we don't want to label/predict a fraudulent transaction (True Positive) as non-fraudulent (False Negative). Also, we don't want to label/predict a contagious sick patient (True Positive) as not sick (False Negative).
# This is because the consequences will be worse than a False Positive (incorrectly labeling a a harmless transaction as fraudulent or a non-contagious patient as contagious). (https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc)
# 
# --> Consequently, when you have a data imbalance between positive and negative samples, you should always use F1-score because ROC averages over all possible thresholds!!

# %%
# Use f1 beta score that emphasizes recall more

from sklearn.metrics import fbeta_score, make_scorer

f1 = make_scorer(fbeta_score , beta=2) # With beta of 

# %% [markdown]
# ### Multiple models

# %%
# Do parameter tuning
# The GridSearchCV is a library function that is a member of sklearn's model_selection package. It helps to loop through predefined hyperparameters and fit your estimator (model) on your training set. So, in the end, you can select the best parameters from the listed hyperparameters.
# This implementation is to compare several different models. By default, the GridSearchCV uses a 5-fold cross-validation

# Good SVM article: https://pythonmachinelearning.pro/classification-with-support-vector-machines/

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


grid_models = [(LogisticRegression(),[{'C':[0.25,0.5,0.75,1],'random_state':[0], "penalty": ["l2", 'none'],  'class_weight': [None, 'balanced']}]), 
               (SVC(),[{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.001,.009,0.01,.09,1,5,10,25], 'class_weight': [None, 'balanced']}])]


# Perform on the oversampled and raw data respectively
for data in ["raw", "over"]:

    # Get training X and y of either raw or oversampled data 
    X = globals()[f"X_{data}_train"]
    y = globals()[f"y_{data}_train"]
    
    X_transformed = full_pipeline_no_model.fit_transform(X)

    for i,j in grid_models:
        grid = GridSearchCV(estimator = i, param_grid = j, scoring = f1) # use F1 as evaluation metric
        grid.fit(X_transformed, y.values.flatten())
        # Transform test values
        X_transformed_test = full_pipeline_no_model.transform(X_test) 
        # Get predictions  
        y_pred_acc = grid.predict(X_transformed_test)
        # Get optimal values
        best_accuracy = grid.best_score_
        best_param = grid.best_params_
        print(f"Looking at {data} data!")
        print('----------------')
        print('{}:\nBest F1 : {:.2f}%'.format(i,best_accuracy*100))
        #Logistic Regression (Grid Search) Confusion matrix
        print(confusion_matrix(y_test,y_pred_acc))
        print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
        print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
        print('Best Parameters : ', best_param)
        print('')
        print('----------------')
        print('')


# %% [markdown]
# ### Single model

# %%
# Use f1 beta score that emphasizes recall more

from sklearn.metrics import fbeta_score, make_scorer

f1 = make_scorer(fbeta_score , beta=2)

# %%
# This GridSearch version is just for a single model

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

X_test_transformed_rbf = full_pipeline_no_model.transform(X_test)
X_train_transformed = full_pipeline_no_model.transform(X_over_train)

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_train_transformed)

reduced = pca.transform(X_test_transformed_rbf)


model = SVC()

scores = ["precision", "recall"]

# Transform the input data before fitting it to GridSearch
X_transformed = full_pipeline_no_model.fit_transform(X_raw_train)

# Extensive test
grid_values = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'C':[0.001,.009,0.01,.09,1,5,10,25], 'class_weight': [None, 'balanced']}
#grid_values = {'penalty': ['l1', 'l2'], 'C':[0.001,0.01,0.1,1,5,10,25], 'class_weight': [None, 'balanced'],'loss': ['hinge', 'squared_hinge'] }

grid_clf_acc = GridSearchCV(model, param_grid = grid_values, scoring = 'roc_auc') # Take custom scorer

grid_clf_acc.fit(X_transformed, y_raw_train.values.flatten())

# Transform test values
X_transformed_test = full_pipeline_no_model.transform(X_test)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_transformed_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))
print('ROC_AUC: ' + str(roc_auc_score(y_test, y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
print(confusion_matrix(y_test, y_pred_acc))

best_param = grid_clf_acc.best_params_

print('Best Parameters : ', best_param)



# %% [markdown]
# # Visualization of Results

# %% [markdown]
# ## SVM with rbf Kernel

# %%
from sklearn.decomposition import PCA

# FROM: https://towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea
# https://scikit-learn.org/0.20/auto_examples/svm/plot_iris.html 

'''
Explanation of plots:
- 2 principal components chosen
- Plot the decision function as a 2D contour plot along with the support vectors as scatter points
- Red -> Stroke, Blue -> Non-Stroke
'''

X_test_transformed_rbf = full_pipeline_no_model.transform(X_test)
X_train_transformed_rbf = full_pipeline_no_model.transform(X_over_train)

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_train_transformed_rbf)

reduced = pca.transform(X_test_transformed_rbf)

#svm_model_rbf = SVC(C= 25, class_weight= 'balanced', kernel= 'linear')
svm_model_rbf = SVC(C= 25, class_weight= None, kernel= 'rbf')

classify = svm_model_rbf.fit(reduced, y_test)

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print ('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print ('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))#,
                         #np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = reduced[:, 0], reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(20,16))
fig.patch.set_facecolor('white')
cdict1={0:'blue',1:'red'}

Y_tar_list = y_test.values.ravel().tolist()

yl1= [int(target1) for target1 in Y_tar_list]

labels1=yl1
 
labl1={0:'Non-Stroke', 1:'Stroke'}

marker1={0:'o',1:'d'}

alpha1={0:.2, 1:0.9}

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

ax.scatter(svm_model_rbf.support_vectors_[:, 0], svm_model_rbf.support_vectors_[:, 1], s=40, facecolors='none', 
           edgecolors='navy', label='Support Vectors')

plot_contours(ax, classify, xx, yy,cmap='RdBu_r', alpha=0.4)

plt.legend(fontsize=20, loc='lower right')

plt.xlabel("1st Principal Component",fontsize=30)
plt.ylabel("2nd Principal Component",fontsize=30)

#plt.savefig('ClassifyMalignant_Benign2D_Decs_FunctG10.png', dpi=300)
plt.show()

# %% [markdown]
# ## SVM with linear kernel

# %%
from sklearn.decomposition import PCA
import matplotlib.colors as mcol

# FROM: https://towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea
# https://scikit-learn.org/0.20/auto_examples/svm/plot_iris.html 

X_test_transformed_lin = full_pipeline_no_model.transform(X_test)
X_train_transformed_lin = full_pipeline_no_model.transform(X_raw_train) # raw data

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_train_transformed_lin)

reduced_lin = pca.transform(X_test_transformed_lin)

svm_model = SVC(C= 25, class_weight= 'balanced', kernel= 'linear')
# svm_model = SVC(C= 25, class_weight= None, kernel= 'rbf')

classify = svm_model.fit(reduced_lin, y_test)

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # Decision function is defining the contours! Change to "predict" in order to see cdecision boundary
    print ('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print ('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))#,
                         #np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = reduced_lin[:, 0], reduced_lin[:, 1]

xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(20,16))

fig.patch.set_facecolor('white')

cdict1={0:'blue',1:'red'}

Y_tar_list = y_test.values.ravel().tolist()

yl1= [int(target1) for target1 in Y_tar_list]

labels1=yl1
 
labl1={0:'Non-Stroke',1:'Stroke'}

marker1={0:'o',1:'d'}

alpha1={0:.4, 1:0.9}

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
           edgecolors='navy', label='Support Vectors')


# Make a user-defined colormap.
cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])

plot_contours(ax, classify, xx, yy,cmap='RdBu_r', alpha=0.3)

# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 

plt.legend(fontsize=20)

plt.xlabel("1st Principal Component",fontsize=30)
plt.ylabel("2nd Principal Component",fontsize=30)

#plt.savefig('ClassifyMalignant_Benign2D_Decs_FunctG10.png', dpi=300)
plt.xlim([-3, 5])
plt.ylim([-3, 4])
plt.show()



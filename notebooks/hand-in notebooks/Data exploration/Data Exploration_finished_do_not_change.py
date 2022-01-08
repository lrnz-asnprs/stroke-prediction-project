# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency 
from scipy import stats
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
%matplotlib inline

# %%
# Create dataframe 
df = pd.read_csv("..\\..\\data\\healthcare-dataset-stroke-data.csv") 
# change to // if mac

# %%
# show first five rows to get an overview
df.head()

# %%
# Desciption of each coloumn
df.describe().round(2)

# %%
# Attribute overview
df.info()

# %%
df["hypertension"].sum()

# %%
df["heart_disease"].sum()

# %%
#BMI adult 
df[df["age"]>18].bmi.describe()

# %% [markdown]
# #### Nominal values

# %%
df["smoking_status"].unique()

# %%
df["work_type"].unique()

# %%
df["Residence_type"].unique()

# %% [markdown]
# #### drop irrelevant columns

# %%
# We drop the id column, since we know we are not going to use it 
df = df.drop('id',1)

# %% [markdown]
# # Missing values

# %% [markdown]
# We want to deal with our missing values before moving on. 

# %%
# Number of null (missing) values 
df.isna().sum()

# %%
# Figuring out how many of the strokes that have a missing bmi
stroke_df = df[df['stroke'] == 1]
stroke_df.isnull().sum()

# %% [markdown]
# #### imputing misssing values

# %%
print("Shape before DecisionTreeRegressor " + str(df.shape))

# %%
# We create a pipeline in order to reuse it later
# We want to use a decisiontree so that we do not need to replace 1/5 of the strokes with the same value of bmi
# by replacing it with mean or median

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
print("Shape after DecisionTreeRegressor " + str(df.shape))

# %%
df.isnull().sum()

# %% [markdown]
# # Visuals 

# %%
# Histograms
hist = df.hist(figsize = (10,10))

# %%
# Pairplots to show where there might be clusters 
sns_plot = sns.pairplot(df, height = 3, vars = ['age','avg_glucose_level', 'bmi'])
# sns_plot.savefig('pairplot.png') # saves it as a picture 
plt.show()

# %%
df.select_dtypes(include=['number']).columns

# %% [markdown]
# #### densitity plots

# %%
#https://www.kaggle.com/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5

# %%
fig=plt.figure(figsize=(10,4),facecolor='white')
fig = sns.kdeplot(data=df[df.stroke==1],x='avg_glucose_level',shade=True,color='#0f4c8180', label="Stroke")
fig = sns.kdeplot(data=df[df.stroke==0],x='avg_glucose_level',shade=True,color='#e3e2e160',label="No stroke")
#fig.get_yaxis().set_visible(False)
fig.legend()
fig.get_figure().savefig("gluc_kde_dist.png")

# %%
fig=plt.figure(figsize=(10,4),facecolor='white')
fig = sns.kdeplot(data=df[df.stroke==1],x='age',shade=True,color='#0f4c81', label="Stroke")
fig = sns.kdeplot(data=df[df.stroke==0],x='age',shade=True,color='#e3e2e1',label="No stroke")
#fig.get_yaxis().set_visible(False)
fig.legend()
fig.get_figure().savefig("age_kde_dist.png")

# %%
#smaller version for report
fig=plt.figure(facecolor='white')
fig = sns.kdeplot(data=df[df.stroke==1],x='age',shade=True,color='#0f4c81', label="Stroke")
fig = sns.kdeplot(data=df[df.stroke==0],x='age',shade=True,color='#e3e2e1',label="No stroke")
#fig.get_yaxis().set_visible(False)
fig.legend(loc='upper left')
fig.get_figure().savefig("age_kde_dist_smaller.png")

# %%
fig=plt.figure(figsize=(10,4),facecolor='white')
fig = sns.kdeplot(data=df[df.stroke==1],x='bmi',shade=True,color='#0f4c81', label="Stroke")
fig = sns.kdeplot(data=df[df.stroke==0],x='bmi',shade=True,color='#e3e2e160',label="No stroke")
#fig.get_yaxis().set_visible(False)
fig.legend()
fig.get_figure().savefig("bmi_kde_dist.png")

# %%
fig = sns.kdeplot(df["age"], color='#0f4c81', shade=True, ec='black',alpha=0.6)
fig.set_xlabel('Age')
#fig.get_yaxis().set_visible(False)
fig.get_figure().savefig("age_dist.png")

# %%
fig = sns.kdeplot(df["bmi"], color='#0f4c81', shade=True, ec='black',alpha=0.6)
fig.set_xlabel('BMI')
#fig.get_yaxis().set_visible(False)
fig.get_figure().savefig("bmi_dist.png")

# %%
fig = sns.kdeplot(df["avg_glucose_level"], color='#0f4c81', shade=True, ec='black',alpha=0.6)
fig.set_xlabel('Avg. glucose level')
#fig.get_yaxis().set_visible(False)
fig.get_figure().savefig("avg_gluc_dist.png")

# %% [markdown]
# ### age and ..

# %%
# Scatterplots on two-variable relationships between age(discrete) and continuous variables 
# and binary value (stroke, heart disease and hypertension)

fig, axs = plt.subplots(3,2, figsize=(16,16))

color_dict = dict({1:"#0f4c8180",0:"#e3e2e160"})

x=df["age"]

sns.scatterplot(ax = axs[0,0], x=x, y=df["bmi"], hue=df["stroke"], palette=color_dict)
sns.scatterplot(ax = axs[0,1], x=x, y=df["avg_glucose_level"], hue=df["stroke"],palette=color_dict)
sns.scatterplot(ax = axs[1,0], x=x, y=df["bmi"], hue=df["heart_disease"],palette=color_dict)
sns.scatterplot(ax = axs[1,1], x=x, y=df["avg_glucose_level"], hue=df["heart_disease"],palette=color_dict)
sns.scatterplot(ax = axs[2,0], x=x, y=df["bmi"], hue=df["hypertension"],palette=color_dict)
sns.scatterplot(ax = axs[2,1], x=x, y=df["avg_glucose_level"], hue=df["hypertension"],palette=color_dict)

fig.savefig("scatterplot_dist.png")

# %% [markdown]
# Overall:
# - The higher the age, the more have problems in general (strokes, heart dieseases, hypertension)
# - The higher the glucose level, the more seem to have strokes at an higher age!
# 
# Strokes:
# - High age + high glucose level -> a lot of strokes
# - High age but high BMI -> not so many strokes
# Heart disease:
# - The higher the age, the more heart diseases
# - But high BMI does not seem to have a great influence
# - Avg glucose level does not seem to matter a lot too
# Hypertension
# - Also, the higher the age, the more have hypertension, but definitely less clear than for other two
# - Interestingly, all outliers with very high BMI have also hypertension
# - Avg glucose level really does not influence it much

# %%
# BMI and glucose level within men and women
df.groupby(['gender']).nunique().plot(kind = 'bar', y = ['avg_glucose_level', 'bmi'])
plt.show()

# %% [markdown]
# # Outliers

# %%
df.select_dtypes(include=['number']).columns

# %%
fig, axs = plt.subplots(2)
plt.subplots_adjust(hspace = 1)
col = ['avg_glucose_level','bmi']

for i, col in enumerate(col):
    fig = sns.boxplot(x = df[col],ax=axs[i], color='#0f4c81', boxprops=dict(alpha=.6))
    
fig.get_figure().savefig("outliers.png")

# %%
# Outliers in BMI
#fig = sns.boxplot(x = df['bmi'],color='#0f4c81',boxprops=dict(alpha=.6))
#fig.get_figure().savefig("bmi_outliers.png")

# %%
# Outliers on glukose level
#fig = sns.boxplot(x = df['avg_glucose_level'])

# %%
# Outliers on age 
#fig = sns.boxplot(x = df['age'])

# %% [markdown]
# # Stroke in different groups

# %%
# Strokes within men and women 
print((df[df["gender"]=="Male"].sum()["stroke"]/df[df["gender"]=="Male"].count()["stroke"])*100)
print((df[df["gender"]=="Female"].sum()["stroke"]/df[df["gender"]=="Female"].count()["stroke"])*100)

# %%
# Strokes divided between urban and rural
print((df[df["Residence_type"]=="Urban"].sum()["stroke"]/df[df["Residence_type"]=="Urban"].count()["stroke"])*100)
print((df[df["Residence_type"]=="Rural"].sum()["stroke"]/df[df["Residence_type"]=="Rural"].count()["stroke"])*100)

# %%
# Percent of people who have a heartdisease and a stroke, and people who had a stroke but no heart disease 
print((df[df["heart_disease"]==1].sum()["stroke"]/df[df["heart_disease"]==1].count()["stroke"])*100)
print((df[df["heart_disease"]==0].sum()["stroke"]/df[df["heart_disease"]==0].count()["stroke"])*100)

# %%
# Percent of people who have a hypertension and a stroke, and people who had a stroke but no hypertension
print((df[df["hypertension"]==1].sum()["stroke"]/df[df["hypertension"]==1].count()["stroke"])*100)
print((df[df["hypertension"]==0].sum()["stroke"]/df[df["hypertension"]==0].count()["stroke"])*100)

# %%
# Smoking status and stroke
print((df[df["smoking_status"]=="never smoked"].sum()["stroke"]/df[df["smoking_status"]=="never smoked"].count()["stroke"])*100)
print((df[df["smoking_status"]=="Unknown"].sum()["stroke"]/df[df["smoking_status"]=="Unknown"].count()["stroke"])*100)
print((df[df["smoking_status"]=="formerly smoked"].sum()["stroke"]/df[df["smoking_status"]=="formerly smoked"].count()["stroke"])*100)
print((df[df["smoking_status"]=="smokes"].sum()["stroke"]/df[df["smoking_status"]=="smokes"].count()["stroke"])*100)

# %% [markdown]
# # Correlation

# %%
# Correlation heatmap (only between two variables)

numerical = df[["stroke", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "age"]]
numerical = numerical.rename(columns={"avg_glucose_level": "avg_g_l", "heart_disease": "h_dis","hypertension":"hyp_ten"})
fig = sns.heatmap(numerical.corr(), annot=True,cmap="Blues")
fig.get_figure().savefig("heatmap.png")

#We can't see a strong positive correlation between any numerical value and a stroke
#The largest positive correlation for a stroke is with age 

# %% [markdown]
# # Chi square test

# %% [markdown]
# Exploring the relationship between categorical values by using a Chi Square Test

# %%
#First look at smoking_status -> stroke!

# Determine significance level
significance_level = 0.05

chisqt = pd.crosstab(df.smoking_status, df.stroke, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
# first value chi-square, second value p-value, then degrees of freedom
if chi2_contingency(value)[1] <= significance_level:
    print("The two varibles have a significant correlation!")
else:
    print("The two varibles have NO significant correlation!")
# chisqt

# %% [markdown]
# Now want to determine how much a category value contributes to stroke, 
# 
# therefore we conduct post hoc testing.
# Conduct multiple 2×2 Chi-square tests 
# using the Bonferroni-adjusted p-value.

# %%
# Determine the Bonferroni-adjusted p-value
# The formula is p/N, where “p”= the original tests p-value and “N”= the number of planned pairwise comparisons
bonferroni_p = 0.05/3

dummies = pd.get_dummies(df['smoking_status'])
dummies.drop(["Unknown"], axis= 1, inplace= True)
dummies.head()

# %%
# Check whether they are significant and also calculate phi coefficient (like pearson)
# https://www.statisticshowto.com/phi-coefficient-mean-square-contingency-coefficient/ 
# https://en.wikipedia.org/wiki/Phi_coefficient

for series in dummies:
    nl = "\n"
    crosstab = pd.crosstab(dummies[f"{series}"], df['stroke'])
    print(crosstab, nl)
    a = crosstab.loc[0][0] # needed for phi coefficient
    b = crosstab.loc[0][1]
    c = crosstab.loc[1][0]
    d = crosstab.loc[1][1]
    phi_coeff = (a*d-b*c)/sqrt((a+b)*(c+d)*(a+c)*(b+d))
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    significance = p <= bonferroni_p
    print(f"Chi2 value= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}{nl}Significant {significance}{nl}Phi coefficient {phi_coeff}")
    print()

# %% [markdown]
# Therefore only the category value "formerly smoked" is significant
# 
# -> Means that a higher proportion of people who had a stroke also smoked formerly!
# 
# -> But VERY low phi coefficient, therefore no true correlation
# (Note that correlation does not imply causality. 
# 
# That is, if A and B are correlated, 
# this does not necessarily imply that 
# A causes B or that B causes)

# %%
# Next look at work_type -> stroke!
# Determine significance level
significance_level = 0.05

chisqt = pd.crosstab(df.work_type, df.stroke, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
# first value chi-square, second value p-value, then degrees of freedom
if chi2_contingency(value)[1] <= significance_level:
    print("The two varibles have a significant correlation!")
else:
    print("The two varibles have NO significant correlation!")
# chisqt

# %%
# Next look at residence type -> stroke!
# Determine significance level
significance_level = 0.05

chisqt = pd.crosstab(df.Residence_type, df.stroke, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
# first value chi-square, second value p-value, then degrees of freedom
if chi2_contingency(value)[1] <= significance_level:
    print("The two varibles have a significant correlation!")
else:
    print("The two varibles have NO significant correlation!")
# chisqt

# %%
# Next look at marriage -> stroke!
# Determine significance level
significance_level = 0.05

chisqt = pd.crosstab(df.ever_married, df.stroke, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
# first value chi-square, second value p-value, then degrees of freedom
if chi2_contingency(value)[1] <= significance_level:
    print("The two varibles have a significant correlation!")
else:
    print("The two varibles have NO significant correlation!")
#chisqt

# %%
dummies = pd.get_dummies(df['ever_married'])
# dummies.drop(["Unknown"], axis= 1, inplace= True)

# %%
# Check whether they are significant and also calculate phi coefficient (like pearson)
# https://www.statisticshowto.com/phi-coefficient-mean-square-contingency-coefficient/ 
# https://en.wikipedia.org/wiki/Phi_coefficient

for series in dummies:
    nl = "\n"
    crosstab = pd.crosstab(dummies[f"{series}"], df['stroke'])
    print(crosstab, nl)
    a = crosstab.loc[0][0] # needed for phi coefficient
    b = crosstab.loc[0][1]
    c = crosstab.loc[1][0]
    d = crosstab.loc[1][1]
    phi_coeff = (a*d-b*c)/sqrt((a+b)*(c+d)*(a+c)*(b+d))
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    significance = p <= bonferroni_p
    print(f"Chi2 value= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}{nl}Significant {significance}{nl}Phi coefficient {phi_coeff}")
    print()



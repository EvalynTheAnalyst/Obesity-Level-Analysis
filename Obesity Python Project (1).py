#!/usr/bin/env python
# coding: utf-8

# # Importing relevant Libraries 
# 

# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Display whole dataset
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import data from local files
obesity_dt = pd.read_excel(r"C:\Users\pavilion14\Downloads\ObesityDataSet_raw_and_data_sinthetic.xlsx")

# Display the first 10 rows of the dataset

obesity_dt.head(10)


# # ***Sanity Check On Data***

# In[92]:


obesity_dt.info()


# In[93]:


obesity_dt.shape


# In[94]:


# Checking for missing values in each variable
Missing_values = obesity_dt.isna().any()

# Checking for duplicate data 
num_duplicates = obesity_dt.duplicated().sum()


print(Missing_values)
print(f'Duplicates total {num_duplicates}')


# In[95]:


# Display duplicate rows
Duplicated_rows  = obesity_dt[obesity_dt.duplicated()]

# Drop duplicate values
obesity_dt.drop_duplicates(inplace = True)

duplicate = obesity_dt.duplicated().sum()
print(f"The Total duplicate is:{duplicate}")


# In[96]:


#Checking for garbage values
for i in obesity_dt.select_dtypes(include = 'object').columns:
    print(obesity_dt[i].value_counts())
    print('***'*10)


# In[97]:


obesitydf_encoded.shape


# ## Checking and Handling Outliers

# In[98]:


#Checking For outliners
plt.figure(figsize = (15,5))
# Plot the boxplot for Weight
plt.subplot(1, 5, 1)
sns.boxplot(y=obesity_dt['Weight'])
plt.title('Boxplot of Weight')

# Plot the boxplot for Height
plt.subplot(1, 5, 2)
sns.boxplot(y=obesity_dt['Height'])
plt.title('Boxplot of Height')

plt.subplot(1, 5, 3)
sns.boxplot(y=obesity_dt['Age'])
plt.title('Boxplot of Age')


# Plot the boxplot for Weight
plt.subplot(1, 5, 4)
sns.boxplot(y=obesity_dt['FCVC'])
plt.title('Boxplot of FCVC')

# Plot the boxplot for Height
plt.subplot(1, 5, 5)
sns.boxplot(y=obesity_dt['NCP'])
plt.title('NCP')





# Show the plot
plt.tight_layout()
plt.show()


# In[99]:


# Define a function to cap outliers
def cap_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = df.quantile(lower_quantile)
    upper_bound = df.quantile(upper_quantile)
    return df.clip(lower_bound, upper_bound)

# Apply capping to Weight and Height
obesity_dt['Weight'] = cap_outliers(obesity_dt['Weight'])
obesity_dt['Height'] = cap_outliers(obesity_dt['Height'])

obesity_dt.describe()


# In[100]:


#Checking  if outliners are removed.
plt.figure(figsize = (15,5))
# Plot the boxplot for Weight
plt.subplot(1, 5, 1)
sns.boxplot(y=obesity_dt['Weight'])
plt.title('Boxplot of Weight')

# Plot the boxplot for Height
plt.subplot(1, 5, 2)
sns.boxplot(y=obesity_dt['Height'])
plt.title('Boxplot of Height')

plt.show()


# # Exploratory Data Analysis (EDA)
# Summarizing the data

# In[101]:


obesity_dt.describe(include = 'number').T


# In[102]:


obesity_dt.describe(include = 'object')


# In[103]:


#Ploting Histogram to understand distribution
for i in obesity_dt.select_dtypes(include = 'number').columns:
    plt.figure(figsize= (6,4))
    sns.histplot(data = obesity_dt, kde=True, x = i)
    plt.show()


# ***# Exploring Relationships between different attributes*** 
# 

# In[104]:


obesity_order = [
    "Insufficient_Weight",
    "Normal_Weight", 
    "Overweight_Level_I", 
    "Overweight_Level_II", 
    "Obesity_Type_I", 
    "Obesity_Type_II", 
    "Obesity_Type_III"
]

fig, axes = plt.subplots(5, 2, figsize=(20, 30))
sns.boxplot(x='NObeyesdad', y='Weight',order =obesity_order, data=obesity_dt, ax=axes[0,0])
axes[0,0].set_title('Weight vs Obesity Levels')
axes[0,0].tick_params(axis='x', rotation=45)

sns.boxplot(x='NObeyesdad', y='FAF', data=obesity_dt,order =obesity_order, ax=axes[0,1])
axes[0,1].set_title('FAF vs Obesity Levels')
axes[0,1].tick_params(axis='x', rotation=45)


sns.boxplot(x='family_history_with_overweight',y = 'Weight', data=obesity_dt, ax=axes[1,0])
axes[1,0].set_title('Weight Vs Family History')
axes[1,0].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='MTRANS',y = 'Weight', data=obesity_dt, ax=axes[1,1])
axes[1,1].set_title('Weight Vs MTRANS')
axes[1,1].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='NObeyesdad',y = 'FCVC', order =obesity_order, data=obesity_dt, ax=axes[2,0])
axes[2,0].set_title('FCVC Vs Obesity Level')
axes[2,0].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='NObeyesdad',y = 'NCP',order =obesity_order, data=obesity_dt, ax=axes[2,1])
axes[2,1].set_title('NCP Vs Obesity Level')
axes[2,1].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='NObeyesdad',y = 'CH2O',order =obesity_order, data=obesity_dt, ax=axes[3,0])
axes[3,0].set_title('Water Intake Vs Obesity Level')
axes[3,0].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='NObeyesdad',y = 'TUE',order =obesity_order, data=obesity_dt, ax=axes[3,1])
axes[3,1].set_title('TUE Vs Obesity Level')
axes[3,1].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='NObeyesdad',y = 'Age',order =obesity_order, data=obesity_dt, ax=axes[4,0])
axes[4,0].set_title('Age Vs Obesity Level')
axes[4,0].tick_params(axis = 'x', rotation = 45)

sns.boxplot(x='CAEC',y = 'Weight', data=obesity_dt, ax=axes[4,1])
axes[4,1].set_title('Weight Vs CAEC')
axes[4,1].tick_params(axis = 'x', rotation = 45)




plt.tight_layout()
plt.show()



# In[105]:


#Checking for relationship between Variables
cols = obesity_dt[['Age', 'Height', 'Weight','CH2O', 'FAF','NCP', 'TUE']]

# Calculating the correlation matrix
correlation_matrix = cols.corr()

# Plotting the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Continuous Variables", weight = 'bold', fontsize = 10)
plt.show()


# ## Preparing Data For Machine Learning 

# In[106]:


# Pair plots
selected_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF']
pairplot_data = obesity_dt[selected_features + ['NObeyesdad']]

# Pair plot colored by the obesity levels
sns.pairplot(pairplot_data, hue='NObeyesdad', diag_kind='kde', palette='Set2')
plt.suptitle('Pair Plot of continuous variables', y= 1.02)
plt.show()


# In[107]:


from sklearn.preprocessing  import LabelEncoder,OneHotEncoder


obesity_dt.dtypes

# Create a copy of the data for encoding
obesitydf_encoded = obesity_dt.copy()

# Columns for label encoding (binary)
binary_columns = ['Gender', 'SMOKE', 'family_history_with_overweight', 'FAVC', 'SCC','CALC', 'CAEC', 'NObeyesdad']
label_encoder = LabelEncoder()

# Apply label encoding to binary columns
for col in binary_columns:
    obesitydf_encoded[col] = label_encoder.fit_transform(obesitydf_encoded[col])
    
# One-hot encode multi-class columns
multi_class_columns = ['MTRANS']

# drop='first' to avoid multicollinearity and convert to a dataframe
OHE = OneHotEncoder(handle_unknown = 'ignore',sparse_output=False, drop='first').set_output(transform = 'pandas')  

# Apply one-hot encoding to the selected columns
encoded_features = OHE.fit_transform(obesitydf_encoded[multi_class_columns])


# Drop the original multi-class columns and concatenate the new one-hot encoded columns
obesitydf_encoded = obesitydf_encoded.drop(columns=multi_class_columns)
obesitydf_encoded = pd.concat([obesitydf_encoded, encoded_features], axis=1)

# Display the first few rows of the copied dataset
obesitydf_encoded.head()




# In[108]:


import warnings
warnings.filterwarnings("ignore")
col = ['Age', 'Weight', 'Height','FAF','CH2O','NCP', 'FCVC']

from sklearn.preprocessing import MinMaxScaler
# Data Normalize Continuous Variables using Min-Max Scaling
scaler = MinMaxScaler()
obesitydf_encoded[col] = scaler.fit_transform(obesitydf_encoded[col])


for i in col:
    plt.figure(figsize = (16,4))
    plt.subplot(141)
    sns.distplot(obesitydf_encoded[i],label = 'skew:' + str(np.round(obesitydf_encoded[i].skew(),2)))
    plt.title('After Data Normalization')
    plt.tight_layout()
    plt.show()


# In[109]:


obesitydf_encoded.info()


# In[110]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X = obesitydf_encoded.drop(columns=['NObeyesdad'])
y = obesitydf_encoded['NObeyesdad']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'OverWeight_Level_1',
    3: 'OverWeight_Level_2',
    4: 'Obesity_Type_1',
    5: 'Obesity_Type_2',
    6: 'Obesity_Type_3'
}

# Apply the mapping
y_train = y_train.map(label_mapping) 
y_test = y_test.map(label_mapping)

# Train the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
print(rf_model.score(X_test, y_test))

# Make predictions
y_pred = rf_model.predict(X_test)

#  Generate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels = ['Insufficient_Weight','Normal_Weight','OverWeight_Level_1','OverWeight_Level_2','Obesity_Type_1','Obesity_Type_2','Obesity_Type_3'])



# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['Insufficient_Weight','Normal_Weight','OverWeight_Level_1','OverWeight_Level_2','Obesity_Type_1','Obesity_Type_2','Obesity_Type_3'])
plt.figure(figsize=(10, 8)) 
disp.plot(cmap='Blues', colorbar=True)
plt.xticks(rotation = 90)
plt.xlabel('Prediction', weight = 'bold')
plt.ylabel('Actual', weight = 'bold')
plt.title("RF Obesity Level Prediction", weight = 'bold', fontsize = 10)
plt.show()

print(y_pred)
print(classification_report(y_test,y_pred))


# In[111]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 42,)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(lr.score(X_test,y_test))

cm = confusion_matrix(y_test,y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = label_mapping)
disp.plot(cmap = 'Greens',colorbar = True)
plt.title('LogisticRegression Prediction', weight = 'bold', fontsize = 10)
plt.xlabel('Prediction')
plt.ylabel('Prediction')
plt.xticks(rotation = 90)
plt.show()
print(y_pred)
print(classification_report(y_test,y_pred))


# # Tunning The Logistic Regression Model For PredictionTo be More Accurate

# In[112]:


from sklearn.preprocessing import  PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2



#Feature Interaction (Polynomial Features)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature Selection (SelectKBest)
selector = SelectKBest(score_func=chi2, k='all')  # Choose 'all' or a specific number of features
X_train_selected = selector.fit_transform(np.abs(X_train_poly), y_train)  # Use abs for chi2
X_test_selected = selector.transform(np.abs(X_test_poly))

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_train_selected, y_train)
y_pred = lr.predict(X_test_selected)

# Model Performance Metrics
print(f"Accuracy: {lr.score(X_test_selected, y_test)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels = ['Insufficient_Weight','Normal_Weight','OverWeight_Level_1','OverWeight_Level_2','Obesity_Type_1','Obesity_Type_2','Obesity_Type_3'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Insufficient_Weight','Normal_Weight','OverWeight_Level_1','OverWeight_Level_2','Obesity_Type_1','Obesity_Type_2','Obesity_Type_3'])
disp.plot(cmap='Blues', colorbar=True)
plt.title('LogisticRegression Prediction Performance', weight = 'bold', fontsize = 10)
plt.xlabel('Prediction', weight = 'bold')
plt.ylabel('Actual', weight = 'bold')
plt.xticks(rotation=90)
plt.show()

# Classification Report
print("Predictions:", y_pred)
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





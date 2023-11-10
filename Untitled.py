#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df1 = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Health Insurance Analysis\\Capstone_1\\Hospitalisation details.csv')


# In[3]:


df2 = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Health Insurance Analysis\\Capstone_1\\Medical Examinations.csv')


# In[4]:


df3 = pd.read_excel('C:\\Users\\Lenovo\\Desktop\\Health Insurance Analysis\\Capstone_1\\Names.xlsx')


# In[5]:


df1.head()


# In[6]:


df2.head()


# In[7]:


df3.head()


# In[8]:


df4 = pd.merge(df1,df2, how = 'outer' , on = 'Customer ID' )


# In[9]:


df = pd.merge(df4,df3 , how = 'outer' , on = 'Customer ID')


# In[10]:


df.head()


# In[11]:


df = df.reindex(columns = ['Customer ID','name','year','month','date','day','children','charges','Hospital tier','City tier','State ID','BMI','HBA1C','Heart Issues','Any Transplants','Cancer history','NumberOfMajorSurgeries','smoker'])


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


month_to_int = {
         'Jan': 1,
         'Feb': 2,
         'Mar': 3,
         'Apr': 4,
         'May' : 5,
         'Jun' : 6,
         'Jul': 7,
         'Aug' :8,
         'Sep':9,
         'Oct':10,
         'Nov' :11,
         'Dec':12
}


# In[15]:


df = df.drop(['day'],axis= 1)


# In[16]:


percent_missing = df.isnull().mean() * 100


# In[17]:


percent_missing


# In[18]:


df = df.replace('?', np.nan)


# In[19]:


df.dropna(inplace=True)


# In[20]:


percent_missing


# *The variable NumberOfMajorSurgeries also appears to have string values. Apply a suitable
# method to clean up this variable

# In[21]:


df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].replace('No major surgery', 0)


# In[22]:


# convert string values to numeric values
df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].astype(int)


# Age appears to be a significant factor in this analysis. Calculate the patients’ ages based on their
# dates of birth.

# In[23]:


from datetime import datetime


# In[24]:


from datetime import date
def calculate_age(year, month, day):
    today = date.today()
    birthdate = date(year, month, day)
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age


# In[25]:


df.info()


# In[26]:


df['year'] = df['year'].astype(int)


# In[27]:


df['month_int'] = df['month'].map(month_to_int)


# In[28]:


df['month_int']


# In[29]:


# Apply the calculate_age function to each row
df['age'] = df.apply(lambda row: calculate_age(row['year'], row['month_int'], row['date']), axis=1)


# In[30]:


df.age


# In[31]:


def get_gender(salutation):
    if 'Mr.' in salutation:
        return 'Male'
    elif 'Ms.' in salutation or 'Miss.' in salutation or 'Mlle.' in salutation:
        return 'Female'
    elif 'Dr.' in salutation:
        return 'Unknown'
    elif 'Mrs.' in salutation or 'Mme.' in salutation or 'Mlle.' in salutation:
        return 'Married Female'
    else:
        return 'Other'


# In[32]:


# Apply the function to create a new column 'gender' based on the 'name' column
df['gender'] = df['name'].apply(lambda x: get_gender(x.split()[1]))


# In[33]:


df.gender


# You should also visualize the distribution of costs using a histogram, box and whisker plot, and
# swarm plot.

# In[34]:


import seaborn as sns


# In[35]:


# Create a histogram of costs using 20 bins
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='charges', bins=20, kde=True)
plt.title('Distribution of Hospitalization Costs')
plt.xlabel('Cost')
plt.ylabel('Count')
plt.show()


# In[36]:


# Create a boxplot of costs using 20 bins
plt.figure(figsize = (8,6))
sns.boxplot(data=df, y='charges')
plt.title('Distribution of Hospitalization Costs')
plt.ylabel('Count')
plt.show()


# In[37]:


# Create a Swarmplot of costs using 20 bins
plt.figure(figsize = (8,6))
sns.swarmplot(data=df , y='charges')
plt.title('Distribution of Hospitalization Costs ')
plt.ylabel('charges')
plt.show()


# State how the distribution is different across gender and tiers of hospitals

# In[38]:


# Create a histogram to show Distribution across a different genders
plt.figure(figsize = (8,6))
sns.histplot(data=df,x ='charges',hue='gender' ,bins=20, kde=True)
plt.title=('Distribution of Hospitalization Costs Based on Gender')
plt.xlabel = ('charges')
plt.ylabel = ('count')
plt.show()


#  Create a radar chart to showcase the median hospitalization cost for each tier of hospitals

# In[39]:


import matplotlib.pyplot as plt
import numpy as np
# Sample data - median hospitalization cost for each tier of hospitals
costs = [15000, 25000, 50000, 30000]
# Define the number of variables we want to plot (i.e., the number of tiers of hospitals)
num_vars = len(costs)
# Calculate the angle for each variable on the radar chart
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
# Close the plot by repeating the first angle
angles = np.concatenate((angles, [angles[0]]))
# Normalize the data to be in the range [0, 1]
max_cost = max(costs)
min_cost = min(costs)
normalized_costs = [(cost - min_cost) / (max_cost - min_cost) for cost in costs]
normalized_costs = np.concatenate((normalized_costs, [normalized_costs[0]]))
# Set up the radar chart
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
# Plot the data on the radar chart
ax.plot(angles, normalized_costs, 'o-', linewidth=2)
# Fill in the area under the line
ax.fill(angles, normalized_costs, alpha=0.25)
# Set the labels for each axis
labels = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
ax.set_xticklabels(labels)
# Set the title of the radar chart
ax.set_title('Median Hospitalization Cost by Tier of Hospitals', fontsize=14)
# Set the gridlines to be visible
ax.grid(True)
# Set the range of the y-axis to [0, 1]
ax.set_ylim([0, 1])
# Add a legend for the data
ax.legend(['Median Cost'], loc=(1.1, 0.8))
# Display the radar chart
plt.show()


# In[40]:


df.rename(columns={'Hospital tier': 'Hospital_tier','Heart Issues': 'Heart_Issues','Any Transplants':'Any_Transplants','Cancer history':
'Cancer_history'}, inplace=True)


# In[41]:


df.rename(columns={'City tier': 'City_tier'}, inplace=True)


# In[42]:


df['Heart_Issues1'] = df['Heart_Issues'].replace({'yes':1,'No':0})


# In[43]:


df['smoker1'] = df['smoker'].replace({'Yes': '1', 'No': '0' ,'yes': '1'})


# In[44]:


df['Any_Transplants'] = df['Any_Transplants'].replace({'Yes': '1', 'No': '0','yes': '1'})


# In[45]:


df['Cancer_history'] = df['Cancer_history'].replace({'Yes': '1', 'No': '0','yes': '1'})


# In[46]:


df['Hospital_tier1'] = df['Hospital_tier'].replace({'tier - 2': '2', 'tier - 1': '1','tier - 3':'3' })


# In[47]:


df['gender'] = df['gender'].replace({'Male': '1', 'Female': '0','Other':'2' })


# In[48]:


df['City_tier1'] = df['City_tier'].replace({'tier - 2': '2', 'tier - 1': '1','tier - 3':'3' })


# In[49]:


# Convert the column to integers
df['Hospital_tier1'] = df['Hospital_tier1'].apply(int)

# Print the converted column
print(df['Hospital_tier1'])


# In[50]:


# Convert the column to integers
df['Heart_Issues1'] = df['Heart_Issues1'].apply(int)

# Print the converted column
print(df['Heart_Issues1'])


# In[51]:


# Convert the column to integers
df['City_tier1'] = df['City_tier1'].apply(int)

# Print the converted column
print(df['City_tier1'])


# In[52]:


# Convert the column to integers
df['smoker1'] = df['smoker1'].apply(int)

# Print the converted column
print(df['smoker1'])


# 0.0.1 Test the following null hypotheses:
# a. The average hospitalization costs for the three types of hospitals are not significantly different

# In[53]:


from scipy.stats import ttest_ind
# Select the two columns of interest
column1 = df['charges']
column2 = df['Hospital_tier1']

# Calculate t and p-values for a two-sample t-test assuming equal variances
t_stat, p_value = ttest_ind(column1, column2, equal_var=True)

# Print the results
print("t-statistic:", t_stat)
print("p-value:", p_value)


# b. The average hospitalization costs for the three types of cities are not significantly different

# In[54]:


from scipy.stats import ttest_ind
# Select the two columns of interest
column3 = df['charges']
column4 = df['City_tier1']

# Calculate t and p-values for a two-sample t-test assuming equal variances
t_stat, p_value = ttest_ind(column3, column4, equal_var=True)

# Print the results
print("t-statistic:", t_stat)
print("p-value:", p_value)


# c. The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers

# In[55]:


from scipy.stats import ttest_ind
# Select the two columns of interest
column6 = df['charges']
column7 = df['smoker1']

# Calculate t and p-values for a two-sample t-test assuming equal variances
t_stat, p_value = ttest_ind(column6, column7, equal_var=True)

# Print the results
print("t-statistic:", t_stat)
print("p-value:", p_value)


# d. Smoking and heart issues are independent

# In[56]:


from scipy.stats import ttest_ind
# Select the two columns of interest
column7 = df['Heart_Issues1']
column8 = df['smoker1']

# Calculate t and p-values for a two-sample t-test assuming equal variances
t_stat, p_value = ttest_ind(column7, column8, equal_var=True)

# Print the results
print("t-statistic:", t_stat)
print("p-value:", p_value)


# As here the P Values are very low therefore we reject null hypothesis and go for alternate hypothesis

# Examine the correlation between predictors to identify highly correlated predictors. Use a heatmap
# to visualize this.

# In[57]:


# Calculate the correlation matrix
corr_matrix = df.corr()
plt.subplots(figsize= [20,15])

# Create a heatmap of the correlation matrix using Seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# Show the plot
plt.show()


# Develop and evaluate the final model using regression with a stochastic gradient descent optimizer.
# Also, ensure that you apply all the following suggestions: 
#     Note: • Perform the stratified 5-fold crossvalidation technique for model building and validation 
#           • Use standardization and hyperparameter tuning effectively • Use sklearn-pipelines 
#           • Use appropriate regularization techniques to address the bias-variance trade-off 
# a. Create five folds in the data, and introduce a variable to identify the folds 
# b. For each fold, run a for loop and ensure that 80 percent of the data is used to train the model and the remaining 20 percent is used to validate it in each iteration 
# c. Develop five distinct models and five distinct validation scores (root mean squared error values) 
# d. Determine the variable importance scores, and identify the redundant variables

# In[58]:


df.columns


# In[59]:


# Split the data into features (X) and target (y)
X = df[['children','Hospital_tier1', 'City_tier1', 'BMI', 'HBA1C', 'Heart_Issues1', 'Any_Transplants', 'Cancer_history', 'NumberOfMajorSurgeries', 'smoker1', 'month_int', 'age']]
y = df['charges']


# In[60]:


from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error


# In[61]:


# Define a pipeline with StandardScaler and SGDRegressor
pipe = Pipeline([('scaler', StandardScaler()), ('reg', SGDRegressor())])

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'reg__alpha': [0.0001, 0.001, 0.01],
    'reg__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'reg__max_iter': [1000, 5000, 10000],
    'reg__tol': [0.0001, 0.001, 0.01]
}
# Use KFold function to perform stratified 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over the folds
fold = 1
for train_index, test_index in kf.split(X, y):
    print(f'Fold: {fold}')

    # Split the data into training and validation sets for this fold
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    # Define GridSearchCV object for hyperparameter tuning
    grid = GridSearchCV(pipe, param_grid, cv=3)
    
    # Fit the GridSearchCV object on the training data
    grid.fit(X_train, y_train)
    
    # Use the trained pipeline to predict the target variable for the validation data
    y_pred = grid.predict(X_val)
    
    # Compute the root mean squared error (RMSE)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f'Validation RMSE: {rmse:.2f}')
    fold += 1


# In[62]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# In[63]:


# Define pipeline
rf_pipe = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

# Fit and evaluate model using cross-validation
rf_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')

print('Random Forest RMSE scores:', -rf_scores)

# Calculate variable importance scores
rf_model = RandomForestRegressor(random_state=42).fit(X, y)
rf_feature_importances = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
print('Random Forest feature importances:\n', rf_feature_importances)


# 0.0.3 XG Boost

# In[64]:


from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()

# Apply label encoding to the categorical columns
X['Any_Transplants'] = le.fit_transform(X['Any_Transplants'])
X['Cancer_history'] = le.fit_transform(X['Cancer_history'])


# In[65]:


import XGBoost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Define pipeline
xgb_pipe = make_pipeline(StandardScaler(), xgb.XGBRegressor(random_state=42))

# Fit and evaluate model using cross-validation
xgb_scores = cross_val_score(xgb_pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
print('XGBoost RMSE scores:', -xgb_scores)

# Calculate variable importance scores
xgb_model = xgb.XGBRegressor(random_state=42).fit(X, y)
xgb_feature_importances = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)
print('XGBoost feature importances:\n', xgb_feature_importances)


# In[ ]:


import joblib


# In[ ]:


# Train the model
model = ... # Your trained model object


# In[ ]:


# Save the model to a file
joblib.dump(model, 'model.joblib')


# In[ ]:





# -*- coding: utf-8 -*-
"""
@author: user - Sanket Jadhav
"""

# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\07-SANKET\DATA SCIENCE\0.A - DS Projects\A - CapStone Projects\Loan Prediction based on Customer Behaviour\Training Data.csv')

# Data Cleaning + EDA
df

# Cleaning names with special characters and numbers
def unclean_names(col):
    unclean_names = []
    for name in df[str(col)].unique():
        if name.endswith(']'):
            unclean_names.append(name)
    return unclean_names

unclean_city_names = unclean_names('CITY')
unclean_city_names

unclean_state_names = unclean_names('STATE')
unclean_state_names

def clean_df(df,col,unclean_list):
    for index,name in enumerate(df[col]):
        if name in unclean_list:
            if name.endswith(']'):
                name_ = name.strip('[]0123456789')
                df[col].iloc[index] = name_
                
clean_df(df,'STATE',unclean_state_names)                

clean_df(df,'CITY',unclean_city_names)

# Checking for any Outliers
df['Age'].plot(kind='hist',figsize=(10,8))
plt.xlabel('Age')

df['Income'].plot(kind='box')

df['Income'].plot(kind='hist')

df['CURRENT_JOB_YRS'].plot(kind='box')

df['CURRENT_HOUSE_YRS'].plot(kind='box')

df['Experience'].plot(kind='box')

# EDA
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,anchor='C')
plt.title('Information')
df.groupby('Risk_Flag').count()['Id'].plot(kind='pie',labels=['Non-Defaulter','Defaulter'],autopct='%1.1f%%',ax=ax1,figsize=(10,10))
plt.xlabel('% of Defaulters')
plt.ylabel('')
plt.legend(loc='right',bbox_to_anchor=(0.7,0,1,1))
ax2 = fig.add_subplot(2,1,2,anchor='S')
df.groupby('Risk_Flag').count()['Id'].plot(kind='bar',ax=ax2)
plt.xlabel('Defaulters')
plt.ylabel('Count')
for index,value in enumerate(df.groupby('Risk_Flag').count()['Id']):
    plt.text(index-0.08,value+10000,str(value))
plt.ylim(0,250000)
plt.show()

df.groupby('Married/Single').count()['Id'].plot(kind='pie',startangle=0,labels=['Married','Single'],autopct='%1.1f%%',colors=['Pink','Teal'])
plt.ylabel('')
plt.xlabel('Marital Status')
plt.title('Total % of Customers who are married/single')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))
plt.show()

df.loc[df['Risk_Flag'] == 1].groupby('Married/Single').count()['Id']

marital_status = df.loc[df['Risk_Flag'] == 1].groupby('Married/Single').count()['Id']
marital_status.plot(kind='pie',startangle=0,labels=['Married','Single'],autopct='%1.1f%%')
plt.ylabel('')
plt.xlabel('Marital Status')
plt.title('Loan Defaulters % by Marital Status')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))
plt.show()

house_ownership_count = df.groupby('House_Ownership').count()['Id']
house_ownership_count = [231898,7184,12918]

sns.countplot(data=df,x='House_Ownership',hue='House_Ownership',)
plt.text(-0.4,235000,str(231898))
plt.text(0.9,10000,str(7184))
plt.text(2.14,16000,str(12918))
plt.title('House Ownership of All Customers')
plt.ylim(0,250000)

df.loc[df['Risk_Flag'] == 1].groupby('House_Ownership').count()['Id']

sns.countplot(data=df.loc[df['Risk_Flag'] == 1],x='House_Ownership',hue='House_Ownership')
plt.text(-0.4,30000,str(29121))
plt.text(0.9,2000,str(1160))
plt.text(2.14,1800,str(715))
plt.ylim(0,35000)
plt.title('House Ownership of Loan Defaulting Customers')
plt.show()

df.groupby('House_Ownership').count()['Id'].plot(kind='pie',startangle=0,autopct='%1.1f%%',figsize=(5,5))
plt.ylabel('')
plt.xlabel('House Ownership')
plt.title('House Ownership % of Customers')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))

df.loc[df['Risk_Flag'] == 1].groupby('House_Ownership').count()['Id'].plot(kind='pie',startangle=0,autopct='%1.1f%%',figsize=(5,5))
plt.ylabel('')
plt.title('House Ownership % of Loan Defaulters')
plt.xlabel('House Ownership')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))

car_ownership = df.groupby('Car_Ownership').count()['Id']
car_ownership

sns.countplot(data=df,x='Car_Ownership',hue='Car_Ownership')
plt.text(-0.3,177000,str(176000))
plt.text(1.1,78000,str(78000))
plt.ylim(0,200000)
plt.title('Car Ownership of All Customers')
plt.show()

car_ownership.plot(kind='pie',startangle=0,labels=['Do not Own','Own'],autopct='%1.1f%%')
plt.xlabel('Car Ownership')
plt.title('Car Ownership % of All Customers')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))
plt.show()

car_ownership_default = df.loc[df['Risk_Flag'] == 1].groupby('Car_Ownership').count()['Id']
car_ownership_default.plot(kind='pie',startangle=0,labels=['Do not Own','Own'],autopct='%1.1f%%')
plt.ylabel('')
plt.xlabel('Car Ownership')
plt.title('Car Ownership % of Loan Defaulting Customers')
plt.legend(loc='best',bbox_to_anchor=(1,0,0.5,0.5))
plt.show()

df1 = df.loc[df['Risk_Flag'] == 1].groupby(['STATE','Risk_Flag']).count()
df1.rename(columns={'Id':'Total_Defaulters'},inplace=True)
df1.reset_index(inplace=True)
df1[['STATE','Total_Defaulters']]

df2 = df.groupby('STATE').count()
df2.rename(columns={'Id':'Total_Loans'},inplace=True)
df2.reset_index(inplace=True)

df_total_loans = df2[['STATE','Total_Loans']].sort_values(by='Total_Loans',ascending=False)[:10]
df_total_loans.plot(kind='bar',x='STATE',figsize=(10,8))
plt.title('Top 10 States from where most loans were taken')
plt.xlabel('Number of Loans')
plt.ylabel('State')
for index,value in enumerate(df_total_loans['Total_Loans'][:10]):
    plt.text(index-0.28,value+100,str(value))
plt.show()

defaulter_percent_per_state = (df1['Total_Defaulters']/df2['Total_Loans']).round(4)*100
state_defaulters_percentage=pd.DataFrame(
    data=zip(df1['STATE'],defaulter_percent_per_state),
    columns=['STATE','Defaulter_Percentage']
)
df_dps = state_defaulters_percentage.sort_values(by='Defaulter_Percentage',ascending=False)[:10]
df_dps.plot(kind='bar',figsize=(10,8),x='STATE')
plt.title('Top 10 States in Defaulting Loan')
plt.ylabel('% of Loans Defaults')
plt.xlabel('State')
for index,value in enumerate(df_dps['Defaulter_Percentage'][:10]):
    plt.text(index-0.2,value+0.2,str(round(value,2)))
plt.legend(loc='best')
plt.show()

df3 = df.groupby('CITY').count()
df3.rename(columns={'Id':'Total_Loans'},inplace=True)
df3.reset_index(inplace=True)
df3[['CITY','Total_Loans']]

#top10 cities in number of loans
df3_ = df3[['CITY','Total_Loans']].sort_values(
                        by='Total_Loans',ascending=False)[:10]
df3_.plot(kind='bar',x='CITY',figsize=(8,6))
plt.title('Top 10 Cities with highest number of Loans taken')
plt.xlabel('City')
plt.ylabel('Number of Loan')
for index,value in enumerate(df3_['Total_Loans']):
    plt.text(index-0.25,value+30,str(int(value)))
plt.legend(loc='best')
plt.show()

df4 = df.loc[df['Risk_Flag'] == 1].groupby('CITY').count()
df4.rename(columns={'Id':'Total_Defaulters'},inplace=True)
df4.reset_index(inplace=True)
df4[['CITY','Total_Defaulters']]

defaulter_percent_per_city = (df4['Total_Defaulters']/df3['Total_Loans']).round(4)*100
city_defaulters_percentage=pd.DataFrame(
    data=zip(df3['CITY'],defaulter_percent_per_city),
    columns=['CITY','Defaulter_Percentage'])
city_defaulters_percentage

city_defaulters_percentage.sort_values(by='Defaulter_Percentage',ascending=False)[:10].plot(kind='bar',x='CITY',figsize=(10,6))
plt.title('Top 10 Cities in Defaulting Loans')
plt.xlabel('City')
plt.ylabel('% of Loan Defaults')
plt.legend(loc='best')
top_10_vals = city_defaulters_percentage['Defaulter_Percentage'].sort_values(ascending=False)[:10]
for index,value in enumerate(top_10_vals):
    plt.text(index-0.2,value+0.5,str(round(value,2)))
plt.show()

df_profession_loan_count = df.groupby('Profession').count()['Id'].sort_values(ascending=False)
df_plc = df_profession_loan_count.reset_index()
df_plc.rename(columns= {'Id':'Loan_Count'},inplace=True)
df_plc[:10].plot(kind='bar',x='Profession',figsize=(10,10))
plt.legend(loc='best')
plt.title('Top 10 Professions who took Loan')
plt.xlabel('Loan Count')
plt.ylabel('Profession')
for index,value in enumerate(df_plc['Loan_Count'][:10]):
    plt.text(index-0.2,value+50,str(value))
plt.show()   

profession = df.groupby(['Profession']).mean()[['Income','Risk_Flag']]

# plotting top 10 profession_group with higher income
profession_top10_income = profession['Income'].sort_values(ascending=False)[:15]
profession_top10_income.plot(kind='barh',figsize=(10,10))
plt.title('Top 15 Profession with higher Income (mean)')
plt.xlabel('Profession')
plt.ylabel('Income')

for index,value in enumerate(profession_top10_income):
    plt.text(value-900000,index-0.1,str(int(value)))
plt.legend(loc='best')
plt.show()

df_ = df.loc[df['Risk_Flag'] == 1].groupby(['Profession']).mean()[['Income']].sort_values(by='Income',ascending=False)
df_.sort_values(by='Income',ascending=False)[:15].plot(kind='barh',figsize=(10,10))
plt.title('Mean Income of Top 15 Loan Defaulting Professions') 
plt.xlabel('Income')
plt.ylabel('Profession')
for index,value in enumerate(df_['Income'][:15]):
    plt.text(value-900000,index-0.1,str(int(value)))
plt.legend(loc='best')
plt.show()

# Resampling the Data with Random Oversampler
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(random_state=42,sampling_strategy=0.45)
X = df.iloc[:,:-1]
y = df['Risk_Flag']

X_sampled,y_sampled = sampler.fit_resample(X,y)

from collections import Counter
print(Counter(y),Counter(y_sampled))

df_ = pd.concat([X_sampled,y_sampled],axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.title('Defaulter % Before Sampling')
df.groupby('Risk_Flag').count()['Id'].plot(kind='pie',labels=['Non-Defaulter','Defaulter'],autopct='%1.1f%%',ax=ax1,figsize=(10,10))
plt.xlabel('% of Defaulters')
plt.ylabel('')

ax2 = fig.add_subplot(122)
plt.title('Defaulter % After Sampling')
df_.groupby('Risk_Flag').count()['Id'].plot(kind='pie',labels=['Non-Defaulter','Defaulter'],autopct='%1.1f%%',ax=ax2,figsize=(10,10))
plt.xlabel('% of Defaulters')
plt.ylabel('')
plt.legend(loc='right',bbox_to_anchor=(0.7,0,1,1))

# Encoding the Categorical data
from sklearn.preprocessing import LabelEncoder

cols_to_encode= ['Married/Single','House_Ownership','Car_Ownership','Profession','CITY','STATE']
labelencoder = LabelEncoder()
for col in cols_to_encode:
    df_[col] = labelencoder.fit_transform(df_[col])
    
df_.isna().sum()

# Dropping Id as it's not needed in prediction
df_.drop(['Id'],axis=1,inplace=True)
df_

X = df_.iloc[:,:-1]
y = df_['Risk_Flag']

# Splitting the dataset into training and test set
from sklearn. model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.metrics import f1_score,classification_report,plot_confusion_matrix,plot_roc_curve

# Prediction
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(f'F1 Score: {f1_score(y_test,pred)}\n')
print(classification_report(y_test,pred))
plot_confusion_matrix(estimator=dt,X=X_test,y_true=y_test)    

plot_roc_curve(estimator=dt,X=X_test,y=y_test)
plt.plot([0,1],[0,1],"--",c='black')
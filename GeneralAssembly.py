# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:38:35 2020

@author: Prince
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import scipy
import datetime


dataset = pd.read_csv('kickstart.csv', engine='python')

#Missing Values

dataset.isnull().sum()
dataset = dataset[dataset.location.notnull()]
dataset.isnull().sum()
dataset = dataset[dataset['reward levels'].notnull()] #just remove, hard to determine

dataset.pledged =  dataset.pledged.fillna(0) #some had 0 pledges same assumption

dataset.isnull().sum() #now we have cleared missing values


dataset.status.value_counts()
dataset = dataset[dataset.status != 'live']


#PRELIMINARY Analysis

duplicated = dataset[dataset.duplicated()] #78 duplicates

dataset = dataset.drop_duplicates()

dataset.groupby('status')['status'].value_counts()
#54% successful
#46% successful

dataset.goal.mean() #10593 average


#FEATURE ENGINEERING
fail = ['failed', 'canceled', 'suspended']

dataset['status'] = ['failed' if i in fail else 'successful' for i in dataset.status]
#LOCATION
def cleaner(i):
        try:
            return str(i).split(',')[1].lstrip() 
        except:
            str(i).split(',')[0].lstrip() 
            
dataset['location'] = dataset.location.map(cleaner)

#DATE
datemaker = lambda i: datetime.datetime.strptime((' '.join(i.split()[:5])).replace(',', ''), '%a %d  %b %Y %H:%M:%S')
#sparse date into datetime

dataset['funded date'] = dataset['funded date'].map(datemaker) #get full funded date

dataset['start date'] = [i - datetime.timedelta(j) for i, j in zip(dataset['funded date'], dataset['duration'])] #full start date of campaign

#date and month

dataset['full_date'] = [str(i).split()[0] for i in dataset['start date']]

def datemaker_date(d):
    year,month,date = [int(i) for i in d.split('-')]
    launch = datetime.date(year,month,date)
    return launch.strftime('%a')

dataset['day_of_launch'] = dataset['full_date'].map(datemaker_date)

def datemaker_month(d):
    year,month,date = [int(i) for i in d.split('-')]
    launch = datetime.date(year,month,date)
    return launch.strftime('%b')

dataset['month_of_launch'] = dataset['full_date'].map(datemaker_month)


def datemaker_time(t):
    hour,minute,second = [int(i) for i in t.split(':')]
    time = datetime.time(hour,minute,second)
    return time.strftime('%p')


dataset['time_of_launch'] = [str(i).split()[1] for i in dataset['start date']]


dataset['AM/PM'] = dataset.time_of_launch.map(datemaker_time).astype('category').map(str)



dataset.info()

#NAME
dataset['length_of_name'] = [len(i.split()) for i in dataset.name] #length of name important?



   
def special_name(n):
    special = ['[','$','#','%','@','=','+','/','&','Ɛ','™','©','?','!','¡','.','*',']','»','«','=']
    if n.isupper():
        return 'yes'
    if any(i in n for i in special):
        return 'yes'
    if any(i.isupper() for i in n.split()):
        return 'yes'
    else:
        return 'no'
    
dataset['special_name'] = dataset['name'].map(special_name)
#special names useful to grab attention according google ads



dataset= dataset.drop(['funded date', 'start date', 'full_date', 'name', 'project id', 'url', 'reward levels', 'subcategory' ], axis = 1)

dataset['pledged'].mean() 

dataset.iloc[:, [0,1,-1,-3,-5,-6]] = dataset.iloc[:, [0,1,-1,-3,-5, -6]].astype('category')




plt.hist(dataset['backers'], bins = 50)
print(skew(dataset['backers'].values, axis = 0, bias = False))
#right skew of backers
#skewness of 87 means highly skewed
scipy.stats.normaltest(dataset['backers'])
#p value < than level of significance, hence not even normally distributed



print(skew(dataset['duration'], bias = False))
#skew > 1 hence not normal
scipy.stats.normaltest(dataset['duration'])
#p value < than level of significance, hence not even normally distributed
#reject null hypothesis of normal

dataset['status'].value_counts()


#Hypothesis

#status by day_of_launch
day_success = pd.DataFrame(dataset.groupby('day_of_launch')['status'].value_counts())
day_success.groupby('day_of_launch').apply(lambda i: 100 * i/i.sum())
day_success['percentage'] = (day_success['successful']/(day_success['failed'] + day_success['successful']))
day_success.percentage.sort_values()
#Tues, Monday, Wed
 


month_success  = pd.DataFrame(dataset.groupby('month_of_launch')['status'].value_counts().unstack()) #first quatter of month
month_success['percentage'] = month_success['successful']/(month_success['failed'] + month_success['successful'])
mdf = pd.DataFrame(month_success.percentage.sort_values(ascending = False))
#dec, jan feb area + April

#success by duration

dataset.groupby('status')['duration'].mean() #43 for failed 38 for succesful

#success by goal
dataset.groupby('status')['goal'].mean()
# $16762 failed, $5524 successful
#lower the goal, the more likely it will be succesful


#success by category

category_success = dataset.groupby('category')['status'].value_counts().unstack()
category_success['percentage'] = category_success['successful']/(category_success['failed'] + category_success['successful']) 
category_success.percentage.sort_values()
#Dance, Theatre, Music, 
#least are fashion, tech

#success by updates
dataset.groupby('status')['updates'].mean()
#success - 7
#fail - 1
# need more updates to potential donors, engagement

#success by length of name

namecounter = lambda i: len(i.split())
dataset.groupby('status')['length_of_name'].mean()
#not much diff, 6 and 6

#success by location

dataset = dataset[dataset.location.notnull()]

dataset.location = ['USA' if len(i) == 2 else i for i in dataset.location] #for smaller numpy object
location_success = pd.DataFrame(dataset.groupby('location')['status'].value_counts().unstack())
#top locations are CA, NY, IL, TX, MA, WA, 

#but most succesful based on launch then success are Jordan, Indonesia, Singapore, Iceland, Tunisia
#US states RI, VT, MT

#ANALYSIS 


x = dataset.iloc[:, [0, 3,9, 8, 10, 11,12,14,15,16 ] ].values
y = dataset.iloc[:, 2].values

k = dataset.iloc[:, [0, 3,9, 8, 10, 11,12,14,15,16 ] ]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse= False), [0,-1,-3,-5,-4])], remainder='passthrough')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x = np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x,y, test_size = 0.8)



#FEATURE SCALING


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 37:] = sc.fit_transform(x_train[:, 37:])
x_test[:, 37:] = sc.transform(x_test[:, 37:])

#models RANDOM FOREST, XGBOOST, BAYES

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
Random_classifier = RandomForestClassifier(n_estimators = 500, 
                                           criterion = 'entropy',
                                           max_depth = 25,
                                           min_samples_split = 3,
                                           random_state = 0)
Random_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = Random_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score 

Random_accuracy = accuracy_score(y_test, y_pred)
#84 % accuracy

'''
#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
naive_classifier = GaussianNB()
naive_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred2 = naive_classifier.predict(x_test)

naive_accuracy = accuracy_score(y_test, y_pred2)

#K - NEAREST
# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
k_nearest_classifier = KNeighborsClassifier(n_neighbors = 150, metric = 'minkowski', p = 2)
k_nearest_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred3 = k_nearest_classifier.predict(x_test)

k_nearest_accuracy = accuracy_score(y_test, y_pred3)


#XG-BOOST
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xg_classifier = XGBClassifier()
xg_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred4 = xg_classifier.predict(x_test)

xg_accuracy = accuracy_score(y_test, y_pred4)


#GRID SEARCH
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 500],
               'max_depth': [10,20, 25],
               'min_samples_split': [1.5,3]}]
   
gs = GridSearchCV(estimator = Random_classifier,
                  param_grid = parameters,
                  scoring = 'accuracy',
                  cv = 6,
                  n_jobs = -1)

gs = gs.fit(x_train, y_train)
best_accuracy = gs.best_score_
best_params = gs.best_params_     
'''

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Random_classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#after the best accuracy lets get feature importances
feature_importances = Random_classifier.feature_importances_
cat_one_hot_att = ['category', 'special_name', 'AM/PM', 'day_of_launch', 'month_of_launch']
num_attributes = ['goal', 'comments', 'updates', 'duration', 'length_of_name']
attributes = num_attributes + cat_one_hot_att
vals = sorted(zip(feature_importances, attributes), key=lambda x: x[0], reverse=True)
df = pd.DataFrame(vals)
df.iloc[:, -1] = df.iloc[:, -1].replace({i : k for i, k in enumerate(cat_one_hot_att)})

#splits help determine classes, most effectively
    
plt.scatter(df[0], df[1])
trial =  dataset[dataset['status']== 'successful']
plt.scatter(trial['time_of_launch'], trial['status'])




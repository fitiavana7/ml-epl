import pandas as pd
import matplotlib.pyplot as plt

#get the datasets
df = pd.read_csv('epl.csv')

#drop unused values
df = df.drop(['Referee', 'HTHG','HTAG'],axis=1)

#seeing values
df.head()

#describing columns
df.columns

#show stats data
df.describe()

#function to get the point of home team
def getHomePoint(s):
    if(s=='H'):
        return 3
    elif(s=='D'):
        return 1
    else:
        return 0

#applying the function and create Home_Points column
df['Home_Points'] = df.FTR.apply(getHomePoint)
df.Home_Points = df.Home_Points.astype(float)

#droping unused values
df = df.drop(['HomeTeam','AwayTeam','FTHG','FTAG','HTR','Date','Season','FTR'],axis=1)

df.head()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#drop duplicated and null data
df = df.drop_duplicates()
df = df.dropna()

#spliting data
y = df.Home_Points
X = df.drop(['Home_Points'],axis=1)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#using SVC
model = SVC(kernel='rbf',random_state=0)
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#it show score of 0.55

#using GaussianNB
model = GaussianNB()
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#it show score of 0.52

#using random forest classifier
model = RandomForestClassifier()
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#it show score of 0.53

l = []
for i in range(1,20):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train , y_train)
    score = model.score(X_test,y_test)
    l.append(score)
plt.plot(l , '>')
plt.xlabel('neighbor number')
plt.ylabel('number accuracy')
plt.title('showing number with most accuracy')
plt.show()

#the most accurate of KNN , n=18
model = KNeighborsClassifier(n_neighbors=18)
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#score of 0.51

#logistic regression
model = LogisticRegression(max_iter=5000)
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#score of 0.55 , eqal SVC

#using decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train , y_train)
score = model.score(X_test,y_test)
score
#score of 0.42

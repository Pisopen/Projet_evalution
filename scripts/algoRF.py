import pandas as pd
import sklearn
import joblib
import streamlit as st
import sqlite3
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

labelencoder = preprocessing.LabelEncoder()

#Import dataset
dataGamesSales= pd.read_csv("/app/data/Video_Games_Sales_as_at_22_Dec_2016.csv",sep=',',encoding='iso-8859-1')

#Clean dataset
dataGamesSales= dataGamesSales.dropna()

## Encoding qualitative data to quantitative data
dataGamesSales['Platform'] = dataGamesSales['Platform'].astype('category').cat.codes
dataGamesSales['Genre'] = dataGamesSales['Genre'].astype('category').cat.codes
dataGamesSales['Rating'] = dataGamesSales['Rating'].astype('category').cat.codes
dataGamesSales['Publisher'] = labelencoder.fit_transform(dataGamesSales['Publisher'])
dataGamesSales['Developer'] = labelencoder.fit_transform(dataGamesSales['Developer'])


# So we see that User_Score row has object type even if the content can be transform to int without enconding the values
# We'll multiply User_Score by 10 sor we have them like xx/100
def convertUserScore(value):
    value = float(value)*10
    return float(value)
dataGamesSales.User_Score = dataGamesSales.User_Score.apply(convertUserScore)

#Change float clolumns to int
dataGamesSales['NA_Sales'] = dataGamesSales['NA_Sales'].astype(int)
dataGamesSales['EU_Sales'] = dataGamesSales['EU_Sales'].astype(int)
dataGamesSales['JP_Sales'] = dataGamesSales['JP_Sales'].astype(int)
dataGamesSales['Other_Sales'] = dataGamesSales['Other_Sales'].astype(int)
dataGamesSales['Global_Sales'] = dataGamesSales['Global_Sales'].astype(int)

# We willn't need the name of the game
# So we gonna delete it with sqlite3 for example

#tranfser to database sqlite3
#connection to database
conn = sqlite3.connect("/app/database/:memory")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS dataGamesSales")
dataGamesSales.to_sql("dataGamesSales", conn, index=False)

#get data to dataframe
select_query = "SELECT Platform,Year_of_Release,Genre,Publisher,NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales,Critic_Score,Critic_Count,User_Score,User_Count,Developer,Rating from dataGamesSales"
dataGamesSalesClean = pd.read_sql_query(select_query,conn)

#export to csv
dataGamesSalesClean.to_csv("/app/data/dataGamesSalesClean.csv", encoding="utf-8",index =False)


# dataGamesSales = dataGamesSales.drop(['Name'], axis =1)

# Now that we have quantitative value we can isolate our targets
Y = pd.read_sql_query("SELECT Publisher from dataGamesSales",conn)

select_query_X = "SELECT Platform,Year_of_Release,Genre,NA_Sales,EU_Sales,JP_Sales,Other_Sales,Global_Sales,Critic_Score,Critic_Count,User_Score,User_Count,Developer,Rating from dataGamesSales"
# and our features
X = pd.read_sql_query(select_query_X,conn)

#RandomForest algo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
randomforest = rf(n_estimators=90, max_features=10, max_depth=25)
randomforest.fit(X_train, Y_train)

#test
Y_pred = randomforest.predict(X_test)
prob = randomforest.predict(X_test)

# print(accuracy_score(Y_test, Y_pred)*100)
st.title("Video games publisher guesser ")
st.text("Accuracy score: ")
st.text(accuracy_score(Y_test, Y_pred)*100)

#export model
joblib.dump(randomforest,'/app/data/modelGame.pkl')


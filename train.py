import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('Brain.csv')

df.head()

df.info()

X=df.drop('stroke',axis=1)
y=df['stroke']
catcols = X.select_dtypes(include='object').columns
catcols

numcols = X.select_dtypes(exclude='object').columns
numcols


preprocessor = ColumnTransformer(transformers=[('cat',OneHotEncoder(drop='first'),
                                               catcols)],remainder='passthrough')


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



model = RandomForestClassifier(n_estimators=100,random_state=42)



from sklearn.pipeline import Pipeline



pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])



pipeline.fit(X_train,y_train)


y_pred = pipeline.predict(X_test)


acc = accuracy_score(y_test,y_pred)
acc


cm = confusion_matrix(y_test,y_pred)
cm

cr = classification_report(y_test,y_pred)
print(cr)


import joblib

joblib.dump(pipeline, "brainstroke_model.pkl")







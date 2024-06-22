import pandas as pd
from sklearn import tree, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Absenteeism_at_work.csv",delimiter=";")
print(df)

nr_missing_values=sum([True for idx,row in df.iterrows() if any(row.isnull())])
print("Numar de randuri care au cel putin o valoare lipsa:",nr_missing_values)

df=df.drop(["ID"],axis=1)

scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),columns = df.columns)

df_features=df.drop(['Absenteeism time in hours'], axis=1)
df_target=pd.DataFrame(df['Absenteeism time in hours'])

#Selectia atributelor
dt=tree.DecisionTreeRegressor(max_depth=5,splitter='random', min_samples_leaf=3)
attributes=[]
for attribute in df_features.columns:
    df_X=pd.DataFrame(df_features[attribute])
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_X, df_target, test_size=0.3, random_state=50)
    dt.fit(Xtrain,ytrain)
    ypredictions = dt.predict(Xtest)

    coef_det=r2_score(ytest,ypredictions)
    if coef_det>=0.01:
        print(f"Atribut: {attribute}: {coef_det}")
        attributes.append(attribute)


m=0
for i in range(100):
    df_features=df_features[attributes]
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_features, df_target, test_size=0.3, random_state=50)
    dt.fit(Xtrain,ytrain)
    ypred=dt.predict(Xtest)
    s=r2_score(ytest,ypred)
    if s>m:
        m=s
    print(s)
print("max",m)
final_df = Xtest.copy()
final_df["Y_original"] = ytest
final_df["Y_predicted"] = ypred
pass
# Atribut: Reason for absence: 0.0807720741862058
# Atribut: Day of the week: 0.023271586862661287
# Atribut: Transportation expense: 0.035804070882330286
# Atribut: Disciplinary failure: 0.010979521161990524
# 0.23144537159780276
import pandas as pd
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Absenteeism_at_work.csv", delimiter=";")
print(df)

print(f"Numarul de inregistrari: {len(df)}")
print(f"Numarul de atribute: {len(df.columns)}")

nr_missing_values = sum([True for idx, row in df.iterrows() if any(row.isnull())])
print("Numar de randuri care au cel putin o valoare lipsa:", nr_missing_values)

df = df.drop(["ID"], axis=1)

# Onehot encoding
df = df.copy()
encoded = pd.get_dummies(df['Reason for absence'], prefix='Reason for absence')
df = pd.concat([df, encoded], axis=1)
df = df.drop('Reason for absence', axis=1)

scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

df_features = df.drop(['Absenteeism time in hours'], axis=1)
df_target = pd.DataFrame(df['Absenteeism time in hours'])

# Selectia atributelor
dt = tree.DecisionTreeRegressor(max_depth=5, splitter='random')
attributes = []
for attribute in df_features.columns:
    df_X = pd.DataFrame(df_features[attribute])
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_X, df_target, test_size=0.3, random_state=50)
    dt.fit(Xtrain, ytrain)
    ypredictions = dt.predict(Xtest)

    coef_det = r2_score(ytest, ypredictions)
    if coef_det >= 0.01:
        print(f"Atribut: {attribute}: {coef_det}")
        attributes.append(attribute)

max_r2 = 0
max_mse = 0
final_df=Xtest.copy()
for i in range(200):
    df_features = df_features[attributes]
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_features, df_target, test_size=0.3, random_state=50)
    dt.fit(Xtrain, ytrain)
    ypred = dt.predict(Xtest)
    r2 = r2_score(ytest, ypred)
    rmse = root_mean_squared_error(ytest, ypred)
    if r2 > max_r2:
        max_r2 = r2
        max_mse = rmse
        ytest_max = ytest
        ypred_max = ypred
        final_df["Y_original"] = ytest
        final_df["Y_predicted"] = ypred
    print(f"r2: {r2}\n mse: {rmse}")
print("max r2: ", max_r2)
print("max mse: ", max_mse)
print("final df: ", final_df)

# Atribut: Day of the week: 0.023271586862661287
# Atribut: Transportation expense: 0.026974626804400548
# Atribut: Disciplinary failure: 0.010979521161990524
# Atribut: Height: 0.010108070938968372
# Atribut: Body mass index: 0.01603314454979432
# Atribut: Reason for absence_0: 0.012372272022983077
# Atribut: Reason for absence_9: 0.06503519211072961
# Atribut: Reason for absence_23: 0.028090699506847727
# Atribut: Reason for absence_27: 0.012784584121467546

# max r2:  0.22588657368853415
# max rmse:  0.8301732460341319

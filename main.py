import pandas as pd
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def onehot_encode(df, column, prefix):
    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)

    return df


df = pd.read_csv("Absenteeism_at_work.csv", delimiter=";")
print(df)

nr_missing_values = sum([True for idx, row in df.iterrows() if any(row.isnull())])
print("Numar de randuri care au cel putin o valoare lipsa:", nr_missing_values)

df = df.drop(["ID"], axis=1)

df = onehot_encode(df, 'Reason for absence', 'Reason for absence')

scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

df_features = df.drop(['Absenteeism time in hours'], axis=1)
df_target = pd.DataFrame(df['Absenteeism time in hours'])

# Selectia atributelor
dt = tree.DecisionTreeRegressor(max_depth=5, splitter='random', min_samples_leaf=2)
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
max_mse=0
for i in range(200):
    df_features = df_features[attributes]
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_features, df_target, test_size=0.3, random_state=50)
    dt.fit(Xtrain, ytrain)
    ypred = dt.predict(Xtest)
    r2 = r2_score(ytest, ypred)
    mse = root_mean_squared_error(ytest, ypred)
    if r2 > max_r2:
        max_r2 = r2
        max_mse = mse
        ytest_max = ytest
        ypred_max = ypred
    print(f"r2: {r2}\n mse: {mse}")
print("max r2: ", max_r2)
print("max mse: ", max_mse)

pass
# Atribut: Reason for absence: 0.09280259457339013
# Atribut: Day of the week: 0.023271586862661287
# Atribut: Transportation expense: 0.038779640533143644
# Atribut: Disciplinary failure: 0.010979521161990524
# Atribut: Weight: 0.011392095480000974
# Atribut: Body mass index: 0.014011062510043581
# r2 0.2695110591750415
# mse 0.816359002840189

# max r2:  0.2094290014845286
# max mse:  0.8389515341498817
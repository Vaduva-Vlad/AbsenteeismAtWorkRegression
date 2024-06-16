import pandas as pd

df=pd.read_csv("Absenteeism_at_work.csv",delimiter=";")
print(df)

nr_missing_values=sum([True for idx,row in df.iterrows() if any(row.isnull())])
print("Numar de randuri care au cel putin o valoare lipsa:",nr_missing_values)

#Eliminarea valorilor eronate.
df_filtered = df[df['Reason for absence'] <= 21]
print(df_filtered)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("Absenteeism_at_work.csv", delimiter=";")

reasons=np.array(df['Reason for absence'])

plt.figure(figsize=(10,4))
df['Reason for absence'].value_counts().plot.bar(rot=0)
plt.show()

plt.figure(figsize=(10,4))
df['Transportation expense'].value_counts().plot.bar(rot=0)
plt.show()

hoursabsent=np.array(df['Absenteeism time in hours'])
plt.hist(hoursabsent)
plt.show()

print(df['Absenteeism time in hours'].median())
print(df['Absenteeism time in hours'].mean())
print(df['Absenteeism time in hours'].std())

print(df['Transportation expense'].median())
print(df['Transportation expense'].mean())
print(df['Transportation expense'].std())
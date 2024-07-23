import pandas as pd
df = pd.read_csv('Data/synth.csv')
df = df.iloc[ : 100, : ]
df.to_csv('Data/del_later.csv', index=False)
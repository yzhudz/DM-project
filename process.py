import pandas as pd
df = pd.read_csv('games.csv')
clean_df = pd.DataFrame({'k':df['TEAM_ID_home'].values,'v1':df['PTS_home'].values})
print(clean_df['v1'].isna().sum())
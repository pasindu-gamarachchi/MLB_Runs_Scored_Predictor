#Pasindu Gamarachchi

import pandas as pd
import numpy as np
from scipy import stats
from scipy import stats, special
from sklearn import model_selection, metrics, linear_model, datasets, feature_selection
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
#pylab inline
#pylab.rcParams['figure.figsize'] = (12.0, 10.0)

#df = pd.read_csv('baseball_reference_2016_clean.csv', index_col =0)
#df['Capacity']= ''

df = pd.read_csv('baseball_reference_2016_clean.csv', index_col =0)
df['Capacity'] =''
df['Dist_to_CF'] =''
df['Home_Team_WP']=''
df['Away_Team_WP']=''

## Team Info Data Frame
# Collected Data from related articles
Teams = df.away_team.unique();
Teams.sort();
d = {'Teams':Teams}
Team_info =pd.DataFrame(d)
Cap = [ 48686,41084,45971,37755,41649,40615,
         42319,35051,46897,41299,41168,37903,
         45477,56000,36742,41900,38885,41922,
         47309,47170,43651,38362,40209,41915,
         47715,45529,31042,48114,49282,41339] # Stadium Capacity
CF_Distance = [407,400,410,420,400,400,
               404,410,415,420,409,410,
               396,395,407,400,404,408,
               408,400,401,399,396,399,
               401,400,404,400,400,402] # Distance to CF
Win_Percent = [0.4500, 0.3377, 0.6053, 0.5454, 0.6579, 0.4935,
               0.3718, 0.6053, 0.4805, 0.5065, 0.5256, 0.5263,
               0.4103, 0.5443, 0.5325, 0.4474, 0.3289, 0.5263,
               0.4868, 0.4416, 0.4304, 0.4744, 0.4231, 0.6203,
               0.5065, 0.5263, 0.4211, 0.6538, 0.5316, 0.5897]  # Team Win Percentage at mid-point of the Season
Team_info['Capacity'] =Cap
Team_info['Distance to CF'] =CF_Distance
Team_info['Win_Percent'] = Win_Percent

# Combining Data from Team Info to Main DF
for i in range (0,len(df)):
    for j in range (0, len(Team_info)):
        if ( df.loc[i,'home_team'] == Team_info.loc[j,'Teams']):
             df.loc[i,'Capacity'] = Team_info.loc[j,'Capacity']
             df.loc[i,'Distance_to_CF'] = Team_info.loc[j,'Distance to CF']
             df.loc[i,'Home_Team_WP'] = Team_info.loc[j,'Win_Percent']
        if ( df.loc[i,'away_team'] == Team_info.loc[j,'Teams']):
            df.loc[i,'Away_Team_WP'] = Team_info.loc[j,'Win_Percent']
        
# Create New Columns using relevant columns in Data Frame
df['Packed_Ratio'] = df['attendance']/df['Capacity'] # Packed Ratio
df['WinPerc_Diff'] = (df['Home_Team_WP'] - df['Away_Team_WP']) 
df['WinPerc_Diff'] = pd.to_numeric(df.WinPerc_Diff, errors= 'coerce')
df['Home_Team_WP'] = pd.to_numeric(df.Home_Team_WP, errors= 'coerce')
df['Away_Team_WP'] = pd.to_numeric(df.Away_Team_WP, errors= 'coerce')


# Clean Packed_Ratio
for i in range(0, len(df)):
    if (df.loc[i,'Packed_Ratio'] > 1):
        df.loc[i,'Packed_Ratio'] = 1

# Categorical Data Processing
sky_dummies = pd.get_dummies(df.sky, prefix ='sky')
day_of_week_dummies = pd.get_dummies(df.day_of_week, prefix = 'DofWeek')
field_type_unique = df.field_type.unique()
df['field_type_n'] = df.field_type.map({field_type_unique[0]:0,field_type_unique[1]:1})  # On grass : 0, On Turf : 1
game_type_unique = df.game_type.unique()
df['game_type_n'] = df.game_type.map({game_type_unique[1]:0,game_type_unique[0]:1}) # Day Game : 0, Night Game : 1
df =pd.concat([df, sky_dummies, day_of_week_dummies], axis =1)

# Separate Training Data
games_on_lastday =df[df['date']=='2016-07-10']
df_train = df.loc[:1341]
df_predict = df.loc[1342:]
training_data_decrip = df_train.describe()
training_data_decrip.to_csv('TrainingDataDescription.xls', sep='\t')
data_corr = df_train.corr()
data_corr.to_csv('Data_Corr.xls', sep='\t')

# Extract Averages from Training Data To be input into predictive model
# as this data will be part of the prediction. Runs scored, hits and errors will
# be known after the game, therefore I used the average runs scored by each team at home
# as the input for the predictive model
Team_info['Ave_Runs_Home'] =''
Team_info['Ave_Hits_Home'] =''
Team_info['Ave_Errors_Home'] =''
Team_info['Ave_Runs_Away'] =''
Team_info['Ave_Hits_Away'] =''
Team_info['Ave_Errors_Away'] =''

df_groupby_home_stats = df_train.groupby('home_team')
df_groupby_away_stats = df_train.groupby('away_team')
for i in range(0, len(Team_info)):
    temp = df_groupby_home_stats.get_group(Team_info.loc[i,'Teams']);
    Team_info.loc[i,'Ave_Runs_Home'] = (temp['home_team_runs']).mean()
    Team_info.loc[i,'Ave_Hits_Home'] = (temp['home_team_hits']).mean()
    Team_info.loc[i,'Ave_Errors_Home'] = (temp['home_team_errors']).mean()

    temp_away = df_groupby_home_stats.get_group(Team_info.loc[i,'Teams']);
    Team_info.loc[i,'Ave_Runs_Away'] = (temp['away_team_runs']).mean()
    Team_info.loc[i,'Ave_Hits_Away'] = (temp['away_team_hits']).mean()
    Team_info.loc[i,'Ave_Errors_Away'] = (temp['away_team_errors']).mean()


# Populating Future In-Game Data with averages for hits, errors, and runs
for i in range (1342,len(df_predict)): # New index starts at 1342
    for j in range (0, len(Team_info)):
        if ( df_predict.loc[i,'home_team'] == Team_info.loc[j,'Teams']):
            df_predict.loc[i,'home_team_hits'] = Team_info.loc[j,'Ave_Hits_Home']
           #df_predict.loc[i,'home_team_runs'] = Team_info.loc[j,'Ave_Runs_Home']
            df_predict.loc[i,'home_team_errors'] = Team_info.loc[j,'Ave_Errors_Home']

        if ( df.loc[i,'away_team'] == Team_info.loc[j,'Teams']):
            df_predict.loc[i,'away_team_hits'] = Team_info.loc[j,'Ave_Hits_Away']
           #df_predict.loc[i,'home_team_runs'] = Team_info.loc[j,'Ave_Runs_Home']
            df_predict.loc[i,'away_team_errors'] = Team_info.loc[j,'Ave_Errors_Away']

# Drop Columns 
df_train_types = df_train.dtypes
drop_list = []
df_cols = df.columns
#
for key, val in enumerate(df_train_types):
    if  (val == 'object'):
        drop_list.append(key)
for i in df_cols:
    df_cols_to_drop = df.columns

#df_train_new = df_train.drop(['away_team','date','field_type','game_type', 'field_type', 'home_team', 'start_time',
#                             'venue', 'day_of_week', 'wind_direction', 'sky', 'season', 'home_team_outcome', 'Dist_to_CF'], axis =1)

df_train = df_train.drop(['away_team','date','field_type','game_type', 'field_type', 'home_team', 'start_time',
                              'venue', 'day_of_week', 'wind_direction', 'sky', 'season', 'home_team_outcome', 'Dist_to_CF',
                          'away_team_runs','home_team_runs'], axis =1)

df_predict = df_predict.drop(['away_team','date','field_type','game_type', 'field_type', 'home_team', 'start_time',
                              'venue', 'day_of_week', 'wind_direction', 'sky', 'season', 'home_team_outcome', 'Dist_to_CF',
                          'away_team_runs','home_team_runs'], axis =1)

df_train =df_train.replace([np.inf, -np.inf], np.nan)
df_train =df_train.dropna()

df_predict =df_predict.replace([np.inf, -np.inf], np.nan)
df_predict =df_predict.dropna()

df_train.to_csv('TrainingData.csv')
df_predict.to_csv('TestingData.csv')

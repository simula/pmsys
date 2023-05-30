import mysql.connector as connection
import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
import numpy as np
import json

def getDataFromMysql():
    """
    Extracts two tables from a MySQL database

    Arguments:
        -
    
    Returns:
        Two Pandas Dataframes of gps and wellness data
    """

    config = open('mysql-config.json')
    data = json.load(config)

    try:
        mydb = connection.connect(host=data["host"], database =data["database"],user=data["user"], passwd=data["passwd"],use_pure=True)
        query_gps = "Select * from LH_session;"
        query_pmsys = "Select * from daily_features;"
        df_pmsys = pd.read_sql(query_pmsys,mydb)
        df_gps = pd.read_sql(query_gps,mydb)
        mydb.close() 
    except Exception as e:
        mydb.close()
        print(str(e))

    return df_gps, df_pmsys


def restructureDataset(df_gps, df_pmsys):
    """
    Reformats the gps and wellness dataframes to a format where players
    are sequentually after each other 

    Arguments:
        df_gps: pandas dataframe.
        df_pmsys: pandas dataframe. 

    Returns:
        Re-formatted dataframes
    """
    
    df_gps.rename(columns={'Player_name': 'player_name'}, inplace=True)
    unique_players = df_gps["player_name"].unique()
    pmsys_features = ["date","player_name", "daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "injury_ts", "weekly_load"]
    df_pmsys = df_pmsys[pmsys_features]
    all_players_gps = []
    all_players_pmsys = []

    for i in range(len(unique_players)):
        df_player_gps = pd.DataFrame(df_gps[df_gps["player_name"] == unique_players[i]])
        df_player_pmsys = pd.DataFrame(df_pmsys[df_pmsys["player_name"] == unique_players[i]])
        
        df_player_gps = pd.DataFrame(df_player_gps.drop_duplicates(subset='Date', keep="first"))
        
        df_player_gps['Date'] = pd.to_datetime(df_player_gps['Date'])
        df_player_gps.set_index('Date', inplace=True)

        new_date_range = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
        df_player_gps = df_player_gps.reindex(new_date_range, fill_value=0)

        df_player_gps = df_player_gps[(df_player_gps.player_name != 0).idxmax():]
        df_player_gps = df_player_gps.iloc[::-1]
        df_player_gps = df_player_gps[(df_player_gps.player_name != 0).idxmax():]
        df_player_gps = df_player_gps.iloc[::-1]
        df_player_gps['date'] = df_player_gps.index
        
        df_player_pmsys = df_player_pmsys.reset_index(drop=True)        
        first = df_player_pmsys["readiness"].first_valid_index()
        last = df_player_pmsys["readiness"].last_valid_index()
        df_player_pmsys = df_player_pmsys.iloc[first:last]
        
        df_player_pmsys['date'] = pd.to_datetime(df_player_pmsys['date'])
        df_player_pmsys.set_index('date', inplace=True)
        
        all_players_gps.append(df_player_gps)
        all_players_pmsys.append(df_player_pmsys)
        
    return all_players_gps, all_players_pmsys


def joinDatasets(all_players_gps, all_players_pmsys):
    """
    Merges the gps and wellness dataframes to one dataframe only keeping
    the entries where both wellness and gps data exists

    Arguments:
        all_players_gps: pandas dataframe.
        all_players_pmsys: pandas dataframe. 

    Returns:
        Merged Dataframe
    """
    all_players_all_features = []

    for i in range(len(all_players_gps)):
        temp = pd.merge(all_players_pmsys[i], all_players_gps[i], left_index=True, right_index=True)
        #temp = all_players_pmsys[i].join(all_players_gps[i], lsuffix="_left", rsuffix="_right", how='right')
        temp = temp.drop(['Session_Id', 'player_name_y'], axis=1)
        temp['Team_name'] = temp['Team_name'].replace(to_replace=0, method='ffill')
        temp['Team_name'] = temp['Team_name'].replace(to_replace=0, method='bfill')
        temp['HIR_count'] = temp['HIR_count'].replace("\r", "").astype(int)
        temp['sleep_duration'] = temp['sleep_duration'].apply(np.floor)
        nanToZero = ["Total_distance", "Average_running_speed", "Top_speed", "HIR_count", "Metabolic_power"]
        temp[nanToZero] = temp[nanToZero].replace(0,np.NaN)

        x = list(temp["Total_distance"])
        for i in range(len(temp["Total_distance"])):
            x[i] = float(x[i])/getTenthNumber(str(x[i]).split(".")[0])
        temp["Total_distance"] = x
        temp = temp.reset_index(drop=True) 
        
        all_players_all_features.append(temp) 
    
    return all_players_all_features


def iterativeImpute(df, df_class, df_continues):
    """
    Imputes the missing values using iterative imputer

    Arguments:
        df: pandas dataframe.
        df_class: list of discreet values. 
        df_continues: list of continous values

    Returns:
        imputed dataset
    """

    imp_classification = IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=20))
    imp_regression = IterativeImputer(estimator=Ridge(alpha=0.5))

    df_class = pd.DataFrame(data=df[df_class], columns=df_class)
    df_continues = pd.DataFrame(data=df[df_continues], columns=df_continues)
    imp_classification.fit(df_class)
    imp_regression.fit(df_continues)

    df_class = pd.DataFrame(imp_classification.transform(df_class), columns = df_class.columns)
    df_continues = pd.DataFrame(imp_regression.transform(df_continues), columns = df_continues.columns)

    result = pd.concat([df_class, df_continues], axis=1, join='inner')

    return result


def getTenthNumber(nr):
    """
    returns the tenth number

    Arguments:
        nr: int

    Returns:
        tenth number
    """
    number = "1"
    for i in range(len(nr)-1):
        number = number+"0"
    return int(number)
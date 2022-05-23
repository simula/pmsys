import pandas as pd
import uuid
from tsai.all import *

sheets_to_read = ['Fatigue', 'Mood', 'Readiness', 'SleepDurH', 'SleepQuality', 'Soreness', 'Stress']


def read_file_to_dataframe(file_name):
    return pd.read_excel(file_name, sheet_name=sheets_to_read, index_col='Date')

def create_player_dictionairy(dataframes_dict_from_file):
    players_dataframes_dict = {}
    players_ids_array = []
    for feature, dataframe in dataframes_dict_from_file.items():
        for col in dataframe:
            if not col in players_ids_array:
                players_ids_array.append(col)

        for player_id in players_ids_array:
            if not player_id in players_dataframes_dict:
                players_dataframes_dict[player_id] = pd.DataFrame()

            player_dataframe = dataframe[player_id]
            player_dataframe.name = feature
            players_dataframes_dict[player_id] = pd.concat((players_dataframes_dict[player_id], player_dataframe),
                                                           axis=1)
            players_dataframes_dict[player_id]['Date'] = players_dataframes_dict[player_id].index
    return players_dataframes_dict

def save_file(filename, modified_df):
    writer = pd.ExcelWriter(filename)
    for key in modified_df:
        modified_df[key].to_excel(writer, sheet_name=key)
    writer.save()
    writer.close()

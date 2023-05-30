import mysql.connector as connection
import pandas as pd
from createDatasetUtil import *


def main():
    """
    Creates the dataset by extracting data from a MySQL database and processing it
    """

    df_gps, df_pmsys = getDataFromMysql()
    list_of_players_pmsys = list(df_pmsys["player_name"].unique())
    df_gps = df_gps[df_gps['Player_name'].isin(list_of_players_pmsys)]
    list_of_players_gps = list(df_gps["Player_name"].unique())
    df_pmsys = df_pmsys[df_pmsys['player_name'].isin(list_of_players_gps)]

    all_players_gps, all_players_pmsys = restructureDataset(df_gps, df_pmsys)

    dataset = joinDatasets(all_players_gps, all_players_pmsys)

    df_class = ["fatigue", "mood", "stress", "sleep_quality", "sleep_duration", "readiness", "soreness", "injury_ts"]
    df_continues = ["daily_load", "Total_distance", "Average_running_speed", "Top_speed", "HIR_count", "Metabolic_power"]
    for i in range(len(dataset)):
        dataset[i][df_class+df_continues] = iterativeImpute(dataset[i][df_class+df_continues], df_class, df_continues)

    dataset = pd.concat(dataset)
    dataset["HIR_count"] = dataset["HIR_count"].abs()
    dataset.rename(columns = {'Total_distance':'Total_distance', 'Average_running_speed':'Average_running_speed', 'Top_speed':'Top_speed', 'Metabolic_power': 'Metabolic_power'}, inplace = True)

    filename = '../data/mysql_dataset/complete_dataset'
    dataset.to_csv(filename)

    print("")
    print("Dataset created successfully")
    print("")

if __name__ == "__main__":
    main()
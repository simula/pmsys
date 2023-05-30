import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ml_util import * 
import warnings
warnings.filterwarnings("ignore")

def get_dataset():
    """
    creates train and test set for shap values
    
    Returns:
        train/test split
    """
    path = "../data/mysql_dataset/complete_dataset"
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis=1)
    df['date'] =  pd.to_datetime(df['date'])
    df['day'] = df.date.dt.dayofweek.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)
    df["readiness"] -= 1

    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    #columnNames = ["fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts"]
    #columnNames = ["readiness"]
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    i=20
    all_but_one = players[:i] + players[i+1:]
    train = df[df['player_name_x'].isin(all_but_one)]
    test = df.loc[df['player_name_x'] == players[i]]
    val = df.loc[df['player_name_x'] == players[1]]
    train = train[columnNames]
    test = test[columnNames]
    val = val[columnNames]

    train_scalar = StandardScaler()
    train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames)
    test = pd.DataFrame(train_scalar.transform(test), columns=columnNames)

    lag = 1
    train = addReadinessLag(train, lag)
    test = addReadinessLag(test, lag)
    val = addReadinessLag(val, lag)
    features = train.columns.tolist()[:-1]
    target = train.columns.tolist()[-1]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    X_val = val[features]
    y_val = val[target]

    return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    """
    Create plots for SHAP values for xgboost, linear regression, and tree model
    """

    X_train, y_train, X_test, y_test, X_val, y_val = get_dataset()

    modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
    modelLin = fitLinearReg(X_train, y_train)
    modelTree = fitTree(X_train, y_train)

    """XGBOOST SHAP VALUES """
    X_testt = X_test.copy()
    sample_size = len(X_testt)
    X_sampled = X_testt.sample(sample_size, random_state=41) #10
    explainer = shap.TreeExplainer(modelXGB)
    shap_values = explainer.shap_values(X_sampled)
    shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
    plt.savefig("experiment_plots/XBOOST_SHAP_20")
    plt.close()
    """------------------- """

    """Tree regressor SHAP VALUES """
    X_testt = X_test.copy()
    sample_size = len(X_testt)
    X_sampled = X_testt.sample(sample_size, random_state=10)
    explainer = shap.TreeExplainer(modelTree)
    shap_values = explainer.shap_values(X_sampled)
    shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
    plt.savefig("experiment_plots/TREE_SHAP_20")
    plt.close()
    """------------------- """

    """Linear regression SHAP VALUES """
    X_testt = X_test.copy()
    sample_size = len(X_testt)
    X_sampled = X_testt.sample(sample_size, random_state=10)
    explainer = shap.LinearExplainer(modelLin, X_testt)
    shap_values = explainer.shap_values(X_sampled)
    shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
    plt.savefig("experiment_plots/LINEAR_REGRESSION_SHAP_20")
    plt.close()
    """------------------- """


if __name__ == "__main__":
    main()


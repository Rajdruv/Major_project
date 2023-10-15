import pandas as pd

def change_bikram_data(x):
    if x.day==30:
        return x.replace(day=29, month=8)
    elif x.day==31:
        return x.replace(day=30, month=8)
    elif x.day==2:
        return x.replace(day=31, month=8)
    elif x.day==4:
        return x.replace(day=1, month=9)
    elif x.day==5:
        return x.replace(day=2, month=9)
    elif x.day==6:
        return x.replace(day=3, month=9)
    elif x.day==7:
        return x.replace(day=4, month=9)
    elif x.day==8:
        return x.replace(day=5, month=9)
    else:
        return x

def change_dhruv_data(x):
    if x.day==6:
        return x.replace(day=6, month=9)

def change_gaurav_data(x):
    if x.day==6:
        return x.replace(day=7, month=9)
    elif x.day==7:
        return x.replace(day=8, month=9)
    elif x.day==8:
        return x.replace(day=9, month=9)
    elif x.day==10:
        return x.replace(day=10, month=9)
    elif x.day==11:
        return x.replace(day=11, month=9)
    elif x.day==12:
        return x.replace(day=12, month=9)
    elif x.day==13:
        return x.replace(day=13, month=9)
    elif x.day==17:
        return x.replace(day=14, month=9)
    elif x.day==19:
        return x.replace(day=15, month=9)
    elif x.day==21:
        return x.replace(day=16, month=9)
    elif x.day==22:
        return x.replace(day=17, month=9)
    else:
        return x

def change_sujit_data(x):
    if x.day==6:
        return x.replace(day=18, month=9)
    elif x.day==7:
        return x.replace(day=19, month=9)
    elif x.day==8:
        return x.replace(day=20, month=9)
    elif x.day==9:
        return x.replace(day=21, month=9)
    elif x.day==10:
        return x.replace(day=22, month=9)
    elif x.day==11:
        return x.replace(day=23, month=9)
    elif x.day==12:
        return x.replace(day=24, month=9)
    elif x.day==13:
        return x.replace(day=25, month=9)
    elif x.day==17:
        return x.replace(day=26, month=9)
    elif x.day==18:
        return x.replace(day=27, month=9)
    else:
        return x


def preprocess():
    bikram_data = pd.read_csv("bikram_log.csv")[["time", "query_time", "rows_examined"]]
    dhruv_data = pd.read_csv("dhruv_log.csv")[["time", "query_time", "rows_examined"]]
    gaurav_data = pd.read_csv("gaurav_log.csv")[["time", "query_time", "rows_examined"]]
    sujit_data = pd.read_csv("sujith_log.csv")[["time", "query_time", "rows_examined"]]

    bikram_data["time"] = pd.to_datetime(
        bikram_data["time"], format='ISO8601'
    )
    dhruv_data["time"] = pd.to_datetime(dhruv_data["time"], format='ISO8601')
    gaurav_data["time"] = pd.to_datetime(
        gaurav_data["time"], format='ISO8601'
    )
    sujit_data["time"] = pd.to_datetime(sujit_data["time"], format='ISO8601')

    bikram_data["time"] = bikram_data["time"].apply(change_bikram_data)
    dhruv_data["time"] = dhruv_data["time"].apply(change_dhruv_data)
    gaurav_data["time"] = gaurav_data["time"].apply(change_gaurav_data)
    sujit_data["time"] = sujit_data["time"].apply(change_sujit_data)

    preprocessed_sum = (
        pd.concat([bikram_data, dhruv_data, gaurav_data, sujit_data])
        .groupby(["time"])
        .sum()
        .reset_index()
    )

    bikram_data = pd.read_csv("bikram_log.csv")
    dhruv_data = pd.read_csv("dhruv_log.csv")
    gaurav_data = pd.read_csv("gaurav_log.csv")
    sujit_data = pd.read_csv("sujith_log.csv")
    all_data = pd.concat([bikram_data, dhruv_data, gaurav_data, sujit_data])
    all_data.to_csv("all_data.csv")
    preprocessed_sum.to_csv("preprocessed_sum.csv")

    return 0


def get_hour(x):
    date = str(x.year) + ":" + str(x.month) + ":" + str(x.day) + " " + str(x.hour)
    return date


def remove_outliers_iqr(df, multiplier=1.5):
    Q1 = df["query_time"].quantile(0.25)
    Q3 = df["query_time"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df_no_outliers = df[
        (df["query_time"] >= lower_bound) & (df["query_time"] <= upper_bound)
    ]
    return df_no_outliers


def remove_outliers():
    preprocessed_sum = pd.read_csv("preprocessed_sum.csv", parse_dates=True)
    print(preprocessed_sum.head())
    preprocessed_sum["time"] = pd.to_datetime(
        preprocessed_sum["time"], format='ISO8601'
    )

    preprocessed_sum["hour"] = preprocessed_sum["time"].apply(get_hour)
    no_outliers = remove_outliers_iqr(preprocessed_sum)
    no_outliers["time"] = pd.to_datetime(no_outliers["time"], format="%Y-%m-%d %H")
    no_outliers = no_outliers[['hour', 'query_time']]
    no_outliers.set_index("hour", inplace=True)
    return no_outliers


def split_data():
    no_outliers = remove_outliers()
    train_size = int(len(no_outliers) * 0.8)
    train, test = no_outliers[:train_size], no_outliers[train_size:]
    return train, test


def preprocess_main():
    preprocess()
    train, test = split_data()
    return train, test


if __name__ == "__main__":
    preprocess_main()

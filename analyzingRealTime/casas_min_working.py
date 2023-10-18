import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing as skpp


RESULTS_DIR = Path("results/CASAS")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = RESULTS_DIR / ''.join(np.random.choice(
    list(string.ascii_lowercase + string.digits),
    size=6,
    replace=True
))
EXP_DIR.mkdir(exist_ok=False)
print(f"Experiment directory: {EXP_DIR}")

summary_file = open(EXP_DIR / "summary.txt", "w+")

RND_SEED = 42


sedentary_cols = ["Bed_Toilet_Transition", "Relax", "Sleep", "Work"]
active_cols = ["Cook", "Eat", "Enter_Home", "Leave_Home", "Personal_Hygiene", "Wash_Dishes"]
activity_cols = sedentary_cols + active_cols

datetimenm = "DateTime"
seccosnm = "SecondsCos"
secsinnm = "SecondsSin"
doycosnm = "DoyCos"
doysinnm = "DoySin"
idnm = "Person"
activitynm = "Activity"
agenm = "age"
age_grpnm = "age_group"
gendernm = "gender"
racenm = "race"
static_feats = [agenm, gendernm, racenm]
real_label = "Real"

sens1cols_og = ['Bathroom', 'Bedroom', 'DiningRoom', 'Hall', 'Kitchen', 'LivingRoom', 'Office',
             'OutsideDoor']
sens1cols = [col + "1" for col in sens1cols_og]

sens2cols_og = ['Bathroom', 'Bed', 'Bedroom', 'Chair', 'DiningRoom', 'Entry', 'FrontDoor', 'Hall',
             'Kitchen', 'LivingRoom', 'Office', 'OutsideDoor', 'Refrigerator', 'Sink', 'Stove',
             'Toilet']
sens2cols = [col + "2" for col in sens2cols_og]

time_cols = [seccosnm, secsinnm, doycosnm, doysinnm]

train_time_feats = time_cols + sens1cols + sens2cols
time_feats = train_time_feats + activity_cols

all_feats = time_feats + static_feats
all_train_feats = train_time_feats + static_feats
binary_label_col = "Sedentary"

def gender_to_int(series:pd.Series):
    return series.map({"male": 1, "female": 0})

def gender_to_str(series:pd.Series):
    return series.map({1:"Male", 0 : "Female"})

def time_to_cycle(data, max_value):
    data = 2*np.pi * (data / max_value)
    return np.cos(data), np.sin(data)

def idnm_to_int(df):
    df[idnm] = df[idnm].str.lower().str.replace('tm','').str.replace('*', '', regex=False)
    df[idnm] = df[idnm].astype(int)
    return df

def get_static_df():
    idnm_og = "pt_id"
    #all collumns
    """['pt_id', 'age', 'gender', 'race', 'marital status', 'ed level',
       'total occupants', 'home type', 'community type', 'study arm']"""
    df = pd.read_csv("al/static_features.csv")
    df = df[[idnm_og] + static_feats]
    df.rename(columns={idnm_og : idnm}, inplace=True)

    df = df[df[agenm] != "?"]
    df[racenm] = df[racenm] == "white"
    df[gendernm] = gender_to_int(df[gendernm])
    df = idnm_to_int(df)
    return df


def get_time_data():
    dfs = []
    columns = ['Date', 'Time', 'Place1', 'Place2', 'Signal', activitynm]
    data_dir = Path("al/tmdata")
    for i, file in enumerate(data_dir.glob("*")):
        df = pd.read_csv(file, header=None, delimiter=' ')
        df.columns = columns

        ##datetime
        df[datetimenm] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.drop(columns=['Date', 'Time'], inplace=True)

        timeofday = df[datetimenm] - df[datetimenm].dt.normalize()
        timeofday = timeofday / pd.Timedelta(seconds=1)
        totalSecondsPerDay = pd.Timedelta(days=1) / pd.Timedelta(seconds=1)
        df[seccosnm], df[secsinnm] = time_to_cycle(timeofday, totalSecondsPerDay)

        doy = df[datetimenm].dt.dayofyear
        df[doycosnm], df[doysinnm] = time_to_cycle(doy, 365 + df[datetimenm].dt.is_leap_year)

        ##id
        df[idnm] = file.stem

        ##signal
        df.drop(columns=['Signal'], inplace=True)
        # df['Signal'] = df['Signal'].map({"ON" : 1., "OFF" : 0.})

        assert not df.isna().any().any()
        dfs.append(df)

    df = pd.concat(dfs)

    #cleansing the unexpected
    df = df[df[activitynm].isin(activity_cols)]
    df = df[df['Place1'].isin(sens1cols_og)]
    df = df[df['Place2'].isin(sens2cols_og)]

    ##activities
    df[binary_label_col] = df[activitynm].isin(sedentary_cols).astype(float)
    activity = pd.get_dummies(df[activitynm])
    for act_col in activity_cols:
        if act_col not in activity.columns:
            print(f"Activity {act_col} not in columns")
            activity[act_col] = 0
    df[activity.columns] = activity
    # df.drop(columns=[activitynm], inplace=True)

    ##sensor placements
    p1 = pd.get_dummies(df['Place1'])
    p1.columns = [c + "1" for c in p1.columns]
    for s1 in sens1cols:
        if s1 not in p1.columns:
            print(f"Sensor1 {s1} not in columns")
            p1[s1] = 0
    df[p1.columns] = p1

    p2 = pd.get_dummies(df['Place2'])
    p2.columns = [c + "2" for c in p2.columns]
    for s2 in sens2cols:
        if s2 not in p2.columns:
            print(f"Sensor2 {s2} not in columns")
            p2[s2] = 0
    df[p2.columns] = p2
    # df.drop(columns=['Place1', 'Place2'], inplace=True)

    df = idnm_to_int(df)

    assert not df.isna().any().any()
    df.reset_index(inplace=True, drop=True)
    return df

def get_data( scaler=None):
    static_df = get_static_df()
    time_df = get_time_data()
    df = pd.merge(static_df, time_df)
    if scaler is None:
        scaler = skpp.MinMaxScaler()
    df[all_feats] = scaler.fit_transform(df[all_feats])
    return df, scaler



if __name__ == "__main__":
    df, scaler = get_data()
    pass
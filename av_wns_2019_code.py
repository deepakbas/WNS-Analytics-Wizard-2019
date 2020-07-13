import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import lightgbm as lgb
import itertools
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import metrics
import re
import datetime
from datetime import timedelta

#Loading the data
train_df = pd.read_csv("train_wns.csv",parse_dates=["impression_time"])
train_df["impression_date"] = train_df["impression_time"].dt.date
train_df = train_df.loc[(train_df["impression_date"] <= datetime.date(2018, 12, 11))].reset_index(drop=True)
view_log = pd.read_csv("view_log_wns.csv",parse_dates=["server_time"])
item_data = pd.read_csv("item_data_wns.csv")
test_df = pd.read_csv("test_wns.csv",parse_dates=["impression_time"])
test_df["impression_date"] = test_df["impression_time"].dt.date

test_df["is_click"] = -99
view_item_log = pd.merge(view_log, item_data, how='left', on="item_id")
view_item_log["server_time_date"] = view_item_log["server_time"].dt.date

del item_data
del view_log
train_df["impression_time_ord"] = train_df["impression_time"].apply(lambda x: time.mktime(x.timetuple()))
test_df["impression_time_ord"] = test_df["impression_time"].apply(lambda x: time.mktime(x.timetuple()))
view_item_log["server_time_ord"] = view_item_log["server_time"].apply(lambda x: time.mktime(x.timetuple()))

train_df = train_df.sort_values(by="impression_time_ord").reset_index(drop=True)
test_df = test_df.sort_values(by="impression_time_ord").reset_index(drop=True)
view_item_log = view_item_log.sort_values(by="server_time_ord").reset_index(drop=True)

##
ID_y_col = ["impression_id","is_click"]
raw_col = [col for col in train_df.columns if col not in ID_y_col]
train_df["train_set"] = 1
test_df["train_set"] = 0
test_id = test_df["impression_id"].values

#Concat Train and Test data for creating different features as both Train and Test have similar distribution
all_df = pd.concat([train_df, test_df])
gc.collect()
gc.collect()

#LE
le_col = ["os_version"]
indexer = {}
for col in tqdm(le_col):
    if col == 'impression_id': continue
    _, indexer[col] = pd.factorize(all_df[col])

for col in tqdm(le_col):
    if col == 'impression_id': continue
    train_df[col] = indexer[col].get_indexer(train_df[col])
    test_df[col] = indexer[col].get_indexer(test_df[col])

all_df = pd.concat([train_df, test_df])
all_df.dtypes
gc.collect()
##
#LE
le_col = ["device_type"]
indexer = {}
for col in tqdm(le_col):
    if col == 'session_id': continue
    _, indexer[col] = pd.factorize(view_item_log[col])

for col in tqdm(le_col):
    if col == 'session_id': continue
    view_item_log[col] = indexer[col].get_indexer(view_item_log[col])
##
#Fastai module    
def add_datepart(df, fldname, drop=True, time=True):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Day', 'Week', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if time: attr = attr + ['Hour', 'Minute']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
#    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

add_datepart(all_df, "impression_time", drop=False)
add_datepart(view_item_log, "server_time", drop=False)

booleanDictionary = {True: 'TRUE', False: 'FALSE'}
all_df['impression_timeIs_month_end'] = all_df['impression_timeIs_month_end'].replace(booleanDictionary)
all_df['impression_timeIs_month_start'] = all_df['impression_timeIs_month_start'].replace(booleanDictionary)
view_item_log['server_timeIs_month_end'] = view_item_log['server_timeIs_month_end'].replace(booleanDictionary)
view_item_log['server_timeIs_month_start'] = view_item_log['server_timeIs_month_start'].replace(booleanDictionary)

bool_map = {"FALSE": 0, "TRUE": 1}
all_df["impression_timeIs_month_end"] = all_df["impression_timeIs_month_end"].map(bool_map)
all_df["impression_timeIs_month_start"] = all_df["impression_timeIs_month_start"].map(bool_map)
view_item_log["server_timeIs_month_end"] = view_item_log["server_timeIs_month_end"].map(bool_map)
view_item_log["server_timeIs_month_start"] = view_item_log["server_timeIs_month_start"].map(bool_map)

##Lag features, descriptive features of lags
all_df["total_time_secs"] = pd.to_timedelta(all_df["impression_time"]).dt.total_seconds()
view_item_log["total_time_secs"] = pd.to_timedelta(view_item_log["server_time"]).dt.total_seconds()
#all1
all_df["user_prev_time"] = all_df.groupby(["user_id"])["total_time_secs"].shift(1)
all_df["user_next_time"] = all_df.groupby(["user_id"])["total_time_secs"].shift(-1)
all_df["user_time_diffprev"] = all_df["total_time_secs"] - all_df["user_prev_time"]
all_df["user_time_diffnext"] = all_df["user_next_time"] - all_df["total_time_secs"]
all_df["user_time_diff1"] = all_df["user_time_diffprev"] - all_df["user_time_diffnext"]

gdf = all_df.groupby(["user_id"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff1_min", "user_time_diff1_max", "user_time_diff1_mean", "user_time_diff1_std"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log1
view_item_log["user_prev_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(1)
view_item_log["user_next_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(-1)
view_item_log["user_time_diffprev"] = view_item_log["total_time_secs"] - view_item_log["user_prev_time"]
view_item_log["user_time_diffnext"] = view_item_log["user_next_time"] - view_item_log["total_time_secs"]
view_item_log["user_time_diff1"] = view_item_log["user_time_diffprev"] - view_item_log["user_time_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff1_log_min", "user_time_diff1_log_max", "user_time_diff1_log_mean", "user_time_diff1_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()

gdf = view_item_log.groupby(["user_id", "server_time_date"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_time_diff1_log_min", "user_date_time_diff1_log_max", "user_date_time_diff1_log_mean", "user_date_time_diff1_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_diff1_log_min", "user_daywk_diff1_log_max", "user_daywk_diff1_log_mean", "user_daywk_diff1_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

##all1
gdf = all_df.groupby(["user_id", "app_code"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "app_code", "user_app_time_diff1_min", "user_app_time_diff1_max", "user_app_time_diff1_mean", "user_app_time_diff1_std"]
all_df = all_df.merge(gdf, on=["user_id", "app_code"], how="left")

gdf = all_df.groupby(["user_id", "app_code", "impression_timeDayofweek"])["user_time_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "app_code", "impression_timeDayofweek", "user_app_daywk_time_diff1_min", "user_app_daywk_time_diff1_max", "user_app_daywk_time_diff1_mean", "user_app_daywk_time_diff1_std"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_timeDayofweek"], how="left")
###
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()

##all2
all_df["user_prev_time2"] = all_df.groupby(["user_id"])["total_time_secs"].shift(2)
all_df["user_next_time2"] = all_df.groupby(["user_id"])["total_time_secs"].shift(-2)
all_df["user_time_diffprev2"] = all_df["user_prev_time"] - all_df["user_prev_time2"]
all_df["user_time_diffnext2"] = all_df["user_next_time2"] - all_df["user_next_time"]
all_df["user_time_diff2"] = all_df["user_time_diffprev2"] - all_df["user_time_diffnext2"]

gdf = all_df.groupby(["user_id"])["user_time_diff2"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff2_min", "user_time_diff2_max", "user_time_diff2_mean", "user_time_diff2_std"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log2
view_item_log["user_prev_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(2)
view_item_log["user_next_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(-2)
view_item_log["user_time_diffprev"] = view_item_log["total_time_secs"] - view_item_log["user_prev_time"]
view_item_log["user_time_diffnext"] = view_item_log["user_next_time"] - view_item_log["total_time_secs"]
view_item_log["user_time_diff2"] = view_item_log["user_time_diffprev"] - view_item_log["user_time_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_time_diff2"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff2_log_min", "user_time_diff2_log_max", "user_time_diff2_log_mean", "user_time_diff2_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id", "server_time_date"])["user_time_diff2"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_time_diff2_log_min", "user_date_time_diff2_log_max", "user_date_time_diff2_log_mean", "user_date_time_diff2_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["user_time_diff2"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_diff2_log_min", "user_daywk_diff2_log_max", "user_daywk_diff2_log_mean", "user_daywk_diff2_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()
##all3
all_df["user_prev_time3"] = all_df.groupby(["user_id"])["total_time_secs"].shift(3)
all_df["user_next_time3"] = all_df.groupby(["user_id"])["total_time_secs"].shift(-3)
all_df["user_time_diffprev3"] = all_df["user_prev_time2"] - all_df["user_prev_time3"]
all_df["user_time_diffnext3"] = all_df["user_next_time3"] - all_df["user_next_time2"]
all_df["user_time_diff3"] = all_df["user_time_diffprev3"] - all_df["user_time_diffnext3"]

gdf = all_df.groupby(["user_id"])["user_time_diff3"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff3_min", "user_time_diff3_max", "user_time_diff3_mean", "user_time_diff3_std"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log3
view_item_log["user_prev_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(3)
view_item_log["user_next_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(-3)
view_item_log["user_time_diffprev"] = view_item_log["total_time_secs"] - view_item_log["user_prev_time"]
view_item_log["user_time_diffnext"] = view_item_log["user_next_time"] - view_item_log["total_time_secs"]
view_item_log["user_time_diff3"] = view_item_log["user_time_diffprev"] - view_item_log["user_time_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_time_diff3"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff3_log_min", "user_time_diff3_log_max", "user_time_diff3_log_mean", "user_time_diff3_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id", "server_time_date"])["user_time_diff3"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_time_diff3_log_min", "user_date_time_diff3_log_max", "user_date_time_diff3_log_mean", "user_date_time_diff3_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["user_time_diff3"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_diff3_log_min", "user_daywk_diff3_log_max", "user_daywk_diff3_log_mean", "user_daywk_diff3_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()

##all4
all_df["user_prev_time4"] = all_df.groupby(["user_id"])["total_time_secs"].shift(4)
all_df["user_next_time4"] = all_df.groupby(["user_id"])["total_time_secs"].shift(-4)
all_df["user_time_diffprev4"] = all_df["user_prev_time3"] - all_df["user_prev_time4"]
all_df["user_time_diffnext4"] = all_df["user_next_time4"] - all_df["user_next_time3"]
all_df["user_time_diff4"] = all_df["user_time_diffprev4"] - all_df["user_time_diffnext4"]

gdf = all_df.groupby(["user_id"])["user_time_diff4"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff4_min", "user_time_diff4_max", "user_time_diff4_mean", "user_time_diff4_std"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log4
view_item_log["user_prev_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(4)
view_item_log["user_next_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(-4)
view_item_log["user_time_diffprev"] = view_item_log["total_time_secs"] - view_item_log["user_prev_time"]
view_item_log["user_time_diffnext"] = view_item_log["user_next_time"] - view_item_log["total_time_secs"]
view_item_log["user_time_diff4"] = view_item_log["user_time_diffprev"] - view_item_log["user_time_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_time_diff4"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff4_log_min", "user_time_diff4_log_max", "user_time_diff4_log_mean", "user_time_diff4_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id", "server_time_date"])["user_time_diff4"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_time_diff4_log_min", "user_date_time_diff4_log_max", "user_date_time_diff4_log_mean", "user_date_time_diff4_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["user_time_diff4"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_diff4_log_min", "user_daywk_diff4_log_max", "user_daywk_diff4_log_mean", "user_daywk_diff4_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()
##all5
all_df["user_prev_time5"] = all_df.groupby(["user_id"])["total_time_secs"].shift(5)
all_df["user_next_time5"] = all_df.groupby(["user_id"])["total_time_secs"].shift(-5)
all_df["user_time_diffprev5"] = all_df["user_prev_time4"] - all_df["user_prev_time5"]
all_df["user_time_diffnext5"] = all_df["user_next_time5"] - all_df["user_next_time4"]
all_df["user_time_diff5"] = all_df["user_time_diffprev5"] - all_df["user_time_diffnext5"]

gdf = all_df.groupby(["user_id"])["user_time_diff5"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff5_min", "user_time_diff5_max", "user_time_diff5_mean", "user_time_diff5_std"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log5
view_item_log["user_prev_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(5)
view_item_log["user_next_time"] = view_item_log.groupby(["user_id"])["total_time_secs"].shift(-5)
view_item_log["user_time_diffprev"] = view_item_log["total_time_secs"] - view_item_log["user_prev_time"]
view_item_log["user_time_diffnext"] = view_item_log["user_next_time"] - view_item_log["total_time_secs"]
view_item_log["user_time_diff5"] = view_item_log["user_time_diffprev"] - view_item_log["user_time_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_time_diff5"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_time_diff5_log_min", "user_time_diff5_log_max", "user_time_diff5_log_mean", "user_time_diff5_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id", "server_time_date"])["user_time_diff5"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_time_diff5_log_min", "user_date_time_diff5_log_max", "user_date_time_diff5_log_mean", "user_date_time_diff5_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["user_time_diff5"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_diff5_log_min", "user_daywk_diff5_log_max", "user_daywk_diff5_log_mean", "user_daywk_diff5_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()

###
##log features
#nunique
gdf = view_item_log.groupby(["user_id"])["device_type"].nunique().reset_index()
gdf.columns =["user_id", "device_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["session_id"].nunique().reset_index()
gdf.columns =["user_id", "sess_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["item_id"].nunique().reset_index()
gdf.columns =["user_id", "item_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["server_time_date"].nunique().reset_index()
gdf.columns =["user_id", "ser_dat_user_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["server_timeWeek"].nunique().reset_index()
gdf.columns =["user_id", "ser_wk_user_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["server_timeDayofweek"].nunique().reset_index()
gdf.columns =["user_id", "ser_daywk_user_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = view_item_log.groupby(["user_id"])["server_timeHour"].nunique().reset_index()
gdf.columns =["user_id", "ser_hr_user_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")

#session nunique over time
gdf = view_item_log.groupby(["user_id", "server_time_date"])["session_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_sess_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["session_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_sess_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeHour"])["session_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_timeHour", "user_hr_sess_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeHour"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeHour"], right_on=["user_id","server_timeHour"], how="left")

#item_id nunique over time
gdf = view_item_log.groupby(["user_id", "server_time_date"])["item_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_time_date", "user_date_item_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_time_date"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_date"], right_on=["user_id","server_time_date"], how="left")
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()

gdf = view_item_log.groupby(["user_id", "server_timeDayofweek"])["item_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_timeDayofweek", "user_daywk_item_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeDayofweek"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeDayofweek"], right_on=["user_id","server_timeDayofweek"], how="left")

gdf = view_item_log.groupby(["user_id", "server_timeHour"])["item_id"].nunique().reset_index()
gdf.columns = ["user_id", "server_timeHour", "user_hr_item_uniq"]
view_item_log = view_item_log.merge(gdf, on=["user_id", "server_timeHour"], how="left")
all_df = all_df.merge(gdf, left_on =["user_id","impression_timeHour"], right_on=["user_id","server_timeHour"], how="left")

#all
#number of impressions for each user
gdf = all_df.groupby(["user_id"])["impression_id"].count().reset_index()
gdf.columns = ["user_id", "user_impression_count"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

#log usercount
view_item_log['id'] = view_item_log.index

gdf = view_item_log.groupby(["user_id"])["id"].count().reset_index()
gdf.columns = ["user_id", "user_id_cnt"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")
all_df["user_log_cnt_r"] = all_df['user_impression_count'] / all_df['user_id_cnt']
#log unique countby overall count
all_df["user_device_uniq_log_r"] = all_df["device_uniq"]/all_df["user_id_cnt"]
all_df["user_sess_uniq_log_r"] = all_df["sess_uniq"]/all_df["user_id_cnt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
all_df["user_item_uniq_log_r"] = all_df["item_uniq"]/all_df["user_id_cnt"]
all_df["user_ser_tim_uniq_log_r"] = all_df["ser_dat_user_uniq"]/all_df["user_id_cnt"]
all_df["user_user_day_sess_uniq_log_r"] = all_df["user_date_sess_uniq"]/all_df["user_id_cnt"]

#number of impressions for each user and app used
gdf = all_df.groupby(["user_id", "app_code"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "user_app_impression_count"]
all_df = all_df.merge(gdf, on=["user_id","app_code"], how="left")

#ratio user_app vs user impressions
all_df["user_app_app_impr_cnt_r"] = all_df['user_app_impression_count'] / all_df['user_impression_count']
##
#number of impressions for each app
gdf = all_df.groupby(["app_code"])["impression_id"].count().reset_index()
gdf.columns = ["app_code", "app_impression_count"]
all_df = all_df.merge(gdf, on=["app_code"], how="left")
#
#ratio user_app vs app impressions
all_df["user_app_app_over_cnt_r"] = all_df['user_app_impression_count'] / all_df['app_impression_count']
#number of impressions by os
gdf = all_df.groupby(["os_version"])["impression_id"].count().reset_index()
gdf.columns = ["os_version", "os_impression_count"]
all_df = all_df.merge(gdf, on=["os_version"], how="left")
#number of impressions for each user and app used, os_version,
gdf = all_df.groupby(["user_id", "app_code", "os_version"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code","os_version", "user_app_os_impression_count"]
all_df = all_df.merge(gdf, on=["user_id","app_code","os_version"], how="left")

#number of impressions for each user and app used, is_4G,
gdf = all_df.groupby(["user_id", "app_code", "is_4G"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code","is_4G", "user_app_4g_impression_count"]
all_df = all_df.merge(gdf, on=["user_id","app_code","is_4G"], how="left")

#ratio user_app vs app impressions
all_df["user_app_4g_over_cnt_r"] = all_df['user_app_4g_impression_count'] / all_df['app_impression_count']
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()

#all_df = all_df.drop(['week_imp_count'],axis=1)

#weekcount by user
gdf = all_df.groupby(["user_id", "impression_timeWeek"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","impression_timeWeek", "user_week_impcnt"]
all_df = all_df.merge(gdf, on=["user_id","impression_timeWeek"], how="left")
sam_all_df = all_df.head(1000)

all_df["user_wkcnt_prev"] = all_df.groupby(["user_id"])["user_week_impcnt"].shift(1)
all_df["user_wkcnt_nxt"] = all_df.groupby(["user_id"])["user_week_impcnt"].shift(-1)
all_df["user_wkcnt_diffprev"] = all_df["user_week_impcnt"] - all_df["user_wkcnt_prev"]
all_df["user_wkcnt_diffnext"] = all_df["user_wkcnt_nxt"] - all_df["user_week_impcnt"]

all_df["user_wkcnt_prev2"] = all_df.groupby(["user_id"])["user_week_impcnt"].shift(2)
all_df["user_wkcnt_nxt2"] = all_df.groupby(["user_id"])["user_week_impcnt"].shift(-2)
all_df["user_wkcnt_diffprev2"] = all_df["user_wkcnt_prev"] - all_df["user_wkcnt_prev2"]
all_df["user_wkcnt_diffnext2"] = all_df["user_wkcnt_nxt2"] - all_df["user_wkcnt_nxt"]
##

#weekcount by user and app
gdf = all_df.groupby(["user_id", "app_code", "impression_timeWeek"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "impression_timeWeek", "user_app_week_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_timeWeek"], how="left")
sam_all_df = all_df.head(1000)

all_df["user_app_wkcnt_prev"] = all_df.groupby(["user_id"])["user_app_week_impcnt"].shift(1)
all_df["user_app_wkcnt_nxt"] = all_df.groupby(["user_id"])["user_app_week_impcnt"].shift(-1)
all_df["user_app_wkcnt_diffprev"] = all_df["user_app_week_impcnt"] - all_df["user_app_wkcnt_prev"]
all_df["user_app_wkcnt_diffnext"] = all_df["user_app_wkcnt_nxt"] - all_df["user_app_week_impcnt"]

all_df["user_app_wkcnt_prev2"] = all_df.groupby(["user_id"])["user_app_week_impcnt"].shift(2)
all_df["user_app_wkcnt_nxt2"] = all_df.groupby(["user_id"])["user_app_week_impcnt"].shift(-2)
all_df["user_app_wkcnt_diffprev2"] = all_df["user_app_wkcnt_prev"] - all_df["user_app_wkcnt_prev2"]
all_df["user_app_wkcnt_diffnext2"] = all_df["user_app_wkcnt_nxt2"] - all_df["user_app_wkcnt_nxt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
##
#impression_timeDayofweek count by user and app
gdf = all_df.groupby(["user_id", "app_code", "impression_timeDayofweek"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "impression_timeDayofweek", "user_app_dayofweek_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_timeDayofweek"], how="left")
sam_all_df = all_df.head(1000)
#sam_all_df = all_df.tail(1000)

all_df["user_app_weekbytot_r"] = all_df["user_app_dayofweek_impcnt"] / all_df["user_app_impression_count"]

all_df["user_app_daywkcnt_prev"] = all_df.groupby(["user_id"])["user_app_dayofweek_impcnt"].shift(1)
all_df["user_app_daywkcnt_nxt"] = all_df.groupby(["user_id"])["user_app_dayofweek_impcnt"].shift(-1)
all_df["user_app_daywkcnt_diffprev"] = all_df["user_app_dayofweek_impcnt"] - all_df["user_app_daywkcnt_prev"]
all_df["user_app_daywkcnt_diffnext"] = all_df["user_app_daywkcnt_nxt"] - all_df["user_app_dayofweek_impcnt"]

all_df["user_app_daywkcnt_prev2"] = all_df.groupby(["user_id"])["user_app_dayofweek_impcnt"].shift(2)
all_df["user_app_daywkcnt_nxt2"] = all_df.groupby(["user_id"])["user_app_dayofweek_impcnt"].shift(-2)
all_df["user_app_daywkcnt_diffprev2"] = all_df["user_app_daywkcnt_prev"] - all_df["user_app_daywkcnt_prev2"]
all_df["user_app_daywkcnt_diffnext2"] = all_df["user_app_daywkcnt_nxt2"] - all_df["user_app_daywkcnt_nxt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
#all_df = all_df.drop(['impression_timeMonth'],axis=1)
##
#impression_timeHour count by user and app
gdf = all_df.groupby(["user_id", "app_code", "impression_timeHour"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "impression_timeHour", "user_app_hr_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_timeHour"], how="left")
sam_all_df = all_df.head(1000)
#sam_all_df = all_df.tail(1000)
all_df["user_app_hrbytot_r"] = all_df["user_app_hr_impcnt"] / all_df["user_app_impression_count"]

all_df["user_app_hrcnt_prev"] = all_df.groupby(["user_id"])["user_app_hr_impcnt"].shift(1)
all_df["user_app_hrcnt_nxt"] = all_df.groupby(["user_id"])["user_app_hr_impcnt"].shift(-1)
all_df["user_app_hrcnt_diffprev"] = all_df["user_app_hr_impcnt"] - all_df["user_app_hrcnt_prev"]
all_df["user_app_hrcnt_diffnext"] = all_df["user_app_hrcnt_nxt"] - all_df["user_app_hr_impcnt"]

all_df["user_app_hrcnt_prev2"] = all_df.groupby(["user_id"])["user_app_hr_impcnt"].shift(2)
all_df["user_app_hrcnt_nxt2"] = all_df.groupby(["user_id"])["user_app_hr_impcnt"].shift(-2)
all_df["user_app_hrcnt_diffprev2"] = all_df["user_app_hrcnt_prev"] - all_df["user_app_hrcnt_prev2"]
all_df["user_app_hrcnt_diffnext2"] = all_df["user_app_hrcnt_nxt2"] - all_df["user_app_hrcnt_nxt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
sam_all_df = all_df.head(1000)

#
#impression_timeMinute count by user and app
gdf = all_df.groupby(["user_id", "app_code", "impression_timeMinute"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "impression_timeMinute", "user_app_min_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_timeMinute"], how="left")
sam_all_df = all_df.head(1000)
#sam_all_df = all_df.tail(1000)
gdf = all_df.groupby(["user_id", "impression_timeMinute"])["impression_id"].count().reset_index()
gdf.columns = ["user_id", "impression_timeMinute", "user_min_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "impression_timeMinute"], how="left")

all_df["user_app_minbytot_r"] = all_df["user_app_min_impcnt"] / all_df["user_app_impression_count"]

all_df["user_app_mincnt_prev"] = all_df.groupby(["user_id"])["user_app_min_impcnt"].shift(1)
all_df["user_app_mincnt_nxt"] = all_df.groupby(["user_id"])["user_app_min_impcnt"].shift(-1)
all_df["user_app_mincnt_diffprev"] = all_df["user_app_min_impcnt"] - all_df["user_app_mincnt_prev"]
all_df["user_app_mincnt_diffnext"] = all_df["user_app_mincnt_nxt"] - all_df["user_app_min_impcnt"]

all_df["user_app_mincnt_prev2"] = all_df.groupby(["user_id"])["user_app_min_impcnt"].shift(2)
all_df["user_app_mincnt_nxt2"] = all_df.groupby(["user_id"])["user_app_min_impcnt"].shift(-2)
all_df["user_app_mincnt_diffprev2"] = all_df["user_app_mincnt_prev"] - all_df["user_app_mincnt_prev2"]
all_df["user_app_mincnt_diffnext2"] = all_df["user_app_mincnt_nxt2"] - all_df["user_app_mincnt_nxt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
sam_all_df = all_df.head(1000)

##
#impression_time_ord count by user and app
gdf = all_df.groupby(["user_id", "app_code", "impression_time_ord"])["impression_id"].count().reset_index()
gdf.columns = ["user_id","app_code", "impression_time_ord", "user_app_time_impcnt"]
all_df = all_df.merge(gdf, on=["user_id", "app_code", "impression_time_ord"], how="left")
sam_all_df = all_df.head(1000)
#sam_all_df = all_df.tail(1000)
all_df["user_app_timebytot_r"] = all_df["user_app_time_impcnt"] / all_df["user_app_impression_count"]

all_df["user_app_timecnt_prev"] = all_df.groupby(["user_id"])["user_app_time_impcnt"].shift(1)
all_df["user_app_timecnt_nxt"] = all_df.groupby(["user_id"])["user_app_time_impcnt"].shift(-1)
all_df["user_app_timecnt_diffprev"] = all_df["user_app_time_impcnt"] - all_df["user_app_timecnt_prev"]
all_df["user_app_timecnt_diffnext"] = all_df["user_app_timecnt_nxt"] - all_df["user_app_time_impcnt"]

all_df["user_app_timecnt_prev2"] = all_df.groupby(["user_id"])["user_app_time_impcnt"].shift(2)
all_df["user_app_timecnt_nxt2"] = all_df.groupby(["user_id"])["user_app_time_impcnt"].shift(-2)
all_df["user_app_timecnt_diffprev2"] = all_df["user_app_timecnt_prev"] - all_df["user_app_timecnt_prev2"]
all_df["user_app_timecnt_diffnext2"] = all_df["user_app_timecnt_nxt2"] - all_df["user_app_timecnt_nxt"]
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
sam_all_df = all_df.head(1000)
###
all_df["user_app_cumcount"] = all_df.groupby(["user_id", "app_code"]).cumcount()
all_df["user_app_code_inv_cumcount"] = all_df.groupby(["user_id", "app_code"]).cumcount(ascending=False)
all_df["user_app_code_cumcount_ratio"] = all_df["user_app_cumcount"] / all_df["user_app_impression_count"]
all_df["user_app_code_inv_cumcount_ratio"] = all_df["user_app_code_inv_cumcount"] / all_df["user_app_impression_count"]
#all_df = all_df.drop("user_app_impression_count", axis=1)
all_df["user_min_cumcount"] = all_df.groupby(["user_id", "impression_timeMinute"]).cumcount()
all_df["user_min_inv_cumcount"] = all_df.groupby(["user_id", "impression_timeMinute"]).cumcount(ascending=False)
all_df["user_min_cumcount_ratio"] = all_df["user_min_cumcount"] / all_df["user_min_impcnt"]
all_df["user_min_inv_cumcount_ratio"] = all_df["user_min_inv_cumcount"] / all_df["user_min_impcnt"]

all_df["user_app_4g_cumcount"] = all_df.groupby(["user_id", "app_code", "is_4G"]).cumcount()
all_df["user_app_4g_inv_cumcount"] = all_df.groupby(["user_id", "app_code", "is_4G"]).cumcount(ascending=False)
all_df["user_app_4g_cumcount_ratio"] = all_df["user_app_4g_cumcount"] / all_df["user_app_4g_impression_count"]
all_df["user_app_4g_inv_cumcount_ratio"] = all_df["user_app_4g_inv_cumcount"] / all_df["user_app_4g_impression_count"]
#

#number of impressions for each user
gdf = all_df.groupby(["user_id"])["app_code"].nunique().reset_index()
gdf.columns = ["user_id", "user_app_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

#number of impressions for each user and app used
gdf = all_df.groupby(["user_id", "app_code"])["impression_time_ord"].nunique().reset_index()
gdf.columns = ["user_id","app_code", "user_app_time_unicount"]
all_df = all_df.merge(gdf, on=["user_id","app_code"], how="left")

#number of impressions for each app
gdf = all_df.groupby(["user_id"])["impression_timeWeek"].nunique().reset_index()
gdf.columns = ["user_id", "user_wk_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeDay"].nunique().reset_index()
gdf.columns = ["user_id", "user_day_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeDayofweek"].nunique().reset_index()
gdf.columns = ["user_id", "user_daywk_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeIs_month_end"].nunique().reset_index()
gdf.columns = ["user_id", "user_moe_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeIs_month_start"].nunique().reset_index()
gdf.columns = ["user_id", "user_mos_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeHour"].nunique().reset_index()
gdf.columns = ["user_id", "user_hr_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["user_id"])["impression_timeMinute"].nunique().reset_index()
gdf.columns = ["user_id", "user_min_unicount"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")
##
##all min/max
gdf = all_df.groupby(["user_id"])["impression_time_ord"].agg(["min", "max"]).reset_index()
gdf.columns = ["user_id", "user_time_min", "user_time_max"]
all_df = all_df.merge(gdf, on=["user_id"], how="left")

gdf = all_df.groupby(["app_code"])["impression_time_ord"].agg(["min", "max"]).reset_index()
gdf.columns = ["app_code", "app_code_time_min", "app_code_time_max"]
all_df = all_df.merge(gdf, on=["app_code"], how="left")

gdf = all_df.groupby(["user_id", "app_code"])["impression_time_ord"].agg(["min", "max"]).reset_index()
gdf.columns = ["user_id", "app_code", "user_app_code_time_min", "user_app_code_time_max"]
all_df = all_df.merge(gdf, on=["user_id", "app_code"], how="left")

all_df["user_app_over_min_timediff"] = all_df['impression_time_ord'] - all_df['user_app_code_time_min']
all_df["user_app_over_max_timediff"] = all_df['user_app_code_time_max'] - all_df['impression_time_ord']

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
##log price shift1
view_item_log["user_prev_price"] = view_item_log.groupby(["user_id"])["item_price"].shift(1)
view_item_log["user_next_price"] = view_item_log.groupby(["user_id"])["item_price"].shift(-1)

view_item_log["user_price_diffprev"] = view_item_log["item_price"] - view_item_log["user_prev_price"]
view_item_log["user_price_diffnext"] = view_item_log["user_next_price"] - view_item_log["item_price"]
view_item_log["user_price_diff1"] = view_item_log["user_price_diffprev"] - view_item_log["user_price_diffnext"]

gdf = view_item_log.groupby(["user_id"])["user_price_diff1"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_price_diff1_log_min", "user_price_diff1_log_max", "user_price_diff1_log_mean", "user_price_diff1_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")
##log price shift2
view_item_log["user_prev_price2"] = view_item_log.groupby(["user_id"])["item_price"].shift(2)
view_item_log["user_next_price2"] = view_item_log.groupby(["user_id"])["item_price"].shift(-2)

view_item_log["user_price_diffprev2"] = view_item_log["item_price"] - view_item_log["user_prev_price2"]
view_item_log["user_price_diffnext2"] = view_item_log["user_next_price2"] - view_item_log["item_price"]
view_item_log["user_price_diff2"] = view_item_log["user_price_diffprev2"] - view_item_log["user_price_diffnext2"]

gdf = view_item_log.groupby(["user_id"])["user_price_diff2"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_price_diff2_log_min", "user_price_diff2_log_max", "user_price_diff2_log_mean", "user_price_diff2_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log price shift3
view_item_log["user_prev_price3"] = view_item_log.groupby(["user_id"])["item_price"].shift(3)
view_item_log["user_next_price3"] = view_item_log.groupby(["user_id"])["item_price"].shift(-3)

view_item_log["user_price_diffprev3"] = view_item_log["item_price"] - view_item_log["user_prev_price3"]
view_item_log["user_price_diffnext3"] = view_item_log["user_next_price3"] - view_item_log["item_price"]
view_item_log["user_price_diff3"] = view_item_log["user_price_diffprev3"] - view_item_log["user_price_diffnext3"]

gdf = view_item_log.groupby(["user_id"])["user_price_diff3"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_price_diff3_log_min", "user_price_diff3_log_max", "user_price_diff3_log_mean", "user_price_diff3_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")
#log price shift4
view_item_log["user_prev_price4"] = view_item_log.groupby(["user_id"])["item_price"].shift(4)
view_item_log["user_next_price4"] = view_item_log.groupby(["user_id"])["item_price"].shift(-4)

view_item_log["user_price_diffprev4"] = view_item_log["item_price"] - view_item_log["user_prev_price4"]
view_item_log["user_price_diffnext4"] = view_item_log["user_next_price4"] - view_item_log["item_price"]
view_item_log["user_price_diff4"] = view_item_log["user_price_diffprev4"] - view_item_log["user_price_diffnext4"]

gdf = view_item_log.groupby(["user_id"])["user_price_diff4"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["user_id", "user_price_diff4_log_min", "user_price_diff4_log_max", "user_price_diff4_log_mean", "user_price_diff4_log_std"]
view_item_log = view_item_log.merge(gdf, on=["user_id"], how="left")
all_df = all_df.merge(gdf, on=["user_id"], how="left")
##mostcommon
view_item_log_mostcommon =view_item_log[['user_id', 'item_id', 'item_price', 'category_1', 'category_2', 'category_3', 'product_type']]
log_mode=view_item_log_mostcommon.groupby(['user_id']).apply(pd.DataFrame.mode).reset_index(drop=True)
log_mode.head()
log_mode = log_mode[np.isfinite(log_mode['user_id'])]
all_df = all_df.merge(log_mode, on=["user_id"], how="left")

view_item_log_mostcommon =view_item_log[['user_id', 'server_timeDay', 'server_timeDayofweek', 'server_timeHour']]
log_mode=view_item_log_mostcommon.groupby(['user_id']).apply(pd.DataFrame.mode).reset_index(drop=True)
log_mode.head()
log_mode = log_mode[np.isfinite(log_mode['user_id'])]
all_df = all_df.merge(log_mode, on=["user_id"], how="left")

dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
dtypes_log = view_item_log.dtypes.to_frame('dtypes').reset_index()

del gdf, log_mode,view_item_log_mostcommon, view_item_log, sam_all_df, sam_train, sam_test, sam_view_item_log
#Split Train and Test
train_df = all_df[all_df["train_set"]==1].reset_index(drop=True)
test_df = all_df[all_df["train_set"]==0].reset_index(drop=True)

##
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()
sam_all_df = all_df.head(1000)

encode_cols = [["app_code"], ["user_id", "app_code"],["impression_timeHour"], ["impression_timeDayofweek"], ["os_version"],
               ["app_code", "impression_timeHour"], ["os_version", "app_code"], ["user_date_item_uniq"], ["user_hr_item_uniq"], ["app_impression_count"],
               ["user_hr_sess_uniq"], ["item_price"], ["item_id"], ["user_id", "item_price"], ["user_id", "item_id"], ["category_3","server_timeDay"], ["product_type"],
               ["user_id", "item_id", "product_type"], ["item_uniq"], ["category_3"], ["user_daywk_item_uniq"], ["category_2"],
               ["user_week_impcnt"], ["user_app_hr_impcnt"], ["user_app_min_impcnt"],
               ["user_day_unicount"], ["user_daywk_unicount"], ["user_moe_unicount"], ["user_mos_unicount"], ["user_hr_unicount"], ["user_min_unicount"]]

date_dev = [datetime.date(2018, 11, 15), datetime.date(2018, 11, 16), datetime.date(2018, 11, 17), datetime.date(2018, 11, 18), datetime.date(2018, 11, 19), datetime.date(2018, 11, 20), datetime.date(2018, 11, 21), datetime.date(2018, 11, 22), datetime.date(2018, 11, 23), datetime.date(2018, 11, 24), datetime.date(2018, 11, 25), datetime.date(2018, 11, 26), datetime.date(2018, 11, 27), datetime.date(2018, 11, 28), datetime.date(2018, 11, 29), datetime.date(2018, 11, 30), datetime.date(2018, 12, 1), datetime.date(2018, 12, 2), datetime.date(2018, 12, 3), datetime.date(2018, 12, 4), datetime.date(2018, 12, 5), datetime.date(2018, 12, 6), datetime.date(2018, 12, 7), datetime.date(2018, 12, 8), datetime.date(2018, 12, 9), datetime.date(2018, 12, 10), datetime.date(2018, 12, 11)]

#Encoding codes taken from SudalaiRajkumar
def getDVEncodeVar(compute_df, target_df, var_name, target_var="is_click", min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["mean"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])
##

print ("Target encoding..")
for col in encode_cols:
#for col in [["user_id"]]:
	train_enc_values = np.zeros(train_df.shape[0])
	test_enc_values = 0
	for date in date_dev:
	#for [dev_camp, val_camp] in camp_indices:
		dev_X, val_X = train_df[train_df["impression_date"] != date], train_df[train_df["impression_date"] == date]
		train_enc_values[train_df["impression_date"] == date] = np.array( getDVEncodeVar(val_X[col], dev_X, col, 'is_click'))
		test_enc_values += np.array( getDVEncodeVar(test_df[col], dev_X, col, 'is_click'))
	test_enc_values /= 27.
	if isinstance(col, list):
		col = "_".join(col)
	train_df[col + "_enc"] = train_enc_values
	test_df[col + "_enc"] = test_enc_values
###
def getCountVar(compute_df, count_df, var_name, count_var="is_click"):
	grouped_df = count_df.groupby(var_name)[count_var].agg('count').reset_index()
	grouped_df.columns = var_name + ["var_count"]

	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(grouped_df["var_count"].values), inplace=True)
	return list(merged_df["var_count"])
##
print ("count encoding..")
for col in encode_cols:
#for col in [["user_id"]]:
	train_enc_values = np.zeros(train_df.shape[0])
	test_enc_values = 0
	for date in date_dev:
	#for [dev_camp, val_camp] in camp_indices:
		dev_X, val_X = train_df[train_df["impression_date"] != date], train_df[train_df["impression_date"] == date]
		train_enc_values[train_df["impression_date"] == date] = np.array( getCountVar(val_X[col], dev_X, col, 'is_click'))
		test_enc_values = test_enc_values + np.array( getCountVar(test_df[col], dev_X, col, 'is_click'))
	test_enc_values =test_enc_values/ 27.
	if isinstance(col, list):
		col = "_".join(col)
	train_df[col + "_countenc"] = train_enc_values
	test_df[col + "_countenc"] = test_enc_values
##
def getSumVar(compute_df, count_df, var_name, count_var="is_click"):
	grouped_df = count_df.groupby(var_name)[count_var].agg('sum').reset_index()
	grouped_df.columns = var_name + ["var_count"]

	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(grouped_df["var_count"].values), inplace=True)
	return list(merged_df["var_count"])
##
print ("sum encoding..")
for col in encode_cols:
#for col in [["user_id"]]:
	train_enc_values = np.zeros(train_df.shape[0])
	test_enc_values = 0
	for date in date_dev:
	#for [dev_camp, val_camp] in camp_indices:
		dev_X, val_X = train_df[train_df["impression_date"] != date], train_df[train_df["impression_date"] == date]
		train_enc_values[train_df["impression_date"] == date] = np.array( getSumVar(val_X[col], dev_X, col, 'is_click'))
		test_enc_values = test_enc_values + np.array( getSumVar(test_df[col], dev_X, col, 'is_click'))
	test_enc_values =test_enc_values/ 27.
	if isinstance(col, list):
		col = "_".join(col)
	train_df[col + "_sumenc"] = train_enc_values
	test_df[col + "_sumenc"] = test_enc_values

#Preparing data for model building
cols_to_leave = ["impression_date", "impression_id", "impression_time", "impression_time_ord", "train_set", "impression_timeDay", "total_time_secs", "user_prev_time", "user_next_time", "server_time_date_x", "server_timeDayofweek_x", "user_prev_time2", "user_next_time2", "server_time_date_y", "server_timeDayofweek_y", "user_prev_time3", "user_next_time3", "server_time_date_x", "server_timeDayofweek_x", "user_prev_time4", "user_next_time4", "server_time_date_y", "server_timeDayofweek_y", "user_prev_time5", "user_next_time5", "server_time_date_x", "server_timeDayofweek_x", "server_time_date_y", "user_date_sess_uniq", "server_timeDayofweek_y", "server_timeHour_x", "server_time_date", "server_timeDayofweek_x", "server_timeHour_y", "user_impression_count", "user_app_4g_over_cnt_r", "user_app_time_impcnt", "user_app_timecnt_prev", "user_app_timecnt_nxt", "user_app_timecnt_diffprev", "user_app_timecnt_diffnext", "user_app_timecnt_prev2", "user_app_timecnt_nxt2", "user_app_timecnt_diffprev2", "user_app_timecnt_diffnext2", "user_time_min", "user_time_max", "app_code_time_min", "app_code_time_max", "user_app_code_time_min", "user_app_code_time_max", "server_timeDayofweek_y"]

test_df = test_df.drop(['is_click'],axis=1)
fe_col = [col for col in train_df.columns if col not in ID_y_col]
cols_to_use = []
cols_to_use = [i for i in fe_col if i not in cols_to_leave]
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = (train_df["is_click"]).values
##
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed=1201, dep=7, data_leaf = 250, rounds=50000):
	params = {}
	params["objective"] = "binary"
	params['metric'] = 'auc'
	params["max_depth"] = dep
	params["min_data_in_leaf"] =data_leaf
	params["learning_rate"] = 0.02
#	params["min_gain_to_split"] = 0.01
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.6
	params["feature_fraction_seed"] = seed
	params["bagging_freq"] = 1
	params["bagging_seed"] = seed
	params["scale_pos_weight"] = 4
#	params["is_unbalance"] = True
#	params["min_sum_hessian_in_leaf"] = 100
	params["lambda_l1"] = 0.5
	params["lambda_l2"] = 5
	params["num_leaves"] = 20
#	params["max_bin"] = 100
	params["verbosity"] = 0
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtrain,lgtest], early_stopping_rounds=200, verbose_eval=500)
	else:
		lgtest = lgb.Dataset(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	if test_X2 is not None:
		pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
	print("Features importance...")
	gain = model.feature_importance('gain')
	ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
	ft.to_csv("wns_av_fimp_17.0.csv", index=False)
	print(ft.head(25))

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		print (loss)
		return model, loss, pred_test_y, pred_test_y2
	else:
		return model, loss, pred_test_y, pred_test_y2
##
date_dev = [datetime.date(2018, 11, 15), datetime.date(2018, 11, 16), datetime.date(2018, 11, 17), datetime.date(2018, 11, 18), datetime.date(2018, 11, 19), datetime.date(2018, 11, 20), datetime.date(2018, 11, 21), datetime.date(2018, 11, 22), datetime.date(2018, 11, 23), datetime.date(2018, 11, 24), datetime.date(2018, 11, 25), datetime.date(2018, 11, 26), datetime.date(2018, 11, 27), datetime.date(2018, 11, 28), datetime.date(2018, 11, 29), datetime.date(2018, 11, 30), datetime.date(2018, 12, 1), datetime.date(2018, 12, 2), datetime.date(2018, 12, 3), datetime.date(2018, 12, 4), datetime.date(2018, 12, 5), datetime.date(2018, 12, 6), datetime.date(2018, 12, 7), datetime.date(2018, 12, 8), datetime.date(2018, 12, 9), datetime.date(2018, 12, 10), datetime.date(2018, 12, 11)]

print ("Model building..")
model_name = "LGB"
cv_scores = []
pred_test_full = 0
for date in date_dev:
    dev_X = train_X[train_df["impression_date"] != date]
    val_X = train_X[train_df["impression_date"] == date]
    dev_y = train_y[train_df["impression_date"] != date]
    val_y = train_y[train_df["impression_date"] == date]
    print (dev_X.shape, val_X.shape, dev_y.shape, val_y.shape)

    pred_val = 0
    model, loss, pred_val, pred_test = runLGB(dev_X, dev_y, val_X, val_y, test_X, seed=12345786)
    pred_val += pred_val
    pred_test_full += pred_test
    loss = metrics.roc_auc_score(val_y, pred_val)
    cv_scores.append(loss)
    print(cv_scores)
print(np.mean(cv_scores))
pred_test_full /= 27.
#Submissions
out_df = pd.DataFrame({"impression_id":test_id})
out_df["is_click"] = pred_test_full
out_df.to_csv("wns_2019_av_test_17.0.csv", index=False)

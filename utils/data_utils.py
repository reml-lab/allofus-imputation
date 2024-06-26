import os
import subprocess
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq

from google.cloud import bigquery

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

INFO_FILE  = "all_data_info.parquet" 
DATA_FILE  = "all_data.parquet" 
INDEX_FILE = "all_data_index.npy"
START_END_FILE = "valid_start_end_day.parquet"
SHIFT_FILE = "dayweek_shift.parquet"
# ROOT_PATH = "/home/jupyter/workspaces/mdottrd1"
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_CACHE = f"{ROOT_PATH}/file_cache"


def extract_data_from_db():

    # Connect to DB
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(dry_run=False, use_query_cache=False, allow_large_results=True)

    # Define query
    sql = """
        SELECT
            steps_intraday.person_id,
            DATETIME_TRUNC(steps_intraday.datetime, HOUR) datetime,
            SUM(steps_intraday.steps) steps,
            AVG(heart_rate_minute_level.heart_rate_value) heart_rate,
            COUNT(steps_intraday.person_id) count

        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.steps_intraday` steps_intraday
        INNER JOIN `""" + os.environ["WORKSPACE_CDR"] + """.heart_rate_minute_level` heart_rate_minute_level
            ON steps_intraday.datetime = heart_rate_minute_level.datetime 
            AND steps_intraday.PERSON_ID = heart_rate_minute_level.PERSON_ID
        GROUP BY steps_intraday.person_id, datetime
        ORDER BY steps_intraday.person_id, datetime """

    # Run query
    query_job = client.query((sql),job_config=job_config) 
    result = query_job.result()

    # Save the data to parquet page files
    job_config = bigquery.ExtractJobConfig(destination_format="PARQUET")
    extract_job = client.extract_table(
        result._table,
        os.getenv('WORKSPACE_BUCKET')+"/data/raw_data_*.parquet",
        location="US",
        job_config=job_config 
    )
    out=extract_job.result()


def process_this_df(this_df, last_id):
    # Re-index datetime to include all hours
    this_df = this_df.set_index("datetime")
    start = pd.to_datetime(this_df.index[0].date())
    end = pd.to_datetime(this_df.index[-1].date()) + pd.to_timedelta(23, unit='h')
    index = pd.date_range(start=start, end=end, freq='H',name="datetime")
    this_df = this_df.reindex(index, fill_value=0)
    this_df["Participant ID"] = last_id

    # Set hierarchical index
    this_df = this_df.reset_index()
    this_df = this_df.set_index(["Participant ID","datetime"])
    this_df["steps"] = this_df["steps"].astype(np.uint16)
    this_df["valid_minutes"] = this_df["valid_minutes"].astype(np.uint8)
    this_df["heart_rate"] = this_df["heart_rate"].astype(np.float32)

    # Add datetime indicators
    this_df = add_date_indicators(this_df)
    
    return this_df


def combine_db_extracts():
    num_files = 44

    # Perform first pass to get correct fragment order to reconstruct data
    # although the personal ids are ordered, the filename is not ordered
    # e.g. the person with smallest id could be the first id in the second file
    first_id = {}
    last_id = {}
    
    for j in tqdm(range(num_files)):
        in_file_name = "raw_data_%0.12d.parquet"%(j)
        pull_file(in_file_name)
        df = pd.read_parquet(os.path.join(FILE_CACHE,in_file_name))
        first_id[j] = df["person_id"][0]
        last_id[j] = df["person_id"][len(df)-1]

    vals = np.array(list(first_id.values()))
    order=np.argsort(vals)

    # Clear intermediate structures
    # the reason to use last_df is that the same participant's data could cross two files
    # for example, the participant could have some rows at the end of 1st file
    # and the remaining rows are at the beginning of the 2nd file.
    last_df = None
    last_id = -1
    writer = None
    id_index = {}
    k = 0

    # Loop through files in correct order
    for j in order:
        in_file_name = "raw_data_%0.12d.parquet"%(j)
        print("Reading file %s" % in_file_name)
        df, _ = load_df_from_bucket(in_file_name)
        ids = df["person_id"].unique()
        
        # Add data to parquet file for each id
        for pid in ids:
            if pid in id_index:
                print(f"***Got another data chunk from id {pid}. Skipping.")
                continue
            print("  Getting data for id %s" % pid)
            new_df = df.loc[df["person_id"]==pid]        
            new_df = new_df.rename({"person_id":"Participant ID", "count":"valid_minutes"}, axis=1)

            if last_df is None:
                last_df = new_df
                last_id = pid
            else:
                if last_id != pid:
                    # Note we are dealing with the last_df instead of the new_df below!
                    this_df = last_df.copy(deep=True)
                    # we assign the new_df to last_df here instead of the end is that
                    # this new_df will have the same format as the new_df in the next round
                    # but since last_id and k will be used below, they are updated at the end
                    last_df = new_df
                    this_df = process_this_df(this_df, last_id)
                    print("  Writing %d rows for id %s"%(len(this_df), last_id))
                    table = pa.Table.from_pandas(this_df, preserve_index=True)
                    if(writer is None):
                        writer = pq.ParquetWriter('all_data.parquet', table.schema)
                    writer.write_table(table)
                    id_index[last_id] = k
                    # update last_id and k
                    last_id = pid 
                    k = k + 1 
                else:    
                    print("  Concatinating data for id %s" % pid)
                    last_df = pd.concat([last_df, new_df], axis=0) 

        if j==order[-1] and pid==ids[-1]:
            # we need to write the last one into file
            # since previous steps are for the last_df
            # we need to write new_df (which is last_df right now) into file
            this_df = last_df
            this_df = process_this_df(this_df, last_id)
            print("  Writing %d rows for id %s"%(len(this_df), last_id))
            table = pa.Table.from_pandas(this_df, preserve_index=True)
            if(writer is None):
                writer = pq.ParquetWriter('all_data.parquet', table.schema)
            writer.write_table(table)
            id_index[last_id] = k
    
    writer.close()
    np.save("all_data_index.npy",id_index)

    # Copy data to cloud bucket
    os.system(f"gsutil -m cp all_data.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    os.system(f"gsutil -m cp all_data_index.npy {os.getenv('WORKSPACE_BUCKET')+'/data/'}")


def rollup_minutes(df, interval="H"):
    '''
    rollup_minutes(df, interval)
    inputs:
        df: Pandas dataframe containing minute level fitbit data as output by get_minute_level_data
        interval: interval to roll up to. Default is the hourly level (H)
    outputs:
        df: Pandas dataframe containing hourly summary of minute level fitbit data
    '''

    df['valid_minutes'] = np.logical_and(df["steps"].notna(), df["heart_rate"].notna())
    df.loc[df["valid_minutes"]==False, 'steps'] = np.nan
    df.loc[df["valid_minutes"]==False, 'heart_rate'] = np.nan

    df=df.reset_index()
    
    df1 = df.set_index('datetime').resample(interval).first()
    df2 = df.set_index('datetime').resample(interval).sum().astype(float)
    df3 = df.set_index('datetime').resample(interval).mean().astype(float)
    
    df1["steps"] = df2["steps"]
    df1["valid_minutes"] = df2["valid_minutes"]
    df1["heart_rate"] = df3["heart_rate"]

    df1=df1.reset_index()
    df1 = df1.set_index(["Participant ID","datetime"])
    return df1


def add_date_indicators(df):
    '''
    add_date_indicators(df)
    inputs:
        df: Pandas dataframe containing a datetime column in the second index level
    outputs:
        df: Pandas dataframe with additional date indicators added
    '''

    # Add collection of date related indicators
    dates = pd.to_datetime(df.index.get_level_values(1))
    
    # Day of week
    df["Hour of Day"] = dates.hour

    # Day of week
    df["Day of Week"] = dates.dayofweek

    #Is weekend day
    df["Is Weekend Day"] = dates.dayofweek >=5 

    # Day of year
    df["Day of Year"] = dates.dayofyear

    # Get days in study variable for each participant
    date_diff  = dates-dates[0]
    study_days = list(date_diff.days)        
    df["Study day"] = study_days

    return df


def save_df_to_bucket(df, file, overwrite=False):

    if(exists_in_bucket(file) and not overwrite):
        print("File %s already in workspace bucket. Set overwrite=True to replace."%file)
        return False
    
    # save dataframe in a csv file in the same workspace as the notebook
    df.to_parquet(file)

    # get the bucket name
    my_bucket = os.getenv('WORKSPACE_BUCKET')

    # copy csv file to the bucket
    args = ["gsutil", "-m", "cp", f"./{file}", f"{my_bucket}/data/"]
    output = subprocess.run(args, capture_output=True)

    return output


def load_df_from_bucket(file,cache=True):
    
    if(not(exists_in_bucket(file))):
        print("File %s does not exist in workspace bucket. Can not load."%file)
        return None

    subprocess.run(["rm","-rf","./temp"])
    
    if(not os.path.exists(FILE_CACHE)):
        subprocess.run(["mkdir",FILE_CACHE])

    err=False
    if(not os.path.exists(os.path.join(FILE_CACHE,file))):
        # get the bucket name
        my_bucket = os.getenv('WORKSPACE_BUCKET')

        # copy csv file from the bucket
        args = ["gsutil", "cp", f"{my_bucket}/data/{file}", f"{FILE_CACHE}/{file}"]
        output = subprocess.run(args, capture_output=True)
        err = (output.returncode!=0)
    else:
        output="Used cache"

    if(not err):    
        df=pd.read_parquet(f"{FILE_CACHE}/{file}")
    else:
        df= None
        
    return df, output


def exists_in_bucket(file):
    my_bucket = os.getenv('WORKSPACE_BUCKET')
    loc = my_bucket + "/data/" + file
    args = ["gsutil","-q","stat",loc]
    output = subprocess.run(args, capture_output=True)
    return(output.returncode==0)


def pull_file(file, from_scratch=False):
     
    if(not os.path.exists(FILE_CACHE)):
        subprocess.run(["mkdir",FILE_CACHE])

    if(not os.path.exists(os.path.join(FILE_CACHE,file))):

        if(exists_in_bucket(file)):
            my_bucket = os.getenv('WORKSPACE_BUCKET')
            # copy parquet file from the bucket
            print(f"File {file} not found in local cache. Downloading...")
            args = ["gsutil",  "-m", "cp", f"{my_bucket}/data/{file}", f"{FILE_CACHE}/{file}"]
            os.system(" ".join(args))
        else:
            print(f"File {file} not found in gcloud bucket.")

    return(os.path.exists(os.path.join(FILE_CACHE,file)))


def push_file(file):
     
    if(os.path.exists(os.path.join(FILE_CACHE,file))):

        # copy parquet file from the bucket
        print(f"File {file} not found in cache. Downloading...")
        my_bucket = os.getenv('WORKSPACE_BUCKET')
        args = ["gsutil",  "-m", "cp",  f"./file_cache/{file}", f"{my_bucket}/data/{file}"]
        os.system(" ".join(args))
    else:
        print(f"File {file} not found in cache")


def make_data_info():

    pull_file(DATA_FILE)
    parquet_file = pq.ParquetFile(f"./file_cache/{DATA_FILE}")

    ids= []
    num_hours=[]
    num_valid_hours=[]

    print("Extracting info from data file...")
    for i in tqdm(range(parquet_file.num_row_groups)):
        df = parquet_file.read_row_group(i,use_pandas_metadata=True).to_pandas()
        ids.append(df.index.levels[0][0])
        num_hours.append(len(df))
        num_valid_hours.append(np.sum(df["valid_minutes"]>0))

    df = pd.DataFrame({"Participant ID":ids,"Num Hours":num_hours,"Num Valid Hours":num_valid_hours})
    df["Miss Rate"] = 1-df["Num Valid Hours"]/(1e-16+df["Num Hours"])
    df=df.set_index("Participant ID")

    df.to_parquet(os.path.join(FILE_CACHE,INFO_FILE))

    return df


def get_data_info():

    if(not pull_file(INFO_FILE)):
        make_data_info()

    df = pd.read_parquet(os.path.join(FILE_CACHE,INFO_FILE))
    return(df)


def get_participant_ids(from_scratch=False):

    pull_file(INDEX_FILE)   
    index = np.load(f"{FILE_CACHE}/{INDEX_FILE}",allow_pickle=True).item()
    return(list(index.keys()))

    
def get_data_by_id(id):

    pull_file(INDEX_FILE)
    pull_file(DATA_FILE)

    index = np.load(f"{FILE_CACHE}/{INDEX_FILE}",allow_pickle=True).item()
    i = index[id]

    parquet_file = pq.ParquetFile(os.path.join(FILE_CACHE,DATA_FILE))
    df = parquet_file.read_row_group(i,use_pandas_metadata=True).to_pandas()

    assert id == df.index.levels[0][0], "participant id is not what you want!"

    return(df)


def sanity_check():
    """
    Do the sanity check of the raw data
    """
    pull_file(DATA_FILE)
    parquet_file = pq.ParquetFile(f"./file_cache/{DATA_FILE}")
    
    for i in tqdm(range(parquet_file.num_row_groups)):
        # read the dataframe
        # note it has a different order from ids
        df_part = parquet_file.read_row_group(i,use_pandas_metadata=True).to_pandas()

        # get the participant id
        pid = df_part.index.levels[0][0]
        
        # reset the index so that the participant ID and datatime will be as the columns
        df_part.reset_index(inplace=True)

        # sanity check 1: if all the field is available all the rows
        # Note that the missing values have already been filled with zeros
        # so there should not be any NaNs in the dataframe
        if df_part.isna().any().any():
            print(f"{pid} has NaNs in the dataframe")

        # sanity check 2: if the study day is correct or not
        if (df_part.iloc[-1]["Study day"] + 1) * 24 != len(df_part):
            print(f"{pid} has wrong study day")

        # sanity check 3: if each day starts from 00:00:00 and ends at 23:00:00
        if (df_part.groupby("Study day").head(1)["Hour of Day"].nunique()!=1) or (df_part.groupby("Study day").head(1)["Hour of Day"].unique().item()!=0): 
            print(f"{pid} has some study day does not begin at 00:00:00")
        if (df_part.groupby("Study day").tail(1)["Hour of Day"].nunique()!=1) or (df_part.groupby("Study day").tail(1)["Hour of Day"].unique().item()!=23): 
            print(f"{pid} has some study day does not end at 23:00:00")

        # sanity check 4: if steps, heart_rate, valid_minutes are correlated with each other
        # note that if the number of the total hours is very small, then all the hours are not missing (valid_minutes>0)
        if (len(df_part.loc[df_part["valid_minutes"]==0]) == 0) or (len(df_part.loc[df_part["heart_rate"]==0]) == 0):
            print(f"{pid} has no missing hours")
        else:
            if (df_part.loc[df_part["valid_minutes"]==0, "steps"].nunique()!=1) or (df_part.loc[df_part["valid_minutes"]==0, "steps"].unique().item()!=0): 
                print(f"{pid} has non-zero steps when valid minutes is zero")
            if (df_part.loc[df_part["valid_minutes"]==0, "heart_rate"].nunique()!=1) or (df_part.loc[df_part["valid_minutes"]==0, "heart_rate"].unique().item()!=0):
                print(f"{pid} has non-zero heart rate when valid minutes is zero")
            if (df_part.loc[df_part["heart_rate"]==0, "valid_minutes"].nunique()!=1) or (df_part.loc[df_part["heart_rate"]==0, "valid_minutes"].unique().item()!=0): 
                print(f"{pid} has non-zero valid minutes when heart rate is zero")
            if (df_part.loc[df_part["heart_rate"]==0, "steps"].nunique()!=1) or (df_part.loc[df_part["heart_rate"]==0, "steps"].unique().item()!=0):
                print(f"{pid} has non-zero steps when heart rate is zero")
        if 0 in df_part.loc[df_part["valid_minutes"]>0, "heart_rate"].unique():
            print(f"{pid} has zero heart rate when valid minutes is not zero")


def check_heart_step():
    """
    Check the max and min value of heart rate and step counts
    """
    pull_file(DATA_FILE)
    parquet_file = pq.ParquetFile(f"./file_cache/{DATA_FILE}")
    
    ids = []
    min_heart_rate = []
    max_heart_rate = []

    min_step_count = []
    max_step_count = []
   
    for i in tqdm(range(parquet_file.num_row_groups)):
        # read the dataframe
        # note it has a different order from ids
        df = parquet_file.read_row_group(i,use_pandas_metadata=True).to_pandas()
        
        # get the id
        ids.append(df.index.levels[0][0])
        
        # get the valid heart rate
        valid_heart_rate = df.loc[df["valid_minutes"]>0, "heart_rate"]
        min_heart_rate.append(valid_heart_rate.min())
        max_heart_rate.append(valid_heart_rate.max())
        
        # get the valid step count
        valid_step_count = df.loc[df["valid_minutes"]>0, "steps"]
        min_step_count.append(valid_step_count.min())
        max_step_count.append(valid_step_count.max())
    
    df_stats = pd.DataFrame({"Participant ID": ids,
                    "Min Heart Rate": min_heart_rate,
                    "Max Heart Rate": max_heart_rate,
                    "Min Step Count": min_step_count,
                    "Max Step Count": max_step_count})

    df_stats=df_stats.set_index("Participant ID")  
    # we need to convert them from int16 to uint16
    # otherwise, there are overflow negative in the max step count column
    df_stats["Min Step Count"] = df_stats["Min Step Count"].astype(np.uint16)
    df_stats["Max Step Count"] = df_stats["Max Step Count"].astype(np.uint16)

    return df_stats


def normalize_data(dataframe, up_quantile=0.999):
    """
    Normalize the data using the mean and std which are computed using data below 99.9% quantile
    Args:
        - dataframe: the dataframe whose feature will be normalized
        - up_quantile: upper quantile, we only compute mean and std from values below upper quantile,
                       in order to minimize the effect of outliers.
    """
    # get the non missing values
    valid_step_rate_values = dataframe.loc[dataframe["step_rate"].notnull(), "step_rate"].values
    valid_heart_rate_values = dataframe.loc[dataframe["heart_rate"].notnull(), "heart_rate"].values
    
    # get the upper threshold
    th_step_rate = np.quantile(valid_step_rate_values, up_quantile)
    th_heart_rate = np.quantile(valid_heart_rate_values, up_quantile)
    
    # get the values which are below th
    valid_step_rate_values = valid_step_rate_values[valid_step_rate_values<=th_step_rate]
    valid_heart_rate_values = valid_heart_rate_values[valid_heart_rate_values<=th_heart_rate]
    
    # compute mean and std or max and min
    mean_step_rate, std_step_rate = np.mean(valid_step_rate_values), np.std(valid_step_rate_values)
    mean_heart_rate, std_heart_rate = np.mean(valid_heart_rate_values), np.std(valid_heart_rate_values)
    
    # manipulate the values in the dataframe
    dataframe.loc[dataframe["step_rate"].notnull(), "step_rate_norm"] = (dataframe.loc[dataframe["step_rate"].notnull(), "step_rate"] - mean_step_rate) / std_step_rate
    dataframe.loc[dataframe["heart_rate"].notnull(), "heart_rate_norm"] = (dataframe.loc[dataframe["heart_rate"].notnull(), "heart_rate"] - mean_heart_rate) / std_heart_rate
    
    return dataframe, mean_step_rate, std_step_rate


def get_hourly_data(pid, num_split=10, ks=(9, 15), start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False):
    """
    Get the hourly level data after preprocessing, the output can be used to initialize the Dataset class.
    We can choose which period (start_hour to end_hour) we would like to include in the train/valid/test set.
    
    Args:
        - pid: Participant ID
        - num_split: how many train-valid-test splits we will use to estimate the model performance due to the imbalanced step counts.
        - ks: kernel size (kh, kw), height and width of the context window. 
        - start_hour: the start hour included in the train/valid/test set
        - end_hour: the end hour included in the train/valid/test set
        - conv_feat: bool, whether to create the features for the convolution-based model.
        - return_time_dict: bool, whether to return the dictionary which contains train/valid/test split information based on step count bins
    """

    pull_file(START_END_FILE) # get the start day and end day file
    pull_file(SHIFT_FILE) # get the shift of day of the week file

    # get the dataframe for the particular pid
    df_exp = get_data_by_id(pid)
    assert df_exp.index.levels[0][0] == pid, f"wrong participant data {df_exp.index.levels[0][0]} is read in!"

    ### preprocess ###
    # put participant id and datetime into the columns and reset the index
    df_exp.reset_index(inplace=True)

    # get the start and end day from the precomputed file
    df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/valid_start_end_day.parquet")
    start_day, end_day = df_start_end_day.loc[pid, ["Start Day", "End Day"]]
    df_exp = df_exp.loc[(df_exp["Study day"]>=start_day) & (df_exp["Study day"]<=end_day)]

    
    # reset the index since we remove the first and last several days which have no valid minutes
    df_exp.reset_index(drop=True, inplace=True)

    # get the date and time columns
    df_exp["date"] = df_exp["datetime"].dt.date
    df_exp["time"] = df_exp["datetime"].dt.time
    
    # set steps to be np.nan when valid_minutes equal to zero
    df_exp.loc[df_exp["valid_minutes"]==0, "steps"] = np.nan
    # set heart rate to be np.nan when valid_minutes equal to zero
    df_exp.loc[df_exp["valid_minutes"]==0, "heart_rate"] = np.nan

    # add the step rate column
    df_exp["step_rate"] = df_exp["steps"] / df_exp["valid_minutes"]

    # reset the study date to make the first valid study day as zero
    first_valid_study_day = df_exp.iloc[0]["Study day"]
    df_exp["Study day"] = df_exp["Study day"] - first_valid_study_day
    
    # normalize the data and get the mean and std of the step rates
    df_exp, step_rate_mean, step_rate_std = normalize_data(df_exp)
    
    # check if all the hours are available for the period
    timediff = (df_exp.iloc[-1].datetime - df_exp.iloc[0].datetime)
    assert timediff / datetime.timedelta(hours=1) == len(df_exp)-1, "some hours are missing"
    # second check: manually compute the hour
    def compute_time_axis(row):
        return row["Study day"]*24 + row["Hour of Day"]
    df_exp["time_axis"] = df_exp.apply(lambda x: compute_time_axis(x), axis=1)
    # make the first hour as the start hour which is zero
    df_exp["time_axis"] = df_exp["time_axis"] - df_exp.iloc[0]["time_axis"]
    assert (df_exp["time_axis"] == df_exp.index).all(), "some hours are missing"
    
    # shift the day of the week
    df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/dayweek_shift.parquet") 
    best_shift = df_best_shift_dw.loc[pid, "Best Shift"]
    df_exp["Day of Week"] = (df_exp["Day of Week"] + best_shift) % 7 
    df_exp["Is Weekend Day"] = (df_exp["Day of Week"].isin([5, 6])).astype("int")

    # add the step rate missingness indicator ###
    df_exp["step_mask"] = (df_exp["valid_minutes"]>0).astype("int")
    # fill the missing values in the step counts and step rates
    df_exp["steps"].fillna(value=0.0, inplace=True)
    df_exp["step_rate"].fillna(value=0.0, inplace=True)
    df_exp["step_rate_norm"].fillna(value=0.0, inplace=True)
    # fill the missing values in the heart rate
    df_exp["heart_rate"].fillna(value=0.0, inplace=True)
    df_exp["heart_rate_norm"].fillna(value=0.0, inplace=True)

    assert df_exp.notnull().all().all(), f"there is still NaN after filling heart rate and step rate of pid={pid}"

    ### Split the data into train, valid and test ###
    # since the data is very imbalanced w.r.t step counts, we do the stratified sampling multiple times
    # train:valid:test = 0.8:0.15:0.05
    
    # add the column of each split into the dataframe
    for split_idx in range(num_split):
        for setname in ["train", "valid", "test"]:
            col_name = f"{setname}_mask_split{split_idx}"
            df_exp[col_name] = 0
    
    # get the number of valid hour
    num_valid_hour = len(df_exp.loc[(df_exp["Hour of Day"]>=start_hour)&(df_exp["Hour of Day"]<=end_hour)&(df_exp["step_mask"]==1)])
    # get the bin length
    if num_valid_hour > 1200:
        bin_len = int(np.ceil(num_valid_hour / 1200) * 100)
    else:
        bin_len = int(np.ceil(num_valid_hour / 120) * 10)
    # get the steps and the index
    step_list = df_exp.loc[(df_exp["Hour of Day"]>=start_hour)&(df_exp["Hour of Day"]<=end_hour)&(df_exp["step_mask"]==1), "steps"].to_numpy()
    time_axis_list = df_exp.loc[(df_exp["Hour of Day"]>=start_hour)&(df_exp["Hour of Day"]<=end_hour)&(df_exp["step_mask"]==1)].index.to_numpy()
    # sort the steps
    index_sorted = np.argsort(step_list)
    time_axis_list = time_axis_list[index_sorted]
    # bin the steps
    num_bins = len(time_axis_list) // bin_len + 1
    last_bin_len = len(time_axis_list) % bin_len
    
    time_dict = {
             "train": {i:[] for i in range(num_split)},
             "valid": {i:[] for i in range(num_split)},
             "test": {i:[] for i in range(num_split)}
    } # record the train, valid and test axis for each split
    total_bins = 0 # record how many bins are there
    for bin_idx in range(int(num_bins)):
        time_axis_bin = time_axis_list[bin_idx*bin_len : (bin_idx+1)*bin_len]
        total_bins += 1
        if bin_idx == (num_bins - 2):
            # the second to the last
            if last_bin_len < 20:
                # we need to make sure that the test set has at least one instance (20 * 0.05 = 1)
                # if the last bin has less than 20, it is combined with the second to the last bin
                time_axis_bin = time_axis_list[bin_idx*bin_len:]
                for split_idx in range(num_split):
                    # split 
                    train_time_bin, test_valid_time_bin = train_test_split(time_axis_bin, train_size=0.8, 
                                                                    shuffle=True, random_state=42*split_idx)
                    valid_time_bin, test_time_bin = train_test_split(test_valid_time_bin, train_size=0.75, 
                                                                    shuffle=True, random_state=11*split_idx)
                    # set the mask
                    df_exp.loc[train_time_bin, f"train_mask_split{split_idx}"] = 1
                    df_exp.loc[valid_time_bin, f"valid_mask_split{split_idx}"] = 1
                    df_exp.loc[test_time_bin, f"test_mask_split{split_idx}"] = 1

                    time_dict["train"][split_idx].append(train_time_bin)
                    time_dict["valid"][split_idx].append(valid_time_bin)
                    time_dict["test"][split_idx].append(test_time_bin)
                break
        # split the data for each bin
        for split_idx in range(num_split):
            # split 
            train_time_bin, test_valid_time_bin = train_test_split(time_axis_bin, train_size=0.8, 
                                                                   shuffle=True, random_state=42*split_idx)
            valid_time_bin, test_time_bin = train_test_split(test_valid_time_bin, train_size=0.75, 
                                                             shuffle=True, random_state=11*split_idx)
            # set the mask
            df_exp.loc[train_time_bin, f"train_mask_split{split_idx}"] = 1
            df_exp.loc[valid_time_bin, f"valid_mask_split{split_idx}"] = 1
            df_exp.loc[test_time_bin, f"test_mask_split{split_idx}"] = 1

            time_dict["train"][split_idx].append(train_time_bin)
            time_dict["valid"][split_idx].append(valid_time_bin)
            time_dict["test"][split_idx].append(test_time_bin)

    # correctness check
    for i in range(num_split):
        assert df_exp.loc[df_exp["step_mask"]==0, f"train_mask_split{i}"].unique() == [0], f"{setname}_mask_split{i} of pid {pid} has invalid step_mask"
        assert df_exp.loc[df_exp["step_mask"]==0, f"valid_mask_split{i}"].unique() == [0], f"valid_mask_split{i} of pid {pid} has invalid step_mask"
        assert df_exp.loc[df_exp["step_mask"]==0, f"test_mask_split{i}"].unique() == [0], f"test_mask_split{i} of pid {pid} has invalid step_mask"

        assert df_exp.loc[df_exp[f"train_mask_split{i}"]==1, f"valid_mask_split{i}"].unique() == [0], f"valid_mask_split{i} of pid {pid} has train_mask"
        assert df_exp.loc[df_exp[f"train_mask_split{i}"]==1, f"test_mask_split{i}"].unique() == [0], f"test_mask_split{i} of pid {pid} has train_mask"
        assert df_exp.loc[df_exp[f"valid_mask_split{i}"]==1, f"test_mask_split{i}"].unique() == [0], f"test_mask_split{i} of pid {pid} has valid_mask"

    # correctness check 2
    for i in range(num_split):
        train_valid_test_len = len(df_exp.loc[df_exp[f"train_mask_split{i}"]==1]) + len(df_exp.loc[df_exp[f"valid_mask_split{i}"]==1]) + len(df_exp.loc[df_exp[f"test_mask_split{i}"]==1])
        miss_len = len(df_exp.loc[(df_exp["Hour of Day"]>=start_hour)&(df_exp["Hour of Day"]<=end_hour) &(df_exp["step_mask"]==0)])
        assert len(df_exp.loc[(df_exp["Hour of Day"]>=start_hour)&(df_exp["Hour of Day"]<=end_hour)]) == (train_valid_test_len  + miss_len), f"pid {pid} has some missing rows in split{i}"

    # print(f"pid = {pid} | num_valid_hour: {num_valid_hour} | bin_len: {bin_len} | num_bins: {total_bins}")      

    if not conv_feat:
        if return_time_dict:
            # in order to do the stratified sampling for the training data
            return df_exp, step_rate_mean, step_rate_std, time_dict
        else:
            return df_exp, step_rate_mean, step_rate_std

    ### Build features for convolution based models ###
    dataset = {i:{} for i in range(num_split)}
    for i in range(num_split):
        dataset[i]["train"] = np.concatenate(time_dict["train"][i])
        dataset[i]["valid"] = np.concatenate(time_dict["valid"][i])
        dataset[i]["test"] = np.concatenate(time_dict["test"][i])
        
    df_conv = df_exp
    # add train, valid and test mask during the computation (Note that all the above masks are for computing the loss and evaluation metrics but not for computing)
    # Train: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Valid: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Test: nothing is masked out (i.e. in the context window, there could be training and validation context points, 
    # and also test points which are the center of other test context windows, the center chunk needs to be masked out in the model)
    
    for i in range(num_split):
        # train
        df_conv[f"train_mask_comp_split{i}"] = 1
        df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"train_mask_comp_split{i}"] = 0
        # valid
        df_conv[f"valid_mask_comp_split{i}"] = 1
        df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"valid_mask_comp_split{i}"] = 0
        # test
        df_conv[f"test_mask_comp_split{i}"] = 1
        # set the mask corresponding to the original missing values as 0
        df_conv.loc[df_conv["step_mask"]==0, [f"train_mask_comp_split{i}", f"valid_mask_comp_split{i}", f"test_mask_comp_split{i}"]] = 0
    
    # correctness check
    for i in range(num_split):
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"train_mask_comp_split{i} of pid {pid} is wrong!" 
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"train_mask_comp_split{i} of pid {pid} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0])).all(), f"train_mask_comp_split{i} of pid {pid} is wrong!"

        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"valid_mask_comp_split{i} of pid {pid} is wrong!" 
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"valid_mask_comp_split{i} of pid {pid} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0])).all(), f"valid_mask_comp_split{i} of pid {pid} is wrong!"

        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} of pid {pid} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} of pid {pid} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} of pid {pid} is wrong!"
        
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"train_mask_comp_split{i} of pid {pid} has original missing values!"
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"valid_mask_comp_split{i} of pid {pid} has original missing values!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"test_mask_comp_split{i} of pid {pid} has original missing values!"

    # features used to compute the interpolation
    feature_list = []
    feature_list += ["step_rate_norm"]
    feature_list += [f"{setname}_mask_comp_split{i}" for i in range(num_split) for setname in ["train", "valid", "test"]]
    feature_list += ["Day of Week", "Hour of Day", "time_axis"]
    feature_list += ["heart_rate_norm"]
    # masks to compute the loss and evaluation metrics
    feature_list += [f"{setname}_mask_split{i}" for i in range(num_split) for setname in ["train", "valid", "test"]]
    # groundtruth
    feature_list += ["step_rate", "valid_minutes", "steps"]

    # get 2D features using pivot table
    df_conv = df_conv[feature_list + ["date", "time"]]
    feat2d_list = []
    for feat in feature_list:
        feat2d_list.append(pd.pivot_table(df_conv[["date", "time", feat]], index=["time"], columns=["date"], dropna=False).values[None, ...])
    # concatenate the pivot 
    conv_feat = np.concatenate(feat2d_list, axis=0)

    # we move the padding feature part into the model
    return conv_feat.astype("float32"), feature_list, step_rate_mean, step_rate_std


def get_multiple_pid(pid_list, num_split=10, ks=(9, 15), start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False):
    """
    get the data for multiple participant id, refer to get_hourly_data for the arguments.
    Args:
        - pid_list: python list containing all the pids to be processed.
    """
    pid_data = [get_hourly_data(pid, num_split, ks, start_hour, end_hour, conv_feat, return_time_dict) for pid in tqdm(pid_list)]  # (conv_feat, feature_list, step_rate_mean, step_rate_std)
    
    return pid_data


if __name__ == "__main__":
    # extract the raw data from the database
    extract_data_from_db()

    # combine the extracted files into a single parquet file
    # sample the minute level data into hourly level
    # push the generated files (all_data.parquet, all_data_index.npy)into google bucket
    combine_db_extracts()



from prefect import flow, task
import pandas as pd
import random
import json
import time


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "../logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "run_logs.jsonl")


def write_log(event):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

@task
def extract():
    write_log({"stage": "extract", "msg": "Extracting data"})
    data = pd.DataFrame({"age": [25, 30, "unknown", 22]})
    return data

@task
def transform(df):
    write_log({"stage": "transform", "msg": "Transforming data"})
    if random.random() < 0.5:
        raise ValueError("TransformError: could not convert age to int")
    df['age'] = df['age'].astype(int)
    return df

@task
def load(df):
    write_log({"stage": "load", "msg": "Loading data"})
    time.sleep(1)
    return True

@flow
def run_pipeline():
    write_log({"event": "start", "msg": "Pipeline started"})
    try:
        df = extract()
        df = transform(df)
        load(df)
        write_log({"event": "success", "msg": "Pipeline completed"})
    except Exception as e:
        write_log({"event": "error", "error": str(e)})
        write_log({"event": "failed", "msg": "Pipeline failed"})

if __name__ == "__main__":
    run_pipeline()
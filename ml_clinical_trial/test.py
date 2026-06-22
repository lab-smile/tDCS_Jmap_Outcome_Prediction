import time
from datetime import datetime
import os

now = datetime.now()
# Format as "year_month_date_hour_minute_second"
formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")

experiment_dir = "./results_" + formatted_datetime
log_path = os.path.join(experiment_dir, f'output_{formatted_datetime}.log')

if experiment_dir and not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)

def printer(log_str):
    with open(log_path, "a") as f:
        f.write(log_str)

def hello(printer):
    printer('hello\n')
    time.sleep(10)
    printer('hello\n')
    time.sleep(10)

hello(printer)
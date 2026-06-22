import os
from datetime import datetime

class Logger:
    def __init__(self, log_path, log_filename):
        date_str = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        self.log_path = log_path + f"_{date_str}"
        self.log_filename = log_filename
            
    def write_log(self, log_message, end = "\n"):
        # Ensure the directory exists
        os.makedirs(self.log_path, exist_ok=True)
        # Append the log message to the file
        with open(os.path.join(self.log_path, self.log_filename), 'a') as f:
            f.write(f"{log_message}{end}")

if __name__ == "__main__":
    log_path = f"logger_test_log"
    log_filename = "test.log"
    logger = Logger(log_path, log_filename)
    logger.write_log(f"Check you current working directory")
    logger.write_log(f"The pwd command (which stands for \"print working directory\")")
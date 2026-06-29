import sys
import os
from datetime import datetime

class StreamToLogger:
    """
    Redirects stdout or stderr to both the terminal and a log file,
    adding timestamps (with milliseconds) to the beginning of each line.
    """
    def __init__(self, stream, filepath):
        self.stream = stream
        self.filepath = filepath
        self.is_start_of_line = True
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("")

    def write(self, message):
        if not message:
            return
            
        now = datetime.now()
        timestamp = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond // 1000:03d}] "
        
        lines = message.split('\n')
        
        for i, line in enumerate(lines):
            if line:
                prefix = timestamp if self.is_start_of_line else ""
                out_str = prefix + line
                
                self.stream.write(out_str)
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write(out_str)
                self.is_start_of_line = False
            
            if i < len(lines) - 1:
                self.stream.write('\n')
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write('\n')
                self.is_start_of_line = True

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


def _unwrap(stream):
    """Peel off any StreamToLogger wrappers to reach the genuine console stream.

    Repeated runs in one process (the GUI stays alive between runs) must each wrap the REAL
    stdout/stderr -- never the previous run's wrapper. Nesting wrappers is what made every run
    re-timestamp the line and write it into all earlier runs' log files.
    """
    while isinstance(stream, StreamToLogger):
        stream = stream.stream
    return stream

def setup_run_folder(env_name="bouncing"):
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_folder = f"{now}__{env_name}"
    run_dir = os.path.join(os.getcwd(), "runs", run_folder)

    os.makedirs(run_dir, exist_ok=True)
    log_filepath = os.path.join(run_dir, "log.txt")

    # Wrap the true console streams, restoring first so back-to-back runs never nest wrappers.
    sys.stdout = StreamToLogger(_unwrap(sys.stdout), log_filepath)
    sys.stderr = StreamToLogger(_unwrap(sys.stderr), log_filepath)

    print(f"--- Started Run: {run_folder} ---")

    return run_dir, log_filepath


def teardown_run_logging():
    """Restore the original console streams so the process is clean once a run finishes
    (otherwise stray prints between runs keep landing in the last run's log.txt)."""
    sys.stdout = _unwrap(sys.stdout)
    sys.stderr = _unwrap(sys.stderr)
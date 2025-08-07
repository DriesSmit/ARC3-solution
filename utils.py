import os
import subprocess
import logging
from datetime import datetime


def get_git_info():
    """Return current git commit hash and uncommitted diff as strings"""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    try:
        diff = subprocess.check_output(['git', 'diff']).decode()
    except subprocess.CalledProcessError:
        diff = ''
    return commit, diff


def save_git_info(base_dir):
    """Save git commit and diff to a file in the given directory"""
    commit, diff = get_git_info()
    path = os.path.join(base_dir, 'git_info.txt')
    with open(path, 'w') as f:
        f.write(f"commit: {commit}\n\n")
        f.write(diff)
    print(f"Saved git info to {path}")


def setup_experiment_directory(base_output_dir='runs'):
    """
    Create directories for outputs and logging. Returns base_dir and environment-specific paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(base_output_dir, timestamp)
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create base structure - environment-specific directories will be created as needed
    env_dirs = {}
    
    # Note: Logging setup will be handled by the caller to redirect to this directory
    log_file = os.path.join(base_dir, 'logs.log')
    
    # Save git info in the base directory
    save_git_info(base_dir)
    
    print(f"Experiment directory created: {base_dir}")
    print(f"Logs will be saved to: {log_file}")
    
    return base_dir, log_file


def get_environment_directory(base_dir, game_id):
    """Get or create environment-specific directory for a game_id"""
    env_dir = os.path.join(base_dir, game_id)
    os.makedirs(env_dir, exist_ok=True)
    return env_dir


def setup_logging_for_experiment(log_file_path):
    """Update logging configuration to use the experiment directory's log file"""
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
    
    # Add new file handler pointing to experiment directory
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    print(f"Logging redirected to: {log_file_path}")
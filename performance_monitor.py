import json
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps

try:
    import psutil
except Exception:
    psutil = None

# Optional import for torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

import subprocess
import tracemalloc

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
STATS_FILE = os.path.join(
    "user_data", "performance_stats.jsonl"
)  # Use JSON Lines for easier appending

# Ensure user_data directory exists
if not os.path.exists("user_data"):
    os.makedirs("user_data", exist_ok=True)
stats_lock = threading.Lock()

# Start tracing Python memory allocations for per-function measurements
tracemalloc.start()


def _get_gpu_util_percent():
    """Return current GPU utilization percent via nvidia-smi, or None if unavailable."""
    try:
        # Query GPU utilization in percent
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        # Multiple GPUs return multiple lines; take first GPU
        line = output.strip().splitlines()[0]
        return int(line.strip())
    except Exception:
        return None


# In-memory aggregation for averages to avoid reading the file constantly
aggregated_stats = defaultdict(lambda: {"total_time": 0.0, "count": 0})
aggregation_lock = threading.Lock()


def measure_performance(func):
    """Decorator to measure execution time of a function and log it."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start measuring time and system resources
        # Wall-clock start time
        start_time = time.perf_counter()
        # CPU time start (process and thread)
        start_cpu_process = time.process_time()
        start_cpu_thread = time.thread_time()
        # Python memory allocations start
        start_traced_mem, _ = tracemalloc.get_traced_memory()
        # GPU utilization start
        start_gpu_util = _get_gpu_util_percent()

        if psutil is not None:
            try:
                proc = psutil.Process()
                start_mem = proc.memory_info().rss
                cpu_times = proc.cpu_times()
                start_cpu = cpu_times.user + cpu_times.system
            except Exception:
                start_mem = start_cpu = None
        else:
            start_mem = start_cpu = None

        try:
            start_gpu = (
                torch.cuda.memory_allocated()
                if TORCH_AVAILABLE and torch.cuda.is_available()
                else None
            )
        except Exception:
            start_gpu = None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Record end times
            end_time = time.perf_counter()
            duration = end_time - start_time
            end_cpu_process = time.process_time()
            end_cpu_thread = time.thread_time()
            end_traced_mem, end_traced_peak = tracemalloc.get_traced_memory()
            func_name = func.__name__
            module_name = func.__module__
            full_name = f"{module_name}.{func_name}"

            # Post-call resource measurements
            if psutil is not None:
                try:
                    proc = psutil.Process()
                    end_mem = proc.memory_info().rss
                    cpu_times_end = proc.cpu_times()
                    end_cpu = cpu_times_end.user + cpu_times_end.system
                except Exception:
                    end_mem = end_cpu = None
            else:
                end_mem = end_cpu = None
            # Post-call GPU stats
            try:
                end_gpu = (
                    torch.cuda.memory_allocated()
                    if TORCH_AVAILABLE and torch.cuda.is_available()
                    else None
                )
            except Exception:
                end_gpu = None
            # GPU utilization end
            end_gpu_util = _get_gpu_util_percent()

            # Build log entry with performance and resource stats
            log_entry = {
                "timestamp": time.time(),
                "function": full_name,
                "duration_ms": duration * 1000,  # Store in milliseconds
            }
            # Memory usage
            if start_mem is not None and end_mem is not None:
                log_entry["memory_rss_bytes"] = end_mem
                log_entry["memory_delta_bytes"] = end_mem - start_mem
            # CPU time and utilization
            if start_cpu is not None and end_cpu is not None:
                cpu_delta = end_cpu - start_cpu
                log_entry["cpu_time_ms"] = cpu_delta * 1000
                if duration > 0:
                    log_entry["cpu_percent"] = (cpu_delta / duration) * 100
            # GPU memory usage
            if start_gpu is not None and end_gpu is not None:
                log_entry["gpu_memory_bytes"] = end_gpu
                log_entry["gpu_memory_delta_bytes"] = end_gpu - start_gpu
            # GPU utilization
            if start_gpu_util is not None:
                log_entry["gpu_util_start_percent"] = start_gpu_util
            if end_gpu_util is not None:
                log_entry["gpu_util_end_percent"] = end_gpu_util
            # Python memory allocations via tracemalloc
            log_entry["memory_traced_bytes"] = end_traced_mem
            log_entry["memory_traced_delta_bytes"] = end_traced_mem - start_traced_mem
            log_entry["memory_traced_peak_bytes"] = end_traced_peak
            # CPU time usage per function
            proc_cpu_delta = None
            try:
                proc_cpu_delta = end_cpu_process - start_cpu_process
                log_entry["cpu_process_time_ms"] = proc_cpu_delta * 1000
            except Exception:
                pass
            try:
                thread_cpu_delta = end_cpu_thread - start_cpu_thread
                log_entry["cpu_thread_time_ms"] = thread_cpu_delta * 1000
            except Exception:
                pass
            try:
                with stats_lock:
                    with open(STATS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
            except OSError as e:
                logger.error(f"Error writing performance stats to {STATS_FILE}: {e}")

            # Update in-memory aggregation
            with aggregation_lock:
                aggregated_stats[full_name]["total_time"] += duration
                aggregated_stats[full_name]["count"] += 1

            # Optional: Log to console logger as well
            # logger.debug(f"PERF: {full_name} took {duration:.4f}s")

    return wrapper


def load_and_aggregate_stats():
    """Loads all stats from the file and aggregates them."""
    global aggregated_stats
    temp_aggregated_stats = defaultdict(lambda: {"total_time": 0.0, "count": 0})
    try:
        with stats_lock:  # Ensure file isn't being written to while reading
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            func_name = entry.get("function")
                            duration_ms = entry.get("duration_ms")
                            if func_name and duration_ms is not None:
                                # Convert duration back to seconds for aggregation consistency
                                duration_sec = duration_ms / 1000.0
                                temp_aggregated_stats[func_name][
                                    "total_time"
                                ] += duration_sec
                                temp_aggregated_stats[func_name]["count"] += 1
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping invalid line in {STATS_FILE}: {line.strip()}"
                            )
    except OSError as e:
        logger.error(f"Error reading performance stats file {STATS_FILE}: {e}")
        # Return current in-memory stats if file reading fails
        with aggregation_lock:
            return aggregated_stats.copy()  # Return a copy

    # Update the global in-memory stats with the freshly loaded data
    with aggregation_lock:
        aggregated_stats = temp_aggregated_stats
        return aggregated_stats.copy()  # Return a copy


def get_average_times():
    """Calculates and returns average execution time for each function."""
    averages = {}
    # Load latest aggregates from file first
    current_aggregates = load_and_aggregate_stats()

    for func_name, data in current_aggregates.items():
        if data["count"] > 0:
            avg_time_sec = data["total_time"] / data["count"]
            averages[func_name] = {
                "average_ms": avg_time_sec * 1000,
                "count": data["count"],
            }
    # Sort by average time descending
    sorted_averages = dict(
        sorted(averages.items(), key=lambda item: item[1]["average_ms"], reverse=True)
    )
    return sorted_averages


def clear_performance_stats():
    """Clears the performance statistics file."""
    global aggregated_stats
    try:
        with stats_lock:
            if os.path.exists(STATS_FILE):
                os.remove(STATS_FILE)
        with aggregation_lock:
            aggregated_stats = defaultdict(lambda: {"total_time": 0.0, "count": 0})
        logger.info(f"Performance stats file {STATS_FILE} cleared.")
        return True
    except OSError as e:
        logger.error(f"Error clearing performance stats file {STATS_FILE}: {e}")
        return False


# Load existing stats on module import
load_and_aggregate_stats()

# Example usage (can be removed later)
# @measure_performance
# def example_function(duration):
#     time.sleep(duration)

# if __name__ == "__main__":
#     print("Running example...")
#     example_function(0.1)
#     example_function(0.2)
#     print("Current Stats:", _stats_data)
#     print("Averages:", get_average_times())
#     # save_stats() # atexit handles this

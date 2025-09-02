# file_tracker.py
import sys
import os

called_files = set()

WORKSPACE_ROOT = os.path.abspath("/home/sachin/IsaacLab")

def trace_calls(frame, event, arg):
    if event == "call":
        filename = frame.f_globals.get("__file__")
        if filename:
            abs_path = os.path.abspath(filename)
            if abs_path.startswith(WORKSPACE_ROOT):  # Only your code
                called_files.add(abs_path)
    return trace_calls


def start_tracing():
    sys.settrace(trace_calls)

def stop_tracing():
    sys.settrace(None)
    return list(sorted(called_files))

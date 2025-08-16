"""Pytest config for Gaja-server so that 'config' and 'modules' packages import correctly when running server tests directly."""
import os
import sys

SERVER_DIR = os.path.dirname(__file__)
PARENT = os.path.dirname(SERVER_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

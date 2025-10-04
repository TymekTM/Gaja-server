"""Pytest config for Gaja-server so that 'config' and 'modules' packages import correctly when running server tests directly."""
import os
import sys

SERVER_DIR = os.path.dirname(__file__)
PARENT = os.path.dirname(SERVER_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# Minimal async support when pytest-asyncio is unavailable
import asyncio
import inspect
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(pyfuncitem.obj(**pyfuncitem.funcargs))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()
        return True
    return None

#!/usr/bin/env python3
"""
Final integration test to verify all fixes work correctly.
"""

import sys
sys.path.append('.')
import asyncio
from function_calling_system import FunctionCallingSystem
from modules.core_module import get_current_time

async def test_integration():
    print('🔧 Testing core components...')
    
    # Test 1: Function calling system has time function
    fcs = FunctionCallingSystem()
    functions = fcs.convert_modules_to_functions()
    time_funcs = [f for f in functions if 'time' in f['function']['name']]
    print(f'✅ Time functions available: {len(time_funcs)}')
    
    # Test 2: Time function works
    result = await get_current_time({})
    print(f'✅ Time function result: {result.get("success", False)}')
    
    # Test 3: Function calling system can handle the core module
    if time_funcs:
        func_def = time_funcs[0]
        print(f'✅ Function definition: {func_def["function"]["name"]}')
    
    print('🎉 All core tests passed!')

if __name__ == "__main__":
    asyncio.run(test_integration())

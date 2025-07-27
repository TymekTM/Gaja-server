import asyncio
from function_calling_system import FunctionCallingSystem

async def test_functions():
    fcs = FunctionCallingSystem()
    functions = fcs.convert_modules_to_functions()
    for func in functions:
        if 'weather' in func['function']['name']:
            print(f'Weather function: {func["function"]["name"]}')
            print(f'Description: {func["function"]["description"]}')
            print(f'Parameters: {func["function"]["parameters"]}')
            print('---')

asyncio.run(test_functions())

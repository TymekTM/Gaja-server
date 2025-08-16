import sys, asyncio, json
from collections import deque

# ensure relative imports work when running from project root
if 'Gaja-server' not in sys.path:
    sys.path.append('Gaja-server')

from core.function_calling_system import get_function_calling_system  # noqa: E402
from modules.ai_module import generate_response  # noqa: E402

async def main():
    f1 = get_function_calling_system()
    f2 = get_function_calling_system()
    print('Singleton same object:', f1 is f2)
    funcs1 = f1.convert_modules_to_functions()
    funcs2 = f2.convert_modules_to_functions()
    print('Functions cached same list object:', funcs1 is funcs2, 'count', len(funcs1))
    h = deque([{'role':'user','content':'Powiedz krótko witaj.'}])
    r = await generate_response(h, detected_language='pl', language_confidence=1.0, tracking_id='singleton_test', enable_latency_trace=False)
    print('Resp len', len(r))
    print('Resp preview', r[:160])

if __name__ == '__main__':
    asyncio.run(main())

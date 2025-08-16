import sys, asyncio, time
from collections import deque
if 'Gaja-server' not in sys.path:
    sys.path.append('Gaja-server')
from modules.ai_module import generate_response

async def one_call(i):
    hist = deque([{'role':'user','content':'Powiedz jedno zdanie powitania.'}])
    start = time.perf_counter()
    r = await generate_response(hist, detected_language='pl', language_confidence=1.0, tracking_id=f'baseline_singleton_{i}')
    dur = (time.perf_counter()-start)*1000
    return dur, len(r)

async def main():
    durs = []
    for i in range(2):
        d, l = await one_call(i)
        print(f'Call {i} duration_ms={d:.1f} resp_len={l}')
        durs.append(d)
    if durs:
        print('Avg ms:', sum(durs)/len(durs))

if __name__ == '__main__':
    asyncio.run(main())

"""Test 3x concurrent API calls to chatanywhere."""
import asyncio
import time
import httpx

API_KEY = "sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD"
BASE_URL = "https://api.chatanywhere.tech/v1/chat/completions"
MODEL = "gpt-4o-mini"


async def single_call(client, call_id):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"What is 2+{call_id}? Answer with just the number."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    start = time.time()
    try:
        response = await client.post(BASE_URL, json=payload, headers=headers, timeout=30)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return f"call_{call_id}: OK ({elapsed:.2f}s) answer={answer}"
        else:
            return f"call_{call_id}: HTTP {response.status_code} ({elapsed:.2f}s) body={response.text[:100]}"
    except Exception as error:
        elapsed = time.time() - start
        return f"call_{call_id}: ERROR ({elapsed:.2f}s) {error}"


async def test_concurrency(num_concurrent):
    print(f"\n--- Testing {num_concurrent} concurrent calls ---")
    async with httpx.AsyncClient() as client:
        start = time.time()
        tasks = [single_call(client, i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total = time.time() - start

    for result in results:
        print(f"  {result}")
    print(f"  Total time: {total:.2f}s (vs serial estimate: {num_concurrent * 0.8:.1f}s)")
    return all("OK" in r for r in results)


async def main():
    # Test 1 call first
    ok1 = await test_concurrency(1)
    if not ok1:
        print("Single call failed, aborting")
        return

    # Test 3 concurrent
    ok3 = await test_concurrency(3)

    # Test 5 concurrent
    ok5 = await test_concurrency(5)

    print(f"\n=== RESULTS ===")
    print(f"1 concurrent: {'PASS' if ok1 else 'FAIL'}")
    print(f"3 concurrent: {'PASS' if ok3 else 'FAIL'}")
    print(f"5 concurrent: {'PASS' if ok5 else 'FAIL'}")

    if ok3:
        print("\n3x concurrency is SAFE. Estimated speedup: ~3x")
        print(f"Expected total time with 3x: {4386 * 0.8 / 3 / 60:.0f} min (vs serial {4386 * 0.8 / 60:.0f} min)")


if __name__ == "__main__":
    asyncio.run(main())

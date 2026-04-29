from khaos_personality import Khaos

khaos = Khaos()

queries = [
    "What is the origin of consciousness?",
    "What happens when logic breaks down completely?",
    "Everything is falling apart and I don't know why.",
    "What is the weather today?",  # should NOT activate Khaos
    "Where did the universe come from?",
]

print("\n" + "="*60)
print("KHAOS — ACTIVATION TESTS")
print("="*60)

for q in queries:
    print(f"\nQuery: {q}")
    print("-"*40)
    result = khaos.speak(q)

    if not result["activated"]:
        print("[ Khaos did not activate ]")
        continue

    print(f"Domain: {result['domain']}")
    print(f"Soul drift: {result['soul_drift']:.6f}")
    print(f"\nKhaos speaks:\n{result['response']}")
    print("="*60)

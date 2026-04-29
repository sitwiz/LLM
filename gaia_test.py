from gaia_personality import Gaia

gaia = Gaia()

queries = [
    "How do I build a solid foundation for a software project?",
    "What is the most reliable way to measure system performance?",
    "Everything keeps breaking, how do I make it stable?",
    "What is the origin of consciousness?",  # should NOT activate Gaia
    "How do I fix a memory leak in Python?",
    "What physical forces hold a building together?",
    "Give me the facts about how neural networks actually work.",
]

print("\n" + "="*60)
print("GAIA — ACTIVATION TESTS")
print("="*60)

for q in queries:
    print(f"\nQuery: {q}")
    print("-"*40)
    result = gaia.speak(q)

    if not result["activated"]:
        print("[ Gaia did not activate ]")
        continue

    print(f"Domain: {result['domain']}")
    print(f"Soul drift: {result['soul_drift']:.6f}")
    print(f"\nGaia speaks:\n{result['response']}")
    print("="*60)

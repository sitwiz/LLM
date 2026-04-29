from tartaros_personality import Tartaros

tartaros = Tartaros()

queries = [
    "Why does my distributed system keep failing under load?",
    "What is the root cause of technical debt?",
    "How do I design a scalable database architecture?",
    "What is the weather today?",  # should NOT activate
    "What is really happening when a neural network learns?",
    "Why do complex systems always seem to break in unexpected ways?",
    "What lies beneath the surface of consciousness?",
]

print("\n" + "="*60)
print("TARTAROS — ACTIVATION TESTS")
print("="*60)

for q in queries:
    print(f"\nQuery: {q}")
    print("-"*40)
    result = tartaros.speak(q)

    if not result["activated"]:
        print("[ Tartaros did not activate ]")
        continue

    print(f"Domain: {result['domain']}")
    print(f"Soul drift: {result['soul_drift']:.6f}")
    print(f"\nTartaros speaks:\n{result['response']}")
    print("="*60)

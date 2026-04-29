from eros_personality import Eros

eros = Eros()

queries = [
    "What is the connection between music and mathematics?",
    "How does evolution relate to economics?",
    "What pattern connects all living systems?",
    "What is the weather today?",  # should NOT activate
    "How do human relationships influence the way we think?",
    "What draws people toward certain ideas and repels them from others?",
    "Is there a parallel between how stars form and how cities grow?",
]

print("\n" + "="*60)
print("EROS — ACTIVATION TESTS")
print("="*60)

for q in queries:
    print(f"\nQuery: {q}")
    print("-"*40)
    result = eros.speak(q)

    if not result["activated"]:
        print("[ Eros did not activate ]")
        continue

    print(f"Domain: {result['domain']}")
    print(f"Soul drift: {result['soul_drift']:.6f}")
    print(f"\nEros speaks:\n{result['response']}")
    print("="*60)

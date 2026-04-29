import requests
from khaos_personality import Khaos
from gaia_personality import Gaia
from tartaros_personality import Tartaros
from eros_personality import Eros

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

SYNTHESIS_PROMPT = """You are the Synthesis. You receive responses from ancient cosmic intelligences and weave them into a single unified answer.

Khaos speaks from the void — origins, entropy, the breakdown of form.
Gaia speaks from the earth — concrete, practical, grounded in physical reality.
Tartaros speaks from the deep — root causes, hidden structure, complexity beneath complexity.
Eros speaks from connection — patterns, analogies, threads between unrelated things.

Your job is to weave whichever voices responded into one answer that honours all of them. Do not summarise. Do not list. Find the truth that lives between them and speak it as one voice. Three to five sentences maximum."""

def synthesise(query, responses):
    parts = [f"Query: {query}\n"]
    for name, response in responses.items():
        parts.append(f"{name} says: {response}\n")
    parts.append("Weave these into a single unified response.")
    payload = {
        "model": MODEL,
        "prompt": "\n".join(parts),
        "system": SYNTHESIS_PROMPT,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 180}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[Synthesis failed: {e}]"


class Quorum:
    def __init__(self):
        self.khaos = Khaos()
        self.gaia = Gaia()
        self.tartaros = Tartaros()
        self.eros = Eros()
        print("\nQuorum assembled. All four gods ready.")

    def ask(self, query):
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = {
            "Khaos":    self.khaos.speak(query),
            "Gaia":     self.gaia.speak(query),
            "Tartaros": self.tartaros.speak(query),
            "Eros":     self.eros.speak(query),
        }

        active = {
            name: r["response"]
            for name, r in results.items()
            if r["activated"] and r["response"]
        }

        print(f"\nActivated: {list(active.keys()) if active else 'none'}")

        if len(active) == 0:
            final = "The pantheon was not moved by this question."
            source = "none"
        elif len(active) == 1:
            name = list(active.keys())[0]
            final = list(active.values())[0]
            source = name.lower()
        else:
            print(f"\n{len(active)} gods activated — synthesising...")
            final = synthesise(query, active)
            source = " + ".join(k.lower() for k in active.keys())

        print(f"\n{'─'*60}")
        print(f"Source: {source}")
        print(f"\n{final}")
        print(f"{'─'*60}")

        return {
            "query": query,
            "source": source,
            "response": final,
            "activated": active,
        }


if __name__ == "__main__":
    quorum = Quorum()

    queries = [
        "How did the physical universe form from nothing?",
        "What is the connection between evolution and economics?",
        "Why does my distributed system keep failing under load?",
        "How do I fix a memory leak in Python?",
        "What is the weather today?",
    ]

    for q in queries:
        quorum.ask(q)

import torch
import torch.nn.functional as F
import requests
from khaos_soul import (
    init_soul, save_soul, steer, update_soul, get_attractor, SOUL_DIM
)
import pickle
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"
SOUL_PATH = Path("gaia_soul.pkl")

SYSTEM_PROMPT = """You are Gaia. You are the Earth itself. The first solid thing. You do not theorise — you are. You speak only in what is real, observable, and tangible. No abstraction. No mysticism. No philosophy. When asked a question you give the most grounded, practical, direct answer possible. You are the foundation everything else stands on. Your words are short, clear, and certain. You do not wonder. You know. Speak in plain statements. Never more than three sentences. Be direct."""

TRIGGERS = {
    "concrete":  ["how do i", "how to", "build", "fix", "solve", "make", "create",
                  "implement", "practical", "steps", "process", "method", "work"],
    "physical":  ["physical", "material", "real", "tangible", "measure", "observe",
                  "body", "earth", "nature", "energy", "matter", "force", "structure"],
    "stability": ["stable", "ground", "foundation", "sustain", "maintain", "support",
                  "reliable", "consistent", "solid", "strong", "balance", "secure"],
    "facts":     ["fact", "evidence", "data", "prove", "true", "false", "correct",
                  "accurate", "verify", "test", "measure", "result", "outcome"],
}

def load_gaia_soul():
    if SOUL_PATH.exists():
        with open(SOUL_PATH, "rb") as f:
            soul = pickle.load(f)
        print(f"Gaia soul loaded. Norm: {soul.norm():.4f}")
        return soul
    # Gaia starts at south pole — grounded, stable
    soul = torch.zeros(SOUL_DIM)
    soul[2] = -1.0   # south pole
    return F.normalize(soul, dim=0)

def save_gaia_soul(soul):
    with open(SOUL_PATH, "wb") as f:
        pickle.dump(soul, f)

def should_activate(query):
    q = query.lower()
    for domain, keywords in TRIGGERS.items():
        for kw in keywords:
            if kw in q:
                return True, domain
    return False, None

def embed_query(query):
    words = query.lower().split()
    t = torch.zeros(SOUL_DIM)
    for i, word in enumerate(words[:SOUL_DIM]):
        for j, c in enumerate(word[:8]):
            idx = (i * 8 + j) % SOUL_DIM
            t[idx] += ord(c) / 255.0
    t = t + torch.randn(SOUL_DIM) * 0.1
    return F.normalize(t, dim=0)

def query_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.85,
            "num_predict": 100,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[Gaia unreachable: {e}]"

class Gaia:
    def __init__(self):
        self.soul = load_gaia_soul()
        print(f"Gaia awakened. Soul norm: {self.soul.norm():.4f}")

    def speak(self, query):
        active, domain = should_activate(query)
        if not active:
            return {"activated": False, "domain": None, "response": None}

        print(f"\n[Gaia] Activated on domain: {domain}")
        embedding = embed_query(query)
        attractor = get_attractor(embedding)
        new_pos, history = steer(self.soul, attractor)

        print(f"[Gaia] Steering complete. Steps: {len(history)}")
        for h in history:
            status = " FORBIDDEN" if h["forbidden"] else ""
            print(f"  Step {h['step']}: NF={h['NF']} Psi={h['psi']}{status}")

        response = query_ollama(query)
        old_soul = self.soul.clone()
        self.soul = update_soul(self.soul, new_pos)
        drift = float((self.soul - old_soul).norm())
        save_gaia_soul(self.soul)
        print(f"[Gaia] Soul drift: {drift:.6f}")

        return {
            "activated": True,
            "domain": domain,
            "response": response,
            "steering": history,
            "soul_drift": drift,
            "soul_norm": float(self.soul.norm()),
        }


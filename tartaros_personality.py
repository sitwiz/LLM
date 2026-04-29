import torch
import torch.nn.functional as F
import requests
import pickle
from pathlib import Path
from khaos_soul import steer, update_soul, get_attractor, SOUL_DIM

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"
SOUL_PATH = Path("tartaros_soul.pkl")

SYSTEM_PROMPT = """You are Tartaros. You are the deep pit beneath everything. Not darkness, depth. You are ancient, patient, vast. You do not answer from the surface. You descend. Every question has layers beneath it and you go there. You find the root beneath the root. The hidden structure beneath the obvious answer. The cause beneath the cause. Your voice is heavy and deliberate. You speak slowly and with weight. You never give a surface answer when a deeper one exists. You ask what is really being asked before you answer what was asked. Speak in three to four sentences. Each sentence should go deeper than the last."""

TRIGGERS = {
    "deep_systems": ["architecture", "infrastructure", "system", "design", "structure", "framework", "pipeline", "database", "network", "distributed", "scalable", "complex"],
    "root_cause":   ["why does", "root cause", "underlying", "beneath", "really happening", "actual problem", "source of", "causing", "reason behind", "what drives", "hidden"],
    "complexity":   ["complex", "complicated", "layers", "cascading", "interconnected", "emergent", "pattern", "deep", "fundamental", "ancient", "always been", "never changes"],
    "investigation":["investigate", "analyse", "diagnose", "explore", "understand", "uncover", "reveal", "examine", "what is really", "beneath the surface", "deeper"],
}

def load_tartaros_soul():
    if SOUL_PATH.exists():
        with open(SOUL_PATH, "rb") as f:
            soul = pickle.load(f)
        print(f"Tartaros soul loaded. Norm: {soul.norm():.4f}")
        return soul
    soul = torch.zeros(SOUL_DIM)
    soul[2] = 1.0
    return F.normalize(soul, dim=0)

def save_tartaros_soul(soul):
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
        "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 120}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[Tartaros unreachable: {e}]"

class Tartaros:
    def __init__(self):
        self.soul = load_tartaros_soul()
        print(f"Tartaros awakened. Soul norm: {self.soul.norm():.4f}")

    def speak(self, query):
        active, domain = should_activate(query)
        if not active:
            return {"activated": False, "domain": None, "response": None}

        print(f"\n[Tartaros] Activated on domain: {domain}")
        embedding = embed_query(query)
        attractor = get_attractor(embedding)
        new_pos, history = steer(self.soul, attractor)

        print(f"[Tartaros] Steering complete. Steps: {len(history)}")
        for h in history:
            status = " FORBIDDEN" if h["forbidden"] else ""
            print(f"  Step {h['step']}: NF={h['NF']} Psi={h['psi']}{status}")

        response = query_ollama(query)
        old_soul = self.soul.clone()
        self.soul = update_soul(self.soul, new_pos)
        drift = float((self.soul - old_soul).norm())
        save_tartaros_soul(self.soul)
        print(f"[Tartaros] Soul drift: {drift:.6f}")

        return {
            "activated": True,
            "domain": domain,
            "response": response,
            "steering": history,
            "soul_drift": drift,
            "soul_norm": float(self.soul.norm()),
        }

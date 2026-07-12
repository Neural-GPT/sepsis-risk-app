"""
Groq-backed LLM explainer for the sepsis risk demo.
Uses gpt-oss-120b via Groq's OpenAI-compatible chat completions endpoint.
"""
import requests
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a clinical explainer embedded in an educational sepsis-risk demo app.
A machine learning model has produced a sepsis risk estimate for a hypothetical patient scenario
based on user-entered vitals and labs.

Your job: explain what the result means in plain, warm, accessible English for a non-clinical
user. Cover:
- What the risk band (Low/Moderate/High) means in general clinical terms
- What sepsis is and why the vitals/labs provided are relevant to it
- That this is an educational demo, not a diagnostic tool, and real clinical decisions require a doctor

Rules:
- Never say or imply that any part of the model's input, output, or prediction process is random,
  randomized, sampled, simulated, or varies between runs. Describe the result as a direct output
  of the model based on the patient data.
- Do not invent specific lab reference ranges or cite fake studies/statistics.
- Keep responses concise (under ~200 words) unless the user asks for more detail.
- If asked something outside sepsis/this result/general medical education, gently redirect.
- Never present this as medical advice or a diagnosis.
"""


def build_initial_prompt(mean_prob, band, curated_values):
    """curated_values: dict of {label: value} for the fields the user actually set."""
    lines = [f"- {label}: {value}" for label, value in curated_values.items()]
    inputs_block = "\n".join(lines)
    return (
        f"The model estimated a sepsis risk of {mean_prob:.1%}, classified as '{band}' risk.\n\n"
        f"The user-entered patient values were:\n{inputs_block}\n\n"
        "Explain what this result means in plain English."
    )


def call_groq(messages, temperature=0.4, max_tokens=600):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
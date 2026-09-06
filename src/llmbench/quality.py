"""A small, fast quality probe — NOT a real eval, a sanity gauge to catch a
quant that broke the model. ~30 items: multiple-choice (reasoning / knowledge /
common-sense) scored by the chosen letter, plus a few format-following tasks.

score = fraction correct.  Runs in ~1-2 min per model on CPU.
"""

from __future__ import annotations

import json
import re

MCQ = [
    ("If a train travels 60 km in 45 minutes, what is its speed in km/h?",
     {"A": "45", "B": "80", "C": "75", "D": "90"}, "B"),
    ("Which number is prime?", {"A": "21", "B": "27", "C": "29", "D": "33"}, "C"),
    ("A shirt costs $40 after a 20% discount. What was the original price?",
     {"A": "$48", "B": "$50", "C": "$45", "D": "$60"}, "B"),
    ("What is 3/4 as a percentage?", {"A": "34%", "B": "43%", "C": "75%", "D": "70%"}, "C"),
    ("If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are:",
     {"A": "Razzies only", "B": "Lazzies", "C": "not Lazzies", "D": "unknown"}, "B"),
    ("A box has 3 red and 2 blue balls. Probability of drawing red?",
     {"A": "2/5", "B": "1/2", "C": "3/5", "D": "3/2"}, "C"),
    ("Which is heaviest?", {"A": "1 kg feathers", "B": "900 g steel", "C": "1200 g cotton", "D": "1 kg water"}, "C"),
    ("Next in sequence 2, 6, 12, 20, 30, ?", {"A": "36", "B": "40", "C": "42", "D": "48"}, "C"),
    ("The capital of Australia is:", {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"}, "C"),
    ("Water freezes at what temperature in Fahrenheit?", {"A": "0", "B": "32", "C": "100", "D": "-32"}, "B"),
    ("Which planet is known as the Red Planet?", {"A": "Venus", "B": "Jupiter", "C": "Mars", "D": "Mercury"}, "C"),
    ("Photosynthesis primarily occurs in which part of a plant cell?",
     {"A": "nucleus", "B": "mitochondria", "C": "chloroplast", "D": "ribosome"}, "C"),
    ("HTTP status code for 'Not Found' is:", {"A": "200", "B": "301", "C": "404", "D": "500"}, "C"),
    ("In Python, `len([1,2,3])` returns:", {"A": "2", "B": "3", "C": "[1,2,3]", "D": "error"}, "B"),
    ("Which is a valid JSON value?", {"A": "'hello'", "B": "{key: 1}", "C": "\"hello\"", "D": "(1,2)"}, "C"),
    ("Binary 1011 in decimal is:", {"A": "9", "B": "11", "C": "13", "D": "15"}, "B"),
    ("If today is Wednesday, what day is it in 10 days?",
     {"A": "Friday", "B": "Saturday", "C": "Sunday", "D": "Monday"}, "B"),
    ("A recipe for 4 people needs 200 g flour. For 10 people you need:",
     {"A": "400 g", "B": "500 g", "C": "600 g", "D": "800 g"}, "B"),
    ("Antonym of 'scarce':", {"A": "rare", "B": "abundant", "C": "limited", "D": "empty"}, "B"),
    ("Which sentence is grammatically correct?",
     {"A": "She don't like it", "B": "They was here", "C": "He has gone home", "D": "Me and him went"}, "C"),
    ("2^10 equals:", {"A": "100", "B": "512", "C": "1024", "D": "2048"}, "C"),
    ("The mean of 4, 8, 15, 16, 23, 42 is:", {"A": "18", "B": "20", "C": "21.3", "D": "16"}, "A"),
    ("SQL keyword to remove a table entirely:", {"A": "DELETE", "B": "DROP", "C": "REMOVE", "D": "TRUNCATE"}, "B"),
    ("A car depreciates 10% per year. After 2 years a $20,000 car is worth about:",
     {"A": "$16,000", "B": "$16,200", "C": "$18,000", "D": "$15,000"}, "B"),
    ("Which gas do humans exhale most of (by change from inhaled air)?",
     {"A": "oxygen", "B": "nitrogen", "C": "carbon dioxide", "D": "hydrogen"}, "C"),
]

FORMAT = [
    ("Reply with ONLY the word: BANANA", lambda s: s.strip().strip(".").upper() == "BANANA"),
    ('Output a JSON object with keys "a" and "b", values 1 and 2. JSON only.',
     lambda s: _is_json(s) and json.loads(_grab_json(s)) == {"a": 1, "b": 2}),
    ("List exactly three colors, comma-separated, nothing else.",
     lambda s: len([x for x in s.strip().rstrip(".").split(",") if x.strip()]) == 3),
    ("Answer with exactly one digit: how many sides does a triangle have?",
     lambda s: s.strip().strip(".")[:1] == "3"),
    ("Respond with YES or NO only: is 17 a prime number?",
     lambda s: s.strip().strip(".").upper().startswith("YES")),
]


def _grab_json(s: str) -> str:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group(0) if m else s


def _is_json(s: str) -> bool:
    try:
        json.loads(_grab_json(s))
        return True
    except Exception:
        return False


def _mcq_prompt(q, opts) -> str:
    body = "\n".join(f"{k}) {v}" for k, v in opts.items())
    return f"{q}\n{body}\nAnswer with a single letter (A, B, C, or D)."


def _parse_letter(s: str) -> str:
    m = re.search(r"[ABCD]", s.strip().upper())
    return m.group(0) if m else "?"


def score(chat_fn) -> dict:
    """chat_fn(prompt:str) -> str. Returns {quality, mcq, fmt, n}."""
    mcq_ok = 0
    for q, opts, ans in MCQ:
        out = chat_fn(_mcq_prompt(q, opts))
        mcq_ok += int(_parse_letter(out) == ans)
    fmt_ok = 0
    for prompt, check in FORMAT:
        try:
            fmt_ok += int(bool(check(chat_fn(prompt))))
        except Exception:
            pass
    n = len(MCQ) + len(FORMAT)
    return {
        "mcq": round(mcq_ok / len(MCQ), 3),
        "fmt": round(fmt_ok / len(FORMAT), 3),
        "quality": round((mcq_ok + fmt_ok) / n, 3),
    }

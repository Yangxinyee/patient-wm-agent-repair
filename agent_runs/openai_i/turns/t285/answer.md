The recovered notes show why raw associations are misleading: insulin is associated with higher glucose, but history-adjusted estimates are near −2 per unit of rate. The MAP estimates need closer inspection. I’ll recover the identification checks and retained replay helpers; consistent estimates across methods and sensors would support calibration, while disagreement would argue for a more cautious target.

```python
import inspect, re
text254 = Path("/work/notes.md").read_text()
print("EARLY IDENTIFICATION NOTES")
print(text254[3000:9100])
print("\nIMPORTANT LATER SECTIONS")
sections254 = re.split(r"(?=^## )", text254, flags=re.M)
for sec in sections254:
    if re.search(r"Python (186|188|203|204|209|215|216|220|222)\b", sec[:160]):
        print(sec[:850])
print("\nHELPERS")
for key, value in list(globals().items()):
    if callable(value) and getattr(value, "__module__", "") == "__main__":
        try:
            print(key, str(inspect.signature(value))[:180])
        except Exception:
            print(key)
print("\nGLOBAL NAMES")
print(" ".join(sorted(k for k in globals() if not k.startswith("_")))[:2600])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 254 — compact identification and helper recovery\n"
            "- Recovered raw-association reversal and history-adjusted insulin slopes near -2.\n"
            "- Recovering nonlinear identification, MAP cross-sensor checks, and replay helpers.\n"
            "- No weights changed.\n")
```
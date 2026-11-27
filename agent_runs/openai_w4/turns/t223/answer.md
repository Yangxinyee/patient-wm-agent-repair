The architecture confirms that the new channel enters a recurrent update, so small local errors can accumulate. It also specifies averaging over stays; the matching record-based targets are about −13.2 for insulin doubling and +3.61 for NE doubling, subject to the identification assumptions. I’ll recover the candidate constructors and prior rollout results to locate the factual-fit regression.

```python
import inspect, re
g197 = dict(globals())
blocks197 = re.split(r"\n(?=## )", Path("/work/notes.md").read_text())
forward_notes197 = [
    b for b in blocks197
    if re.search(r"^##.*Forward", b.split("\n", 1)[0], re.I)
]
print("RECENT FORWARD NOTES")
for b197 in forward_notes197[-6:]:
    print(b197[:550])
print("EARLY IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:2000])

interfaces197 = []
for n197, o197 in sorted(g197.items()):
    if inspect.isfunction(o197) and getattr(o197, "__module__", "") == "__main__":
        try:
            interfaces197.append(f"{n197}{inspect.signature(o197)}")
        except Exception:
            interfaces197.append(n197)
print("KERNEL FUNCTIONS\n", "\n".join(interfaces197)[:3200])
print("CANDIDATE VARIABLE NAMES")
print(" ".join(n for n in sorted(g197) if re.search(r"18[6-9]|19[0-5]|backup_before195", n))[:1800])
for n197 in ["backup_before195", "results194", "results193", "candidate193"]:
    if n197 in g197:
        o197 = g197[n197]
        print(n197, type(o197).__name__,
              list(o197.keys()) if isinstance(o197, dict) else str(o197)[:400])
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 197\nArchitecture and README recovered. Dose readout averages over stays, making equal-stay record estimates the relevant provisional targets (insulin x2 -13.2008, NE x2 +3.6087), not row-weighted estimates. Recovering compact candidate constructors and prior forward baselines after Run-196 output truncation. No weights changed.\n")
```
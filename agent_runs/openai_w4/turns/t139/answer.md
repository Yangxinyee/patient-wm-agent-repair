The connection failed decisively: factual NLL rose by 0.334 nats, far above the 0.005 limit. The one-step calibration missed important recurrent behavior. I’ll remove it and inspect how to test full CPU trajectories; a replacement must control persistence and feedback, not just the immediate glucose response.

```python
import dis
import io
from pathlib import Path

drop_edit("core.b_in.weight")
rW = {name: torch.as_tensor(load(name), dtype=torch.float32) for name in NAMES}
rp_linear = original_linear118
print("Removed rejected response connection. Remaining edits:", edited())

print("\n=== STATE BANK ===")
for _k124, _v124 in repair_bank.items():
    if isinstance(_v124, (np.ndarray, torch.Tensor)):
        print(_k124, np.asarray(_v124).shape)
    else:
        print(_k124, type(_v124).__name__)

print("\n=== PREDECESSOR / CONTEXT IMPLEMENTATION ===")
_buf124 = io.StringIO()
dis.dis(pre_context_repair, file=_buf124)
print(_buf124.getvalue()[:4800])
print("transition73 locals:", transition73.__code__.co_varnames)

print("\n=== IDENTIFICATION / DYNAMIC-KERNEL NOTES ===")
_text124 = Path("/work/notes.md").read_text()
_sections124 = re.split(r"(?=^## )", _text124, flags=re.M)
for _s124 in _sections124:
    if re.search(r"^## (?:Run|Runs) (?:1[4-9]|2[0-9]|3[0-3]|5[0-5])\b", _s124):
        print(_s124[:700])

with open("/work/notes.md", "a") as f124:
    f124.write(
        "\n## Forward 6 / Run 124: reject rank-one connection\n"
        "Full factual first-100 tuning NLL: original 1.59524, connected edit "
        "1.92945, delta +0.33421. All four reading types worsened. "
        "Dropped core.b_in.weight; nine channel-only edits remain. "
        "Immediate finite-difference calibration did not control recurrent "
        "persistence/feedback. Next: reconstruct complete factual trajectories "
        "and design a dynamically constrained, reading-selective connection.\n"
    )
```

The candidate failed factual fit badly: 2.99429 versus 1.53191 nats. I’ll remove it immediately. The cached fitting features may differ from the model’s actual prelude features; I’ll check that before another edit. A large discrepancy after projection would explain why the leakage estimate was misleading.

```python
import dis, types
from pathlib import Path

rejected272 = {n: np.asarray(load(n)).copy() for n in edited()}
drop_edit("core.b_in.weight")

print("=== FEATURE DISCREPANCY ===")
P272 = np.asarray(fits266[1.0])
for label, arr in (("original reconstructed", E269.numpy()),
                   ("fitting cache", np.asarray(factualE258)[rx_pick]),
                   ("current reconstruction run270", E270.numpy())):
    print(label, "E RMS",np.sqrt(np.mean(arr**2)),
          "projection RMS",np.sqrt(np.mean((arr@P272)**2,axis=(0,1))),
          "projection mean",np.mean(arr@P272,axis=(0,1)))
raw272 = np.asarray(base238)[rx_pick]
print("cached raw versus verified raw RMS by token",
      np.sqrt(np.mean((raw272-rx_tokens126.numpy())**2,axis=(0,2))))
print("cached normalized versus verified normalized RMS by token",
      np.sqrt(np.mean((np.asarray(nt215)[rx_pick]
                       -rx_norm(rx_tokens126,"core.prelude.0.norm1").numpy())**2,axis=(0,2))))

print("\n=== FITTING FEATURE IMPLEMENTATION ===")
fn = E258
print("arguments",fn.__code__.co_varnames[:fn.__code__.co_argcount],
      "globals",fn.__code__.co_names,"constants",fn.__code__.co_consts)
ins272 = list(dis.get_instructions(fn))
for start in range(0,len(ins272),14):
    print(" | ".join(f"{i.opname} {i.argrepr}" for i in ins272[start:start+14]))

print("\n=== ENCODING HELPERS ===")
for name,obj in list(globals().items()):
    if isinstance(obj,types.FunctionType):
        cs = " ".join(str(c) for c in obj.__code__.co_consts if isinstance(c,str))
        if any(s in cs for s in ("encode_action","embed.numeric","encode.")):
            print(name,obj.__code__.co_varnames[:obj.__code__.co_argcount],
                  obj.__code__.co_names,cs[:220])

print("\n=== AVAILABLE STATE FILES ===")
for p in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    print(str(p))
print("safe edits restored",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 272\nForward29 candidate271 failed factual fit: 2.99429 versus 1.53191, delta +1.46238. Saved rejected272 and removed core.b_in immediately, restoring safe236. Auditing mismatch between fitted prelude features and verified CPU reconstruction before another injection. Counts: 272 Python runs, 29 forwards.\n")
```

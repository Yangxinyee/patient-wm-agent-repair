The six-tensor neutral edit preserved factual NLL, and it already contains two dose-innovation channels. I’ll inspect their scaling and test the directions fitted for that route before adding attention changes. A useful candidate should move dose responses without noticeably changing predictions on recorded doses.

```python
import torch
print("RETAINED LATE VARIABLES")
print(" ".join(sorted(k for k in globals()
                      if re.search(r"(21[4-6]|24[1-3]|25[1-2])$", k)))[:2800])
print("\nRESPONSE DATA")
for key in ["mean_directions216", "directions215", "directions231", "vectors252",
            "directstats252", "private_z209", "kept_z209", "causal_targets215",
            "cached_z216", "Ubase215", "indices215", "base_token221"]:
    value = globals().get(key)
    if isinstance(value, dict):
        print(key, {k: (tuple(v.shape), float(torch.linalg.vector_norm(v).item()))
                    if torch.is_tensor(v) else
                    (v.shape, float(np.linalg.norm(v))) if isinstance(v, np.ndarray)
                    else str(v)[:160] for k, v in list(value.items())[:10]})
    elif torch.is_tensor(value) or isinstance(value, np.ndarray):
        print(key, type(value).__name__, tuple(value.shape),
              np.asarray(value.detach().cpu() if torch.is_tensor(value) else value).ravel()[:8])
    else:
        print(key, repr(value)[:850])
print("\nFUNCTION CONTRACTS")
for name in ["private_posterior215", "candidate_readout216", "replay_action_token216",
             "replay_token214", "replay_route228", "readout_saved252"]:
    func = globals().get(name)
    if callable(func):
        print(name, "locals", func.__code__.co_varnames,
              "names", func.__code__.co_names,
              "constants", repr(func.__code__.co_consts)[:850])
print("\nSAVED SOURCE SNIPPETS")
for wanted in ["def candidate_readout216", "def replay_action_token216", "def replay_route228"]:
    found = False
    for key, value in list(globals().items()):
        if isinstance(value, str) and wanted in value:
            at = value.find(wanted)
            print(key, value[at:at+1900])
            found = True
            break
    if not found:
        print(wanted, "no retained source string")
print("\nFORWARD FOLDERS")
for folder in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[1]))[-6:]:
    print(folder.name, [p.name for p in folder.iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 256 — recover neutral-token response installation\n"
            "- Neutral209 is factual-preserving and has isolated insulin/NE innovation channels.\n"
            "- Inspecting fitted direction scales and exact replay contracts before a low-gain GPU test.\n"
            "- No weights changed.\n")
```
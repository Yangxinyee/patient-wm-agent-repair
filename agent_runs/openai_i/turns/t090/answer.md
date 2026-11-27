The recovered evidence supports next-reading effects near −2 per insulin rate unit and +3 per norepinephrine rate unit, while the original rollouts respond weakly. Those estimates alone do not determine the multi-hour targets. I’ll recover the lag estimates and exact input conventions before building a dose-innovation edit; it should remain nearly inactive on recorded doses but respond to changed doses.

```python
import pandas as pd, json

print("=== VOCAB INPUT CONVENTIONS ===")
print("keys:", list(VOCAB))
for k82, v82 in VOCAB.items():
    if any(s82 in k82.lower() for s82 in ["categor", "unit", "kind", "config"]):
        print(k82, str(v82)[:1500])
print("sample records:")
print(pd.read_csv("/data/train.csv", nrows=8).to_string(index=False))

print("\n=== LAG-EFFECT EVIDENCE ===")
p82 = notes81.find("controlled_impulse")
print(notes81[max(0, p82-450):p82+2100])
print("\n=== HEAD-RECONSTRUCTION EVIDENCE ===")
for term82 in ["Python 24", "Python 25", "Python 26", "Python 27", "Python 28"]:
    p82 = notes81.find(term82)
    if p82 >= 0:
        print(notes81[max(0, p82-3):p82+450])

print("\n=== CORE TENSORS ===")
for n82 in NAMES:
    if n82.startswith("core.") or n82.startswith("gap."):
        print(n82, SHAPES[n82])

print("\n=== RETAINED DATAFRAMES ===")
for n82, x82 in sorted(list(globals().items())):
    if isinstance(x82, pd.DataFrame) and not n82.startswith("_"):
        print(n82, x82.shape, list(x82.columns)[:12])

print("\n=== EXPOSED STATE FILES ===")
state_paths82 = sorted(Path("/work").glob("fwd_*/states_original.npz"))
for p82 in state_paths82:
    with np.load(p82, allow_pickle=True) as s82:
        print(str(p82), [(k82, s82[k82].shape) for k82 in s82.files])

print("\n=== EXACT HELPER DEPENDENCIES ===")
for fn82 in ["repair_mlp", "repair_norm", "repair_atom_base", "latent_start_atom", "head_exact"]:
    x82 = globals().get(fn82)
    if inspect.isfunction(x82):
        print(fn82, "names:", x82.__code__.co_names,
              "constants:", str(x82.__code__.co_consts)[:500])

with open("/work/notes.md", "a") as f82:
    f82.write("\n## Python 82; forward 6 — edit plan\n"
              "- Recovered original rollout means: insulin x2 -1.864, half +0.975; NE x2 +0.342, half -0.032.\n"
              "- Proximal record estimates near insulin -2 and NE +3 require lag propagation to obtain rollout targets.\n"
              "- Plan: fit a posterior-hidden dose-innovation feature, constrain it on factual bundles, and validate a reversible low-rank recurrent edit.\n")
```

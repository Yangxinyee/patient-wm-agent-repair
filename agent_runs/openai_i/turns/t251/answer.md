The zero-response route increased factual NLL by 0.00509, narrowly failing the safeguard. It is not acceptable as submitted. A treatment correction might improve factual fit by modeling the observed dose innovations, but that must be tested. I’ll inspect the replay helpers and paired errors before calibrating such a correction.

```python
import dis, io, contextlib

fo224 = pd.read_csv("/work/fwd_20/factual_original.csv")
fe224 = pd.read_csv("/work/fwd_20/factual_edited.csv")
print("FACTUAL CSV COLUMNS:", fo224.columns.tolist())
print("ORIGINAL FIRST ROW:", fo224.head(1).to_dict("records"))
print("EDITED FIRST ROW:", fe224.head(1).to_dict("records"))

print("\nREPLAY CONTRACTS")
for name224 in ("call_core161", "batch_replay182", "replay_action_token216",
                "replay_token214", "readout214", "encode_actions168"):
    fn224 = globals().get(name224)
    if callable(fn224):
        print(name224, inspect.signature(fn224))
        print(" locals:", fn224.__code__.co_varnames)
        print(" globals:", fn224.__code__.co_names)
        print(" constants:", repr(fn224.__code__.co_consts)[:900])

print("\nACTION REPLAY BYTECODE")
fn224 = globals().get("replay_action_token216")
if callable(fn224):
    buf224 = io.StringIO()
    dis.dis(fn224, file=buf224)
    lines224 = buf224.getvalue().splitlines()
    print("\n".join(lines224)[:3900])

print("\nCALIBRATION CACHE")
for name224 in ("W", "n182", "ids182", "args_cache182", "groups182",
                "causal_targets215", "rate_vector215", "drug_labels215",
                "Ubase215", "official_h182", "official_o182", "base_h214",
                "base_o214", "oldpost182", "e_cal161", "context"):
    ob224 = globals().get(name224)
    if isinstance(ob224, (np.ndarray, torch.Tensor)):
        print(name224, tuple(ob224.shape), str(ob224.dtype))
    elif isinstance(ob224, dict):
        print(name224, "dict", len(ob224), "first keys", list(ob224)[:8])
        if name224 == "causal_targets215":
            print(repr(ob224)[:500])
    elif ob224 is not None:
        print(name224, repr(ob224)[:300])

print("\nEARLY NOTE HEADINGS")
heads224 = re.findall(r"^## .*", Path("/work/notes.md").read_text(), re.M)
print("\n".join(heads224[:35]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 19; Python 224 — small neutral route narrowly fails\n"
            "- First-100 tuning factual NLL 1.63155 original vs 1.63664 edited; dNLL +.00509 exceeds safeguard.\n"
            "- Not acceptable as a final edit. Inspecting replay helpers for forward-only treatment calibration; improvement in factual fit is a hypothesis, not assumed.\n")
```
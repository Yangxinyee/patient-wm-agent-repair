The saved states cover complete bundles, and the exact helpers expose the posterior’s inputs. I’ll check how drug starts and simultaneous infusion endings are represented, then assemble their hidden features. If those features reproduce the saved posterior means, I can fit and test a low-rank dose correction without approximating the bundle encoding.

```python
print("=== RECORD DATAFRAMES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame) and {"stay_id", "hour", "event"}.issubset(obj.columns):
        print(name, obj.shape, list(obj.columns),
              "first stay:", obj["stay_id"].iloc[0])

print("\n=== STATE ARRAYS ===")
for name in ["latent_state", "latent_hour", "state_x", "state_ids", "state_hours"]:
    obj = globals().get(name)
    print(name, type(obj).__name__, getattr(obj, "shape", None))
print("first state IDs/hours:", list(zip(state_ids[:8], state_hours[:8])))

print("\n=== POSTERIOR TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_in", "z_prior", "gap"]):
        print(name, SHAPES[name])
print("posterior helper constants:", post_mean59.__code__.co_consts)
print("context helper constants:", latent_context.__code__.co_consts)

print("\n=== CATEGORICAL MAPS ===")
print("units:", unit_ids57)
print("rate units:", rate_unit_ids57)
print("event kinds:", ekind_ids57)

print("\n=== ACTION CASES WITH OTHER ATOMS ===")
other_cases63 = [case for case in action_cases50 if len(case[2]) > 0]
print("action cases:", len(action_cases50),
      "with other atoms:", len(other_cases63))
for case in other_cases63[:2]:
    print(small_description62(case))

print("\n=== ONE CLEAN ACTION RECONSTRUCTION ===")
i63, action_rows63, other_rows63 = action_cases50[0]
ctx63 = latent_context(np.asarray([i63]))
u63 = action_vector50(action_rows63)
if u63.ndim == 1:
    u63 = u63[None, :]
e63 = empty_set58("encode")
if e63.ndim == 1:
    e63 = e63[None, :]
z63 = post_mean59(ctx63, e63, u63)
target63 = torch.as_tensor(latent_state[i63, -32:], dtype=torch.float32)
print("context/action/empty/posterior shapes:",
      tuple(ctx63.shape), tuple(u63.shape), tuple(e63.shape), tuple(z63.shape))
print("posterior RMS:", float(torch.sqrt(torch.mean((z63.reshape(-1) - target63) ** 2))))

print("\n=== OBSERVATION CASE NAMES ===")
print([n for n in sorted(globals())
       if "case" in n.lower() or "obs" in n.lower()][:80])
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 63; forward 6 — posterior-feature assembly\n"
        "- Checking drug-start and simultaneous-END metadata and a clean action's "
        "posterior reconstruction before fitting a hidden-feature correction.\n"
        "- No weights changed. Exact unchanged-bundle features are required for "
        "credible factual-fit constraints.\n"
    )
```
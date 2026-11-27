The insulin edit is rank one: a dose-sensitive feature triggers a glucose-lowering state update. A possible alternative to weakening it is a small output-layer correction fitted to paired factual states, with guards against cancelling dose responses. I’ll inspect the decoder and aligned state data first; this route is useful only if it improves held-out factual fit.

```python
print("=== DECODER SOURCE ===")
for fn224 in (decode_states175, build_static167):
    try:
        src224 = inspect.getsource(fn224)
        print(src224[:7200 if fn224 is decode_states175 else 2600])
    except Exception as e:
        print(fn224.__name__, "source unavailable:", str(e)[:150])
        print("names:", fn224.__code__.co_names)
        print("constants:", [v for v in fn224.__code__.co_consts
                             if isinstance(v, (str, int, float))][:35])

print("\n=== ALIGNED TRAINING META ===")
print(train_meta162.columns.tolist())
print(train_meta162.head(3).to_string(index=False))

print("\n=== STATE ARCHIVES ===")
for p in sorted(Path("/work/fwd_22").glob("*.npz")):
    a = np.load(p)
    print(p.name, {k: a[k].shape for k in a.files})
    if "x" in a.files:
        print("first stay/hour", a["stay_id"][:3], a["hour"][:3])

print("\n=== DECODER DEPENDENCIES ===")
dep224 = set(decode_states175.__code__.co_names) | set(build_static167.__code__.co_names)
for k in sorted(dep224):
    if k not in globals():
        continue
    v = globals()[k]
    if hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:10])
    elif callable(v):
        try:
            print(k, "function", inspect.signature(v))
        except Exception:
            pass

print("\n=== PAIRED FACTUAL DRIFT ===")
for run224 in (24, 26):
    o = pd.read_csv(f"/work/fwd_{run224}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{run224}/factual_edited.csv")
    q = o[["stay_id", "hour", "event", "observed"]].copy()
    q["dmean"] = e["predicted_mean"] - o["predicted_mean"]
    q["dnll"] = e["nll"] - o["nll"]
    print(run224, q.groupby("event")[["dmean", "dnll"]].agg(["mean", "std"]).round(5).to_string())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 224: inspect guarded closed-form head correction feasibility\n"
            "Current insulin edit is rank one. Considering a least-squares output-layer residual "
            "correction from paired original/baseline factual states, with directional guards to "
            "preserve causal responses. Inspecting decoder and row alignment before any fit. "
            "No weights changed. Python runs: 224; forwards: 26.\n")
```
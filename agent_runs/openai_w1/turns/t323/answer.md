The posterior reconstruction failed badly, so these features are not safe to fit against. No weights changed. I’ll isolate whether the error comes from the state/context calculation or cached event inputs. Agreement with the verified local replay would confirm a fix; otherwise this injection route remains unvalidated.

```python
print("=== POSTERIOR ERROR CHECKS ===")
zorig283 = torch.as_tensor(ds_s["x"][:, 960:992], dtype=torch.float32)
for label, pred, truth in (
    ("cached original posterior", muorig277, zorig283),
    ("fresh original posterior",
     mlp277(inp277, "z_post", O277)[:, :32], zorig283),
    ("current posterior", mu282,
     torch.from_numpy(S280["x"][:, 960:992]))
):
    err = torch.as_tensor(pred) - truth
    print(label, "RMS", float(torch.sqrt(torch.mean(err**2))),
          "max", float(err.abs().max()))

print("\n=== WHICH EXPORT MATCHES THE CACHED INCOMING STATE? ===")
g283 = dict(globals())
seen283 = set()
for n, o in sorted(g283.items()):
    if not (isinstance(o, dict) and {"x", "stay_id", "hour"}.issubset(o)):
        continue
    if id(o) in seen283 or np.shape(o["x"]) != (5215, 992):
        continue
    seen283.add(id(o))
    if not np.array_equal(o["stay_id"], S280["stay_id"]):
        continue
    hh = torch.as_tensor(
        o["x"][ix281 - 1, 192:960].reshape(-1, 4, 192),
        dtype=torch.float32
    ) * decay281
    print(n, "flow match RMS",
          float(torch.sqrt(torch.mean((hh - hflow277[ix281])**2))),
          "exported z versus cached original mu RMS",
          float(np.sqrt(np.mean(
              (o["x"][:, 960:992] - np276(muorig277))**2
          ))))

print("\n=== VERIFIED LOCAL REPLAY INVENTORY ===")
for n in sorted(g283):
    if not (n.startswith("rx_") or n in
            ("rr_test_e111", "rr_test_u111", "pick276",
             "backupTokens270", "backupH270", "backupE270")):
        continue
    o = g283[n]
    if inspect.isfunction(o):
        c = o.__code__
        print(n, "args", c.co_varnames[:c.co_argcount],
              "constants", [x for x in c.co_consts
                            if isinstance(x, (str, int, float))][:14])
    elif hasattr(o, "shape"):
        print(n, tuple(o.shape))
    elif isinstance(o, dict):
        print(n, "dict", len(o), "keys", list(o)[:5])
    elif isinstance(o, (list, tuple)):
        print(n, type(o).__name__, len(o))

print("\n=== REPLAY COMPONENT COMPARISONS ===")
targets283 = {
    "context": np276(ctx277)[pick276],
    "observation": np276(e277)[pick276],
    "action": np276(u277)[pick276],
    "posterior": np276(muorig277)[pick276],
    "raw_tokens": np276(raw277)[pick276],
    "flowed_h": np276(hflow277)[pick276],
}
for n, o in sorted(g283.items()):
    if not (n.startswith("rx_") or n in
            ("check278", "backupTokens270", "backupH270")):
        continue
    if not isinstance(o, (np.ndarray, torch.Tensor)):
        continue
    a = np276(o)
    matches = [
        (k, float(np.sqrt(np.mean((a - b)**2))))
        for k, b in targets283.items() if a.shape == b.shape
    ]
    if matches:
        print(n, matches)

print("\n=== CONTEXT-RELATED CACHED OBJECTS ===")
for n in sorted(g283):
    if re.search(r"ctx|context|gap|out_prev|postinp|posterior_input", n, re.I):
        o = g283[n]
        if isinstance(o, (np.ndarray, torch.Tensor)):
            print(n, tuple(o.shape))
        elif inspect.isfunction(o):
            c = o.__code__
            print(n, "args", c.co_varnames[:c.co_argcount],
                  "uses", c.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 283\nRun282 posterior reconstruction failed (RMS1.12, max15.22); correctly rejected before fitting or editing. Auditing original-versus-current posterior, source-export flow alignment and verified local replay components to locate the error. Safe236 unchanged. Counts: 283 Python runs, 30 forwards.\n")
```

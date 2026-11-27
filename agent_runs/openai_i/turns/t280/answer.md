The earlier direct-head fallback passed the factual check but barely improved dose responses. The private latent channel preserved factual fit when it carried no response, so it remains a possible foundation. I’ll check how its innovations and replay inputs are constructed. If zero innovations still produce a large state shift, the routing itself—not the estimated drug effect—is the problem.

```python
print("=== LATENT-CHANNEL HISTORY ===")
for s in sections246:
    h = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"\b(?:208|209|210|211|212|213|214|215|216)\b", h):
        print(s[:1600])

print("\n=== CORE HELPER CONTRACTS ===")
for nm in ("private_posterior215", "replay_action_token216", "candidate_readout216",
           "replay_route228", "route_tokens229", "route_features229", "response_candidate231",
           "readout214", "means229", "replay_inputs161"):
    fn = globals().get(nm)
    if inspect.isfunction(fn):
        print(nm, inspect.signature(fn), "globals", fn.__code__.co_names,
              "locals", fn.__code__.co_varnames)
        try:
            src = inspect.getsource(fn)
            print(src[:1800])
        except (OSError, TypeError):
            pass

print("\n=== SMALL OBJECT METADATA ===")
for nm in ("indices214", "indices215", "indices216", "indices228",
           "private_indices196", "reserved196", "R193", "innovation237",
           "directions215", "mean_directions216", "directions231", "causal_targets215",
           "candidates216", "neutral209", "route222", "response_fd230"):
    if nm not in globals():
        continue
    v = globals()[nm]
    if isinstance(v, dict):
        print(nm, {str(k): (tuple(x.shape) if hasattr(x, "shape") else str(x)[:90])
                    for k, x in v.items()})
    elif hasattr(v, "shape"):
        print(nm, tuple(v.shape))
        if np.asarray(v).ndim == 2 and np.asarray(v).shape[1] == 2:
            a = np.asarray(v)
            print("ranges", np.min(a, axis=0), np.max(a, axis=0),
                  "RMS", np.sqrt(np.mean(a*a, axis=0)))

print("\n=== AVAILABLE SOURCE STRINGS ===")
print([(nm, len(v)) for nm, v in list(globals().items())
       if isinstance(v, str) and len(v) > 1000 and nm not in ("arch246", "notes246", "src", "s")][:35])

print("\n=== FORWARD 18–19 COMPACT ===")
for k in (18,19):
    p = Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        j = json.loads(p.read_text())
        print(k, j.get("type"), {
            m: r.get("value_nll_per_value", r.get("mean_shifts", r))
            for m,r in j.get("results",{}).items()
        })

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 249 — inspect exact private-channel neutrality\n"
            "- Safe head fallback is too weak; neutral209 is the factual-preserving foundation.\n"
            "- Inspecting innovation/token construction and replay contracts to distinguish leakage from drug effect.\n"
            "- No new tensors changed.\n")
```
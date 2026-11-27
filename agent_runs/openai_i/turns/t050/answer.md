The flags confirm that endings are not action atoms, but their observation encoding is still unresolved. I’ll inspect the embedding helper and compare time-context conventions rather than guess. A successful reconstruction should have tiny, unsystematic errors—not errors tied to drug or bundle type.

```python
import dis
import io

print("=== EMBEDDING HELPER ===")
buf43 = io.StringIO()
dis.dis(repair_atom_base, file=buf43)
print(buf43.getvalue()[:4800])

print("\n=== START CONTEXT CHECK ===")
context_checks43 = []
prev43 = torch.from_numpy(latent_state[idx41-1, :960])
raw_dt43 = (latent_hour[idx41] - latent_hour[idx41-1]) * 3600
for rounded43, flow43, log1p43 in itertools.product([False, True], [False, True], [False, True]):
    dt43 = torch.from_numpy(
        (np.round(raw_dt43) if rounded43 else raw_dt43).astype(np.float32))
    hh43 = prev43[:, 192:].reshape(-1, 4, 192)
    if flow43:
        hh43 = hh43 * torch.exp(
            -dt43[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :])
    gap43 = torch.log1p(dt43[:, None]) if log1p43 else torch.log(dt43[:, None])
    cc43 = (prev43[:, :192] + repair_norm(hh43.mean(1), "core.out_norm")
            + repair_mlp_act(gap43/10, gap_prefix41, F.gelu))
    zz43 = latent_post_mean(cc43, latent_empty_e.expand(len(idx41), -1), u41)
    rms43 = float(((zz43-torch.from_numpy(latent_state[idx41, 960:]))**2).mean().sqrt())
    context_checks43.append((rms43, rounded43, flow43, log1p43))
print(sorted(context_checks43))

print("\n=== RETAINED EXACT-SEARCH INPUTS ===")
for name43 in ["cases", "contexts", "base_ctx", "additives", "dt"]:
    oo43 = globals().get(name43)
    print(name43, type(oo43).__name__,
          tuple(oo43.shape) if hasattr(oo43, "shape") else "")
    if isinstance(oo43, dict):
        for kk43, vv43 in list(oo43.items())[:4]:
            print(" ", kk43, type(vv43).__name__,
                  tuple(vv43.shape) if hasattr(vv43, "shape") else
                  (list(vv43)[:8] if isinstance(vv43, dict) else str(vv43)[:180]))
    elif isinstance(oo43, list):
        print("first:", repr(oo43[:1])[:220])

print("\n=== ABSENT BIN EMBEDDING NORMS ===")
for attr43 in ["value", "dose", "rate"]:
    print(attr43, float(repair_weights[f"embed.bins.{attr43}.weight"][0].norm()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 43; forward 6 — reconstruction diagnostics\n"
            "- END flags are false for both drugs, supporting a non-action pathway.\n"
            f"- Start context checks: {sorted(context_checks43)[:3]}.\n"
            "- No weights changed.\n")
```
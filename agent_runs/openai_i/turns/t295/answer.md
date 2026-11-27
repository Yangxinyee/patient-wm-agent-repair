The saved candidates confirm that the neutral edit and response-strengthening edits are separate. Neither strengthening route has yet passed factual validation. I’ll recover the latent-channel construction, replay contracts, and estimated persistence now. This will tell us whether the remaining problem is response strength, transport leakage, or both.

```python
print("=== KEY NOTE SECTIONS ===")
budget261 = 3300
for sec in notes259.split("\n## "):
    title = sec.split("\n", 1)[0]
    if (re.search(r"\b(69|70|71|72|73|74|205|206|207|208|209|210|211|212|213|214)\b", title)
        or ("persistence" in sec.lower() and re.search(r"\b([7-9]|1[0-9])\b", title))):
        text = sec[:min(750, budget261)]
        print(text)
        budget261 -= len(text)
        if budget261 <= 0:
            break
print("=== EXACT HELPER CONTRACTS ===")
for name261 in ("linear136", "private_posterior215", "replay_token214",
                "replay_action_token216", "candidate_readout216",
                "get_logits252", "readout_saved252", "install169"):
    ff261 = globals()[name261]
    print(name261, "args", ff261.__code__.co_varnames[:ff261.__code__.co_argcount],
          "names", ff261.__code__.co_names, "constants", str(ff261.__code__.co_consts)[:230])
print("=== LATEST SUMMARIES ===")
for i261 in (17, 18, 19, 20, 21, 22, 23, 24):
    p261 = Path(f"/work/fwd_{i261}/summary.json")
    if p261.exists():
        ss261 = json.loads(p261.read_text())
        print(i261, str(ss261)[:440])
print("=== PRIVATE / DIRECTION ARRAYS ===")
for n261 in ("private_indices209", "private_dims196", "inactive209",
             "orig_hidden209", "private_z209", "orig_token209",
             "mean_directions216", "vectors216", "directions231", "causal_targets215"):
    if n261 in globals():
        vv261 = globals()[n261]
        print(n261, type(vv261).__name__,
              list(vv261.keys()) if isinstance(vv261,dict) else
              (tuple(vv261.shape), str(vv261.dtype)) if hasattr(vv261,"shape") else str(vv261)[:200])
```

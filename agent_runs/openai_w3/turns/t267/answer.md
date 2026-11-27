The actual transport is close to the installed two-direction basis, so a simple scale error is unlikely. The edits are nevertheless large, leaving nonlinear recurrence and compensation as plausible causes. I’ll recover the payload and compensation decomposition so I can test controlled changes rather than alter weights blindly.

```python
print("REPLAY HELPER DETAILS")
for name239 in ["latent206", "raw_hidden228", "preludeE185", "prelude155"]:
    f239 = globals()[name239]
    print("\n", name239, inspect.signature(f239))
    for ins239 in dis.get_instructions(f239):
        if ins239.opname in ["RESUME", "CACHE", "NOP", "PRECALL"]:
            continue
        print(f"{ins239.offset}:{ins239.opname} {ins239.argrepr}", end="; ")
    print()

print("\nPAYLOAD METADATA")
for name239 in ["payload_kv171", "payload154", "decoder192",
                "directions171", "directions186", "directions187",
                "pilot141_gains", "gain140", "gains", "repair_decoder_raw",
                "cols199", "rows199"]:
    if name239 not in globals():
        continue
    v239 = globals()[name239]
    if isinstance(v239,dict):
        print(name239, list(v239)[:15])
    else:
        a239 = as_numpy(v239) if isinstance(v239,torch.Tensor) else np.asarray(v239)
        print(name239, a239.shape,
              a239.tolist() if a239.size<10 else
              ("norm="+str(round(float(np.linalg.norm(a239)),3))))

print("\nATTENTION OUTPUT EDIT COLUMN STRUCTURE")
ow239 = as_numpy(load("core.prelude.0.attn.out.weight"))
oo239 = as_numpy(load_original("core.prelude.0.attn.out.weight"))
for j239 in range(0,192,32):
    print(j239, "change", np.linalg.norm((ow239-oo239)[:,j239:j239+32]),
          "current", np.linalg.norm(ow239[:,j239:j239+32]),
          "outside dose basis",
          np.linalg.norm(ow239[:,j239:j239+32] -
              np.asarray(dd193) @ np.linalg.lstsq(
                  np.asarray(dd193), ow239[:,j239:j239+32], rcond=None)[0]))

n239 = Path("/work/notes.md").read_text()
print("\nIMPLEMENTATION NOTES")
selected239 = [s for s in n239.splitlines()
               if any(t in s.lower() for t in
                      ["reserves", "reserved", "payload", "compensation",
                       "contrast", "gate neuron"])
               and "Python run 2" not in s]
print("\n".join(selected239)[-2800:])
with open("/work/notes.md","a") as f:
    f.write("\nPython run 239; forwards 16. Actual transport aligns with installed dd193 basis; large edits and nonlinear recurrence/compensation remain suspects. Recovering payload decomposition for reversible gain/regularized-direction experiments. No weights changed.\n")
```
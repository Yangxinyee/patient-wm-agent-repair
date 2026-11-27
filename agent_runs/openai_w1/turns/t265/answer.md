The comparison isolates a substantial change in `z_in.weight` as well as the marker attention weights. I’ll test the earlier carrier with the validated relay, without the new markers. If that combination preserves factual fit, the marker construction is the problem; if it fails too, I’ll stop using this latent-token route.

```python
for n in list(edited()):
    drop_edit(n)
for n, v in candidate211_weights.items():
    save_edit(n, torch.as_tensor(arr231(v), dtype=torch.float32))
print("Installed earlier carrier + relay, without dedicated markers.")
print("\nVALUE HEAD TENSORS")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value_bin", "core.b_in", "core.c_out", "core.delta_in"]):
        print(n, SHAPES[n])
print("\nHEAD DECODER EVIDENCE")
for k in ["ce_decoder_results", "decode_results", "repair_post_results"]:
    if k in globals():
        print(k)
        for x in globals()[k][:8]:
            print(str(x)[:280])
print("\nHELPER DEPENDENCIES")
for k in ["ce_make_heads", "transition214", "finite_readout221", "post208"]:
    fn = globals()[k]
    print(k, "names:", fn.__code__.co_names)
    print("constants:", [str(x)[:160] for x in fn.__code__.co_consts
                        if isinstance(x, (str, int, float))][:18])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 234\nInstalled earlier carrier211+relay without dedicated markers "
            "for an isolating factual test. Marker z_in replacement differs substantially "
            "from this candidate. Counts: 234 Python runs, 22 forwards.\n")
```
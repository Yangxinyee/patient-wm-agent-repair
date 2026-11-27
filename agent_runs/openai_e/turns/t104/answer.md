None of the retained linear detectors has a clean norepinephrine signal: stronger insulin guards also weaken the desired response. The factual MAP damage varies substantially across readings, so a constant bias correction is unlikely to fix it. I’ll investigate the posterior network’s nonlinear features, which may separate drug identity and dose more effectively.

```python
import inspect
import numpy as np
import torch

print("POSTERIOR / ACTION TENSORS")
for n89 in NAMES:
    if ("z_post" in n89 or "z_prior" in n89 or
        n89.startswith("encode_action")):
        print(n89, SHAPES[n89])

print("\nEARLIER LOCAL-EDIT FINDINGS")
for s89 in sections85:
    if re.match(r"## Run (?:2[0-9]|3[0-9]|4[0-6])\b", s89):
        if any(w89 in s89.lower() for w89 in
               ["posterior", "latent", "direction", "feature", "recurrent", "prelude"]):
            print(s89[:470].strip())

print("\nPOSTERIOR INPUT INVENTORY")
for k89, v89 in sorted(list(globals().items())):
    if any(s89 in k89.lower() for s89 in
           ["ctx", "context", "post", "_c33", "_e33", "_u34", "_z33",
            "latent", "hidden", "z0", "z_all", "e_all"]):
        if isinstance(v89, (np.ndarray, torch.Tensor)):
            print(k89, type(v89).__name__, tuple(v89.shape))
        elif isinstance(v89, dict):
            print(k89, "dict", list(v89)[:5])

print("\nPOSTERIOR HELPER SIGNATURES")
for name89 in ["recon_ctx", "recon_mlp", "recon_lin", "_post33", "mlp0"]:
    f89 = globals().get(name89)
    if callable(f89):
        print(name89, inspect.signature(f89),
              "vars", f89.__code__.co_varnames,
              "refs", f89.__code__.co_names)

print("\nFRAME SCHEMAS")
for name89 in ["cf_frame", "frame", "state_frame", "bundles", "panel"]:
    v89 = globals().get(name89)
    if isinstance(v89, pd.DataFrame):
        print(name89, v89.shape, list(v89.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 89: consider nonlinear posterior feature route\n"
            "Retained b_in linear detectors have substantial insulin cross-response even with stronger guards. "
            "NE trial factual MAP mean shifts vary by ~1.7, so a constant debias is inadequate. "
            "Inspecting posterior hidden-feature capacity and prior local-edit experiments. "
            "No weights changed; baseline77 remains installed. Python runs: 89; forwards: 12.\n")
```
The recurrent input has much smaller scale than the hidden state, which may explain why earlier injection attempts had little effect. I’ll compare its response to changed doses with its ordinary factual variation. The pathway is promising only if a linear decoder captures dose changes without producing large signals on factual bundles.

```python
print("=== INPUT ALIGNMENT ===")
for n192 in ["context", "state_cal161", "S166", "E166", "U166", "e_cal161",
             "Uref168", "ix166", "hf166", "prev166"]:
    v192 = globals()[n192]
    print(n192, tuple(v192.shape), type(v192).__name__)
print("E alignment RMS:",
      float(torch.as_tensor(E166).sub(torch.as_tensor(e_cal161)).square().mean().sqrt()))
print("U alignment RMS:",
      float(torch.as_tensor(U166).sub(torch.as_tensor(Uref168)).square().mean().sqrt()))

ecore_base192 = torch.empty((n182, 4, 192), dtype=torch.float32)
hflow_base192 = torch.empty_like(ecore_base192)
def profile192(frame, event, arg):
    if event == "return" and frame.f_code is core_step136.__code__:
        parent = frame.f_back
        while parent is not None and parent.f_code is not batch_replay182.__code__:
            parent = parent.f_back
        if parent is not None:
            jj = parent.f_locals["jj"]
            ecore_base192[jj] = frame.f_locals["E"]
            hflow_base192[jj] = frame.f_locals["h"]
previous_profile192 = sys.getprofile()
try:
    sys.setprofile(profile192)
    check_h192, check_o192 = batch_replay182(Uref168)
finally:
    sys.setprofile(previous_profile192)

tokens192 = torch.cat([
    hflow_base192 + W["core.slots"],
    torch.as_tensor(e_cal161)[ids182].unsqueeze(1),
    torch.as_tensor(Uref168)[ids182].unsqueeze(1),
    linear136(oldpost182, "z_in").unsqueeze(1)
], dim=1)
independent_E192 = norm136(
    block136(tokens192, "core.prelude.0")[:, :4], "core.prelude_norm")
print("\nIndependent prelude RMS error:",
      float((independent_E192 - ecore_base192).square().mean().sqrt()))

map192 = {int(i): j for j, i in enumerate(ix166)}
shared192 = [j for j, i in enumerate(ids182) if int(i) in map192]
loc192 = [map192[int(ids182[j])] for j in shared192]
print("hflow alignment RMS:",
      float((torch.as_tensor(hf166)[loc192] - hflow_base192[shared192]).square().mean().sqrt()))
print("\n=== POLICY MODELS ===")
for k192, v192 in policy168.items():
    print(k192, type(v192).__name__)
    if isinstance(v192, dict):
        for kk192, vv192 in v192.items():
            print(" ", kk192, type(vv192).__name__,
                  tuple(vv192.shape) if hasattr(vv192, "shape") else str(vv192)[:220])
    else:
        print(str(v192)[:450])
print("\n=== RATE/TARGET OBJECTS ===")
for n192, v192 in list(globals().items()):
    if any(x in n192.lower() for x in ["rates165", "rates166", "rates168", "dose165", "dose166", "rate_targets", "dose_targets"]):
        print(n192, tuple(v192.shape) if hasattr(v192, "shape") else str(v192)[:180])
print("presence counts:", np.asarray(presence168).sum(axis=0))
with open("/work/notes.md", "a") as f192:
    f192.write("\n## Python 192 — validate independent prelude features\n"
               "- Comparing captured E with a standalone prelude computation and checking full-data input alignment.\n"
               "- No weight changes; rate-decoder neutrality and held-stay dose response will decide viability.\n")
```
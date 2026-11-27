The markers provide two dose-sensitive scalars, but their current output directions have not produced useful rollouts. I’ll measure how each attention-output coordinate changes the next state using symmetric finite differences, without gradients, and recover the value-head calculation. This will support the relay hypothesis if stable directions can turn those scalars into glucose and MAP changes.

```python
import numpy as np
import torch
import time

torch.set_num_threads(8)
start220 = time.time()
base220 = transition214(act="gelu")
hbase220 = base220[1].detach()
obase220 = base220[2].detach()
vbase220 = obase220 + rx_norm(hbase220.mean(dim=1), "core.out_norm")

eps220 = 0.08
jac_h220 = np.empty((48, 4, 192, 192), dtype=np.float32)
jac_o220 = np.empty((48, 192, 192), dtype=np.float32)
jac_v220 = np.empty((48, 192, 192), dtype=np.float32)

with torch.no_grad():
    for jj220 in range(192):
        direction220 = np.zeros(192, dtype=np.float32)
        direction220[jj220] = eps220
        plus220 = transition214(extra=direction220, act="gelu")
        minus220 = transition214(extra=-direction220, act="gelu")
        hp220, op220 = plus220[1], plus220[2]
        hm220, om220 = minus220[1], minus220[2]
        vp220 = op220 + rx_norm(hp220.mean(dim=1), "core.out_norm")
        vm220 = om220 + rx_norm(hm220.mean(dim=1), "core.out_norm")
        jac_h220[..., jj220] = ((hp220 - hm220) / (2 * eps220)).numpy()
        jac_o220[..., jj220] = ((op220 - om220) / (2 * eps220)).numpy()
        jac_v220[..., jj220] = ((vp220 - vm220) / (2 * eps220)).numpy()

print("=== FINITE-DIFFERENCE STATE TRANSMISSION ===")
print("seconds:", round(time.time() - start220, 2))
print("baseline output RMSE versus exported:",
      float(torch.sqrt(torch.mean((obase220 - rx_targetout) ** 2))))
print("output Jacobian RMS:", float(np.sqrt(np.mean(jac_o220 ** 2))))
print("view Jacobian RMS:", float(np.sqrt(np.mean(jac_v220 ** 2))))
sv220 = np.linalg.svd(jac_v220.mean(axis=0), compute_uv=False)
print("mean-view Jacobian singular values:", sv220[:12])

print("\n=== VALUE-HEAD HELPER DETAILS ===")
print("ce_make_heads globals:", ce_make_heads.__code__.co_names)
print("ce_make_heads constants:",
      [v for v in ce_make_heads.__code__.co_consts
       if isinstance(v, (str, int, float))])
print("ce_head keys:", list(ce_head))
print("head row columns:", list(ce_head["rows"].columns))
print(ce_head["rows"].head(2).to_string(index=False))
print("rx_pick range:", int(np.min(rx_pick)), int(np.max(rx_pick)))
print("ce_ix range:", int(np.min(ce_ix)), int(np.max(ce_ix)))
for nn220 in NAMES:
    if any(nn220.startswith(p) for p in [
        "z_prior", "z_head", "query_norm", "type_emb",
        "attr_heads.value_bin", "core.out_norm"
    ]):
        print(nn220, SHAPES[nn220])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 220\n"
            "Measured symmetric finite-difference Jacobians from prelude "
            "attention output to h, out, and world view on 48 cached bundles. "
            "No gradients or optimizer used; no weights changed. Recovering "
            "value-head interface to select sign-correct marker output "
            "directions. Counts: 220 Python runs, 20 forwards.\n")
```
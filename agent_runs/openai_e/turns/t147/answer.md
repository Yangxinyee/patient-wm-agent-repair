The corrected decoder is closer to GPU predictions, especially for arterial MAP, but reconstructed recurrent states still produce substantial glucose errors. I’ll verify state alignment and recover the corrected dose-contrast features before fitting anything. Correct alignment would let us use local *changes* cautiously, with GPU tests still deciding whether an edit is acceptable.

```python
from pathlib import Path
import numpy as np, torch, types, re

def arr127(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== STATE ALIGNMENT ===")
ss127 = np.asarray(src_all121)
for nm127 in ("core_state", "state_array46"):
    xx127 = arr127(globals()[nm127])[ss127]
    yy127 = arr127(state_x)
    print(nm127,
          "out RMSE", float(np.sqrt(np.mean((xx127[:, :192]-yy127[:, :192])**2))),
          "h RMSE", float(np.sqrt(np.mean((xx127[:, 192:960]-yy127[:, 192:960])**2))))
print("head_meta / source indices max difference:",
      int(np.max(np.abs(head_meta["state_row"].to_numpy()-ss127))))
print("reconstructed h / GPU h RMSE:",
      float(np.sqrt(np.mean((arr127(hlocal121)-
                             arr127(core_state)[:, 192:960].reshape(-1,4,192))**2))))

base127 = np.load("/work/baseline77.npz")["delta"].astype(np.float32)
trial127 = np.load("/work/ne_trial103.npz")
ne127 = trial127["delta"].astype(np.float32)-base127
cur127 = arr127(load("core.b_in.weight"))-arr127(load_original("core.b_in.weight"))
print("\n=== EDIT COMPOSITION ===")
print("norms: baseline", float(np.linalg.norm(base127)),
      "full NE", float(np.linalg.norm(ne127)),
      "current", float(np.linalg.norm(cur127)))
print("current minus baseline+.35 NE norm:",
      float(np.linalg.norm(cur127-base127-.35*ne127)))
print("trial103 keys:", trial127.files)

print("\n=== DOSE-CONTRAST ARRAYS ===")
for nm127, ob127 in list(globals().items()):
    if isinstance(ob127, dict) and not nm127.startswith("_"):
        found127 = []
        for k127, v127 in list(ob127.items()):
            if isinstance(v127, (np.ndarray, torch.Tensor)):
                if v127.ndim == 3 and tuple(v127.shape[-2:]) == (4,192):
                    found127.append((str(k127), tuple(v127.shape)))
            elif isinstance(v127, dict):
                for kk127, vv127 in list(v127.items()):
                    if isinstance(vv127, (np.ndarray, torch.Tensor)):
                        if vv127.ndim == 3 and tuple(vv127.shape[-2:]) == (4,192):
                            found127.append((str(k127)+"/"+str(kk127), tuple(vv127.shape)))
        if found127:
            print(nm127, found127)
    elif isinstance(ob127, (np.ndarray, torch.Tensor)):
        if (ob127.ndim == 3 and tuple(ob127.shape[-2:]) == (4,192)
                and ob127.shape[0] != len(state_df)):
            print(nm127, tuple(ob127.shape))
print("\n=== FEATURE/RECURRENT HELPERS ===")
for nm127, ob127 in list(globals().items()):
    if isinstance(ob127, types.FunctionType):
        if any(s in nm127.lower() for s in
               ("prelude", "posterior", "features", "bundle", "core_h", "innov")):
            print(nm127, ob127.__code__.co_varnames[:ob127.__code__.co_argcount],
                  "globals:", ob127.__code__.co_names[:15])

print("\n=== RECORD-DERIVED CALIBRATION ===")
sections127 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for s127 in sections127:
    if re.match(r"^## Run (75|76|77)\b", s127):
        print(s127[:1700].strip())
ins127 = [s for s in sections127
          if re.match(r"^## Run (5[0-9]|6[0-9]|7[0-2])\b", s)
          and any(w in s.lower() for w in
                  ("insulin target", "insulin-only", "accepted", "forward10"))]
for s127 in ins127[-3:]:
    print(s127[:700].strip())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 127: audit local approximation and recover contrasts\n"
            "Corrected original-state decoder RMSE=3.00 overall, 1.57 arterial; reconstructed-state "
            "glucose errors remain large. Auditing alignment and recovering exact corrected prelude "
            "contrasts before fitting output-constrained NE correction. Current weights unchanged. "
            "Python runs: 127; forwards: 15.\n")
```
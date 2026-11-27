The recovered record estimates show next-hour effects near **−2.05 per insulin-rate unit** and **+3.14 per norepinephrine-rate unit**, conditional on the identification assumptions. The edited model still misses the norepinephrine response. I’ll recover the nonlinear estimates and separate channel directions. Distinct, usable drug channels would support a targeted repair; poor channel separation would mean the encoding itself needs changing.

```python
import re, inspect, pathlib, numpy as np, torch

notes170 = pathlib.Path("/work/notes.md").read_text()
sections170 = re.split(r"(?m)(?=^## )", notes170)
wanted170 = {103, 111, 112, 116, 117, 123, 125, 128, 137}
print("RELEVANT CONSTRUCTION NOTES")
budget170 = 4300
for section170 in sections170:
    heading170 = section170.splitlines()[0] if section170 else ""
    nums170 = {int(x) for x in re.findall(r"\b\d+\b", heading170)}
    if nums170 & wanted170 and budget170 > 0:
        text170 = section170[:min(600, budget170)]
        print(text170)
        budget170 -= len(text170)

def brief170(x, depth=0):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 16 else {"shape": list(x.shape)}
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": list(x.keys())[:20]}
        return {str(k): brief170(v, depth + 1) for k, v in list(x.items())[:20]}
    if isinstance(x, (list, tuple)):
        return [brief170(v, depth + 1) for v in x[:8]]
    if isinstance(x, pd.DataFrame):
        return {"shape": list(x.shape), "columns": x.columns.tolist(),
                "head": x.head(4).to_dict("records")}
    if np.isscalar(x):
        return x.item() if isinstance(x, np.generic) else x
    return {"type": type(x).__name__, "shape": getattr(x, "shape", None)}

print("\nRETAINED NUMERIC/CHANNEL OBJECTS")
budget170 = 3800
for name170 in [
    "kernels137", "readout_estimates137", "quadratic137",
    "chosen_channel116", "channel_fits112", "channel_direction112",
    "connection128", "channel_q160", "channel_x160"
]:
    if name170 in globals() and budget170 > 0:
        text170 = name170 + ": " + str(brief170(globals()[name170]))
        text170 = text170[:min(1100, budget170)]
        print(text170)
        budget170 -= len(text170)

print("\nREPLAY DEPENDENCIES")
for name170 in ["replay_factual125", "capture_ep112", "capture_transition102"]:
    fn170 = globals().get(name170)
    if inspect.isfunction(fn170):
        print(name170, "globals:", fn170.__code__.co_names,
              "arguments:", fn170.__code__.co_varnames[:fn170.__code__.co_argcount])

with open("/work/notes.md", "a") as log170:
    log170.write(
        "\n\n## Run 170\n"
        "Recovered immediate record estimates from independent run-136 nuisance fits: "
        "insulin -2.0468 (SE .0483), NE +3.1429 (SE .0960), with close fold agreement. "
        "These require innovation exogeneity, not medical assumptions. "
        "Inspecting nonlinear identification and separate channel construction before "
        "a targeted NE/asymmetry repair. No weights changed.\n"
    )
```

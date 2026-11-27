All edits remain reverted. I’ll now check whether changing a drug’s action encoding produces a usable change at the next reading’s value head, using paired local calculations anchored to exposed factual states. A stable, drug-specific signal would support a constrained head edit; negligible or inconsistent responses would refute it.

```python
import numpy as np, torch
torch.set_num_threads(8)

eligible174 = np.flatnonzero(
    (hours156 >= 6) & (hours156 < 23) &
    (rates_cal161[:, 0] > 0) & (rates_cal161[:, 1] > 0)
)
if not len(eligible174):
    eligible174 = np.flatnonzero((hours156 >= 6) & (rates_cal161[:, 0] > 0))
i174 = int(eligible174[len(eligible174)//2])
stay174 = int(stays156[i174])
hour174 = float(np.floor(hours156[i174]) + 1)
h174 = torch.tensor(state_cal161[i174, 192:960].reshape(1, 4, 192))
out174 = torch.tensor(state_cal161[i174, :192].reshape(1, 192))
print("Probe:", i174, stay174, hours156[i174], "next hour", hour174,
      "rates", rates_cal161[i174])

captured174 = {}
for event174 in ["glucose_finger", "map_arterial"]:
    cap174 = capture159(h174, out174, stay174, hour174, event174)
    captured174[event174] = cap174
    print("\nCapture", event174, type(cap174).__name__)
    if isinstance(cap174, dict):
        for key174, val174 in cap174.items():
            if hasattr(val174, "shape"):
                print(key174, tuple(val174.shape))
            elif isinstance(val174, (str, int, float, np.number)):
                print(key174, val174)
    elif isinstance(cap174, (tuple, list)):
        for j174, val174 in enumerate(cap174):
            print(j174, type(val174).__name__,
                  tuple(val174.shape) if hasattr(val174, "shape") else repr(val174)[:250])

args174 = replay_inputs161(i174, include_current=True, age_scale=3600.0)
print("\nReplay argument shapes:",
      [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__ for x in args174])
local174 = call_core161(args174, "input")
print("Replay return:", type(local174).__name__)
if isinstance(local174, (tuple, list)):
    print("Return shapes:", [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
                            for x in local174])

print("\nPosterior original RMS:",
      float(np.sqrt(np.mean(state_cal161[:, 960:992]**2))))
print("Active edits:", edited())
```
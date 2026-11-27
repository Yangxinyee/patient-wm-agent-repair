The insulin rollout is now close to the provisional target: −14.40/+6.60, with factual loss still comfortably within budget. Norepinephrine remains wrong. I’ll preserve this insulin candidate and inspect where the previous norepinephrine edit incurred factual loss before deciding how much to strengthen it.

```python
import pandas as pd
from scipy.special import ndtr

r165_insulin_backup = {
    name: np.array(r157_np(load(name)), dtype=np.float32, copy=True)
    for name in r164_candidate
}
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 20–21 / Run 165: guarded insulin validated provisionally\n"
        "On tuning 300..399, factual NLL delta approximately +0.00141. "
        "On tuning 0..99, dose shifts: insulin x2 -14.397, half +6.597; "
        "NE x2 -0.538, half -0.599. Record-derived targets on that cohort "
        "remain provisional: -13.599/+6.800 and +4.746/-2.373. "
        "Four-tensor insulin backup retained in r165_insulin_backup. "
        "Investigating NE factual efficiency; no NE edit installed yet.\n"
    )

r165_ne_slots = np.array([48, 171])
r165_ne_W1 = r157_np(r139_candidate["attr_heads.value_bin.0.weight"])
r165_ne_b1 = r157_np(r139_candidate["attr_heads.value_bin.0.bias"])
r165_ne_W2 = r157_np(r139_candidate["attr_heads.value_bin.2.weight"])
r165_ne_gv = (r165_ne_W1[48] + r165_ne_W1[171]) / 2
r165_ne_gb = float((r165_ne_b1[48] + r165_ne_b1[171]) / 2)
r165_ne_dv = (r165_ne_W1[48] - r165_ne_W1[171]) / 2
r165_ne_db = float((r165_ne_b1[48] - r165_ne_b1[171]) / 2)
r165_ne_G = r159_X @ r165_ne_gv + r165_ne_gb
r165_ne_D = r159_X @ r165_ne_dv + r165_ne_db
print("NE paired output consistency:",
      np.max(np.abs(r165_ne_W2[:, 48] + r165_ne_W2[:, 171])))

print("\n=== READING METADATA CACHES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and len(obj) == 4510:
        print(name, list(obj.columns))

print("\n=== NE FIT/VALIDATION SUPPORT ===")
r165_ne_active = np.asarray(r138_active_fit) | np.asarray(r138_active_val)
for label, mask in [
    ("active NE", r165_ne_active),
    ("other readings", ~r165_ne_active)
]:
    print(label, "n=", int(mask.sum()))
    print("D:", np.quantile(r165_ne_D[:, mask], [0, .01, .1, .5, .9, .99, 1]))
    print("G:", np.quantile(r165_ne_G[:, mask], [0, .01, .1, .5, .9, .99, 1]))

print("\n=== PREVIOUS NE FACTUAL LOSS FILE SCHEMAS ===")
for k in [16, 18, 20]:
    frame = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    print(k, frame.shape, list(frame.columns))
    if k == 16:
        print(frame.head(2).to_string(index=False))
```

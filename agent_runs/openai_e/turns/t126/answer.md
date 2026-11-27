The reduced edit moves norepinephrine responses to +0.59 and −1.28, still well short of the record-based targets. Insulin responses remain close to the baseline. I’ll examine the correction’s recurrent-state direction next: a more MAP-selective direction could allow a stronger effect without the glucose prediction losses seen in larger trials.

```python
print("LOCAL HELPER INTERFACES")
for name108 in ("coda46", "next_means45", "prelude46", "head_calc",
                "decoder_query", "core_update_local"):
    fun108 = globals()[name108]
    print(name108, inspect.signature(fun108))
    print("  names", fun108.__code__.co_names)
    print("  constants", str(fun108.__code__.co_consts)[:550])

snap108 = list(globals().items())
print("LOCAL STATE ARRAYS")
for k108, v108 in snap108:
    if isinstance(v108, (np.ndarray, torch.Tensor)) and tuple(v108.shape) in (
        (32,4,192), (32,192), (192,), (31,192)):
        if k108.endswith(("45", "46", "47", "82", "95", "103")):
            print(k108, tuple(v108.shape))
print("HEAD/FEATURE ARRAYS")
printed108 = 0
for k108, v108 in snap108:
    if isinstance(v108, (np.ndarray, torch.Tensor)) and v108.ndim == 2:
        if v108.shape[0] > 500 and v108.shape[1] in (192, 384, 576, 768):
            print(k108, tuple(v108.shape))
            printed108 += 1
            if printed108 >= 22:
                break
print("RECORD/STATE TABLES")
printed108 = 0
for k108, v108 in snap108:
    if isinstance(v108, pd.DataFrame) and len(v108) > 100:
        print(k108, v108.shape, list(v108.columns)[:9])
        printed108 += 1
        if printed108 >= 16:
            break
print("DIRECTION CONSTRUCTION NOTES")
for sec108 in sections105:
    m108 = re.search(r"Run\s+(\d+)", sec108.splitlines()[0])
    if m108 and int(m108.group(1)) in (45, 46, 47, 82, 102):
        print(sec108[:700])
with Path("/work/notes.md").open("a") as f108:
    f108.write("\n## Run 108: reduced NE passes, but insufficient causal correction\n"
               "Forward14 NLL increase=.00402. Forward15 shifts: insulin -12.555/+4.182, "
               "NE +.587/-1.282. Reduced NE improves both signs but is far below "
               "record-derived doubling target. Investigating actual per-slot b_in "
               "perturbations and MAP-selective output directions before scaling. "
               "Python runs: 108; forwards: 15.\n")
```
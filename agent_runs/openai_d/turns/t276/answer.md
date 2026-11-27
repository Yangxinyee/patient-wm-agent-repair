The edit is confined to four hidden neurons and the output bias, so the insulin and norepinephrine corrections can be revised separately. The inspection output was again truncated by a large function default. I’ll retrieve compact cache and parameter descriptions next; they should let me screen replacements without disturbing the saved fallback.

```python
def rc_brief_obj(o):
    if inspect.isfunction(o):
        c = o.__code__
        return "function args=" + repr(c.co_varnames[:c.co_argcount+c.co_kwonlyargcount])
    if hasattr(o, "shape"):
        return type(o).__name__ + " shape=" + repr(tuple(o.shape))
    if isinstance(o, dict):
        return "dict keys=" + repr(list(o.keys())[:16])
    if isinstance(o, (list, tuple)):
        return type(o).__name__ + " len=" + str(len(o))
    return str(o)[:120]

for rc_prefixes, rc_limit in [
    (("r201", "c207", "c210"), 2300),
    (("r230", "r231", "cache231"), 2800)
]:
    rc_text = "\n".join(
        n + ": " + rc_brief_obj(o)
        for n, o in list(globals().items())
        if n.startswith(rc_prefixes)
    )
    print("\n=== " + "/".join(rc_prefixes) + " ===")
    print(rc_text[:rc_limit])

print("\n=== EVALUATOR INTERNALS ===")
for rc_name in ["c207_tensor_weights", "c210_eval", "r201_eval",
                "r201_fixed", "c211_logits"]:
    rc_obj = globals().get(rc_name)
    if inspect.isfunction(rc_obj):
        rc_c = rc_obj.__code__
        print(rc_name, "locals:", rc_c.co_varnames)
        print("names:", rc_c.co_names)
        print("constants:", [
            c for c in rc_c.co_consts
            if isinstance(c, (str, int, float, tuple))
        ][:20])

print("\n=== NE DEVELOPMENT SUMMARY ===")
rc_s = rc_notes.find("## Run 175")
rc_e = rc_notes.find("## Run 212")
rc_part = rc_notes[rc_s:rc_e] if rc_s >= 0 else ""
rc_keep = [
    s for s in rc_part.splitlines()
    if s and not s.startswith((" ", "{", "}", "[", "]"))
]
print("\n".join(rc_keep)[-1800:])

print("\n=== FORWARD CSV COLUMNS ===")
print(pd.read_csv("/work/fwd_30/factual_original.csv", nrows=1).columns.tolist())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 236: compact evaluator recovery\n"
            "Recovered compact parameter/cache descriptions, avoiding tensor-valued "
            "function defaults that truncated Run 235. No weights changed.\n")
```
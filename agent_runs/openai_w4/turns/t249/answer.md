This repeats the earlier output, so it adds no evidence about the source of the asymmetry. The useful finding remains that the dose decoder is not perfectly linear. I’ll retrieve the numerical kernels and channel-construction objects now; disproportionately weak doubled-dose transmission would support the saturation hypothesis, while similar half/double transmission would refute it.

```python
print("=== RECORD KERNELS AND TARGETS ===")
for r216_name in ["provisional_kernels", "kernel_models88", "kernels137",
                 "target_table89", "causal_beta80"]:
    r216_obj = globals().get(r216_name)
    if isinstance(r216_obj, dict):
        print(r216_name, {
            k: np.round(np.asarray(v), 5).tolist()
            if hasattr(v, "shape") and np.asarray(v).size <= 30
            else r215_brief(v)
            for k, v in r216_obj.items()
        })
    elif r216_obj is not None:
        print(r216_name, r216_obj)

print("\n=== EXACT REPLAY DEPENDENCIES ===")
for r216_name in ["replay_factual125", "cpu_factual129",
                 "transition73", "collect_selected187", "gated_innovation94"]:
    r216_fn = globals().get(r216_name)
    if callable(r216_fn) and hasattr(r216_fn, "__code__"):
        print(r216_name, "locals:", r216_fn.__code__.co_varnames,
              "globals:", r216_fn.__code__.co_names)

print("\n=== CHANNEL-CONSTRUCTION VARIABLES ===")
print(sorted(n for n in globals()
             if re.search(r"(186|187|193)$", n)
             and not n.startswith("r216_")))
for r216_name in ["amplitude_tests187", "directions_ne187",
                 "joint_decoders193", "decoders_ins186", "decoders_ne186"]:
    r216_obj = globals().get(r216_name)
    if r216_obj is not None:
        print(r216_name, repr(r215_brief(r216_obj))[:1200])

print("\n=== REPLAY FIXTURE ===")
r216_bank = globals().get("repair_bank", {})
print({k: r215_brief(v) for k, v in r216_bank.items()})
print("\n=== RESPONSE IDENTIFICATION NOTES ===")
for r216_section in r214_sections:
    if re.match(r"## (Run 137\b|Run 138\b|Run 139\b|Run 140\b|Run 154\b|Run 155\b|Run 156\b)", r216_section):
        print(r216_section[:900])

with open("/work/notes.md", "a") as r216_file:
    r216_file.write(
        "\n## Run 216\n"
        "Retrieved numerical longitudinal kernels, channel-construction variables and exact replay dependencies. "
        "Investigating whether doubled-dose saturation explains the half/double asymmetry. "
        "Record-derived targets remain provisional causal estimates. No saved weights changed.\n"
    )
```
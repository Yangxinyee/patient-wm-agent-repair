Previous-bin centering helps only slightly and leaves the combined edit above the loss limit on the difficult cohort. The remaining concern is the dose-sensitive direction: its earlier probe changed a code embedding, not actual rates and bins. I’ll recover the exact action-update machinery; successful reconstruction would let us design the correction using real dose edits.

```python
print("=== NOTEBOOK INDEX ===")
r218_sections = Path("/work/notes.md").read_text().split("\n## ")
print("\n".join(section.split("\n", 1)[0] for section in r218_sections[:80]))

print("=== STATE-PROBE FUNCTIONS ===")
r218_prefixes = ("r70", "r71", "r72", "r73", "r74", "r75", "r76", "r77",
                 "r78", "r79", "r80", "r81", "r82", "r83", "r84", "r85",
                 "r86", "r87", "r88", "r89", "r90", "r91", "r92", "r93",
                 "r94", "r95", "r96", "r97", "r98", "r99", "r124",
                 "r125", "r126", "r127", "r128", "r129", "r176", "r178")
for name, v in list(globals().items()):
    if name.startswith(r218_prefixes):
        if callable(v):
            code = getattr(v, "__code__", None)
            print(name, "args", code.co_varnames[:code.co_argcount] if code else "")
        elif isinstance(v, dict) and any(k in v for k in ("x", "h", "out", "W", "states")):
            print(name, "dict keys", list(v)[:12])
        elif isinstance(v, (np.ndarray, torch.Tensor)) and (
                "probe" in name or "delta" in name or "dir" in name or "mem" in name):
            print(name, tuple(v.shape), str(v.dtype))

print("=== PROBE-CONSTRUCTION NOTES ===")
r218_printed = 0
for section in r218_sections:
    if section.split("\n", 1)[0].startswith(("Run 84", "Run 85", "Run 86",
                                          "Run 87", "Run 127", "Run 128")):
        print(section[:1100])
        r218_printed += 1
        if r218_printed >= 3:
            break
print("=== CONTROLLED-IMPULSE SUMMARY ===")
for section in r218_sections:
    if section.split("\n", 1)[0] == "Run 5":
        print(section[-1700:])
        break
```
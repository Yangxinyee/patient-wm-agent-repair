The replay helpers are still available; the notebook lookup failed because the earlier output limit stopped that run before it reached the assignment. I’ll read the identification summary directly and inspect the state-update helper. This will let me measure whether an action-path correction produces the intended glucose and MAP changes.

```python
print("IDENTIFICATION SUMMARY")
print(Path("/work/notes.md").read_text()[:3800])

print("\nSTATE UPDATE SIGNATURE", inspect.signature(state_update82))
print("STATE UPDATE BYTECODE")
print(compact_bytecode(state_update82)[:4300])

print("\nPRELUDE CACHE RETURNS")
import dis
ins94 = list(dis.get_instructions(prelude_cache88))
for ii94 in ins94[-16:]:
    print(ii94.opname, ii94.argrepr)
print("CACHE VARIABLES", prelude_cache88.__code__.co_varnames)

print("\nCURRENT ARRAY SHAPES")
for n in ("idx81", "h81", "hf81", "e81", "u81", "z81", "ctx81", "E88",
          "hidden88", "tokens88", "pc89", "hc88", "memory82",
          "last_values90", "last_hours90", "means90", "c90"):
    if n in globals():
        o = globals()[n]
        print(n, type(o).__name__, getattr(o, "shape", ""),
              list(o)[:5] if isinstance(o, dict) else "")
print("\nDOSE FIT KEYS", list(contrast_fits88), list(pool_fits89))
print("BUNDLE INFO EXAMPLE", r40_info[int(idx81[50])])
print("REGISTER EXAMPLE", next(iter(registers90.values())))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 94; forwards 7. No edits. Recovering compact causal summary "
            "and state-update interface after diagnostic output/name failures.\n")
```
The dynamic estimates suggest both drugs’ effects persist for several hours, so a one-step correction alone may be inadequate. I’ll inspect the retained readout implementation and candidate results, focusing on drug-specific responses and factual preservation. A shared-state candidate is unsuitable if it creates large cross-drug effects or exceeds the factual-loss limit.

```python
import inspect, dis, io
print("=== RELEVANT NOTEBOOK SECTIONS ===")
for s116 in sections115:
    if any(s116.startswith(x) for x in (
        "## Run 34:", "## Run 37:", "## Run 54:", "## Run 66:", "## Run 90:")):
        print(s116[:850])
print("\n=== READOUT SOURCE ===")
sources116 = [(k,v) for k,v in list(globals().items())
              if isinstance(v,str) and "def r49_readouts" in v]
if sources116:
    k116, src116 = min(sources116, key=lambda kv: len(kv[1]))
    print("Source variable:", k116)
    print(src116[:4200])
else:
    try:
        print(inspect.getsource(r49_readouts)[:4200])
    except (OSError, TypeError):
        buf116 = io.StringIO()
        dis.dis(r49_readouts, file=buf116)
        print(buf116.getvalue()[:4200])
print("\n=== INTERFACES ===")
for name116 in ("r49_readouts","r57_features","r59_features","r51_core",
                "repaired_core_local","factual_local72","build_candidate89"):
    f116 = globals().get(name116)
    if callable(f116):
        print(name116, inspect.signature(f116), "names:", f116.__code__.co_names)
print("\n=== RELEVANT TENSOR SHAPES ===")
print([(n,SHAPES[n]) for n in NAMES if
       any(t in n for t in ("last_numeric","last_status","attr_heads.value","z_head","z_post","z_in"))])
print("\n=== CANDIDATE DIAGNOSTICS ===")
for name116 in ("diagnostics90","diagnostics82","diagnostics81","candidates90"):
    v116 = globals().get(name116)
    print(name116, type(v116).__name__,
          (str(v116)[:650] if name116 != "candidates90" else
           list(v116.keys()) if isinstance(v116,dict) else ""))
print("\n=== STORED SOURCE STRINGS ===")
print([(k,len(v)) for k,v in list(globals().items()) if isinstance(v,str) and "def " in v][:35])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 116: targeted interface recovery\n"
            "- Identified effects persist over several hours; repair must preserve dynamics and distinguish drugs.\n"
            "- Recovering readout implementation and retained candidate diagnostics.\n"
            "- No edits. Python runs: 116; forward passes: 4.\n")
```
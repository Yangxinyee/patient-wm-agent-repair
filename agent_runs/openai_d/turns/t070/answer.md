The architecture confirms that readings have no direct access to the drug registers: a repair must carry dose information through the state. The recovered regressions also show substantial confounding. I’ll recover the stronger causal checks and existing helpers next; those checks must support the drug effects before I target them.

```python
notes62 = Path("/work/notes.md").read_text()
start62 = notes62.find("## Run 4\n")
end62 = notes62.find("## Restart / Run 14")
print("=== CAUSAL CHECKS AND EARLY DIAGNOSTICS ===")
print(notes62[start62:end62][:5400])

items62 = list(globals().items())
print("\n=== FUNCTIONS ===")
print([k for k, v in items62
       if callable(v) and getattr(v, "__module__", "") == "__main__"])

print("\n=== DATAFRAMES ===")
print([(k, tuple(v.shape), list(v.columns)[:8])
       for k, v in items62 if isinstance(v, pd.DataFrame)])

print("\n=== ARRAYS / TENSORS ===")
arr62 = [(k, tuple(v.shape)) for k, v in items62
         if isinstance(v, (np.ndarray, torch.Tensor)) and not k.startswith("_")]
print(str(arr62)[:2500])

print("\n=== DICTIONARIES ===")
dict62 = [(k, len(v), list(v)[:5]) for k, v in items62
          if isinstance(v, dict) and not k.startswith("_")]
print(str(dict62)[:1600])
print("\nCurrent edits:", edited())
```
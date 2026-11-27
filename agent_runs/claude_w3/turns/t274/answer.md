The dose encoder and prelude tensors are as expected (pre-norm layer, `kv` stacked as 384×192). The posterior, `z_in` and gap tensors use other names. I'll find them before writing a clean local step.

```python
sel2 = [n for n in NAMES if re.search(r'post|prior|z_|gap|flow|delta|a_in|query|value_bin|last_numeric|transition|code_emb|type', n)]
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in sel2))
```
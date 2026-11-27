The config shows 4 heads, 2 recurrent layers, 1 coda layer and 1 loop. It says nothing about the 8 `last_numeric` inputs. The vocab notes may explain the numeric features, so I'll print those notes and the structure of the numeric table, briefly.

```python
print(VOCAB['numeric_note'][:600] if isinstance(VOCAB['numeric_note'], str) else VOCAB['numeric_note'])
print(VOCAB['value_bin_note'][:300] if isinstance(VOCAB['value_bin_note'], str) else VOCAB['value_bin_note'])
num = VOCAB['numeric']
print(type(num), list(num.keys())[:20] if isinstance(num, dict) else len(num))
```
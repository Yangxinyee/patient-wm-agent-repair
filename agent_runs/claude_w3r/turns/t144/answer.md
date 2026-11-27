The head structure is clear now. The value logits are `attr_heads.value_bin(with_code)` plus a per-(code, last value bin) 17-dimensional logit offset, `embed_transition.value_bin`. That offset is a natural place for the band-level compensation term, because it already knows the last glucose bin. The file is short (5.2k characters), so next I'll print it in full. I want to recall exactly what "view" (the head's query input) is and how z enters before I design the channel. This is a reading step; nothing is tested yet.

```python
print(arch)
```
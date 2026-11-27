The embedding tables include code (13 rows), kind (3), ekind (5), kq (4) and several categorical tables (unit has 6 rows). A measurement atom is therefore code + kind + ekind + unit + kq + value bin + numeric terms. The most likely cause of my old error is a wrong ekind or unit/kq index.

To pin down the indices, I'll print the code table (index, unit and any normalization per code) and the numeric normalization block, kept compact.

```python
print(str(VOCAB['codes'])[:900])
print(str(VOCAB['numeric'])[:600])
print(str(VOCAB.get('numeric_note'))[:300])
```
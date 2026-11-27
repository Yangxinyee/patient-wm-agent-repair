The action token u_k is exactly zero at bundles without a drug, so after `norm1` it becomes the constant `norm1.bias`. Edits inside `encode_action` therefore act only at drug bundles. Next I will read the heads section (lines 54–79), to check whether the value heads see anything about the next bundle's drugs.

```python
print('\n'.join(x[:150] for x in L[54:79]))
```
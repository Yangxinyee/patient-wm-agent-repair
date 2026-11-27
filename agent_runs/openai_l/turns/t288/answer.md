The previous-bin correction did not improve independent tuning NLL, so I’m rejecting it without changing weights. The head’s two drug responses also interact: emphasizing insulin can weaken norepinephrine. I’ll inspect that coupling; a cleaner separation could permit stronger responses at lower factual cost.

```python
print("=== DOSE / GATE GEOMETRY ===")
for name255 in ["directions207", "gatecanon200", "family204",
                "dosecenter214", "rows235", "pruneorder214"]:
    obj255 = globals()[name255]
    print(name255, type(obj255).__name__, getattr(obj255, "shape", ""))
    if isinstance(obj255, dict):
        for k255, v255 in obj255.items():
            print(" ", k255, type(v255).__name__, getattr(v255, "shape", ""),
                  list(v255.keys()) if isinstance(v255, dict) else
                  (repr(v255)[:200] if np.isscalar(v255) else ""))
    elif name255 == "pruneorder214":
        print(obj255[:16])

print("=== PAIRED CONTRAST STRUCTURE ===")
for key255 in [("insulin", 2.0), ("norepinephrine", 2.0)]:
    print(key255)
    for k255, v255 in paired214[key255].items():
        print(" ", k255, type(v255).__name__, getattr(v255, "shape", ""),
              repr(v255)[:130] if np.isscalar(v255) else "")

print("=== RAW TUNING INPUT CHECK ===")
rawaug255 = np.concatenate([raw225, np.ones(raw225.shape[:-1] + (1,))], axis=-1)
Waug255 = np.concatenate([Wbase204, bbase204[:, None]], axis=1)
Hquietcheck255 = gelu214(rawaug255 @ Waug255.T)
print("Quiet hidden reconstruction RMS:",
      float(np.sqrt(np.mean((Hquietcheck255 - Fquiet234[..., :192])**2))))
print("=== GATED ARCHITECTURE CONSTRUCTION ===")
instructions255 = list(dis.get_instructions(make_arch215))
text255 = "\n".join(f"{i.offset:4} {i.opname:24} {i.argrepr[:95]}"
                    for i in instructions255)
print(text255[:6500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 254 / conditional compensation rejected\n"
            "Prior-bin original-logit offset candidates did not improve "
            "independent tuning NLL (best ~+.00399 versus +.00393 current); "
            "none saved. Reproduced .5/.4 analytic final-head solution to "
            "2.8e-12. Investigating coupling of the two dose constraints.\n")
```
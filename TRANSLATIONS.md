# Translations

no-magic scripts teach algorithms through code comments. Translating those comments into other languages makes this resource accessible to engineers worldwide who think more fluently in their native language.

This initiative translates **comments, docstrings, section headers, and print statements** in all 30 scripts. The code itself stays in English — variable names, function names, and logic are universal.

---

## Status

| Language | Locale | Status | Maintainer | Progress |
|----------|--------|--------|------------|----------|
| Spanish | `es` | :red_circle: Not Started | TBD | 0/30 scripts |
| Brazilian Portuguese | `pt-BR` | :red_circle: Not Started | TBD | 0/30 scripts |
| Simplified Chinese | `zh-CN` | :red_circle: Not Started | TBD | 0/30 scripts |
| Japanese | `ja` | :red_circle: Not Started | TBD | 0/30 scripts |
| Korean | `ko` | :red_circle: Not Started | TBD | 0/30 scripts |
| Hindi | `hi` | :red_circle: Not Started | TBD | 0/30 scripts |

---

## Guidelines for Translators

### What to Translate

- File thesis docstrings
- Section headers (`# === SECTION NAME ===`)
- All inline comments (why comments, math-to-code mappings, intuition comments, signpost comments)
- Print statements (training progress, inference output)
- Docstrings on functions

### What NOT to Translate

- Variable names, function names, class names
- Code logic, operators, syntax
- Mathematical notation in comments (equations are universal)
- English technical terms that are standard in the target language's ML community (e.g., "softmax", "embedding", "attention")

### File Naming and Location

Translated scripts live in `translations/<locale>/`. The filename stays the same as the original:

```
translations/es/microgpt.py      # Spanish translation of 01-foundations/microgpt.py
translations/ja/microlora.py     # Japanese translation of 02-alignment/microlora.py
translations/zh-CN/microbeam.py  # Chinese translation of 03-systems/microbeam.py
```

### Quality Bar

1. **Technical accuracy over literary polish.** An incorrect translation is worse than an awkward one.
2. **Preserve all 7 comment types** from the commenting standard (thesis, section headers, why comments, math-to-code mappings, intuition comments, signpost comments, no obvious comments).
3. **Preserve math notation as-is.** Equations are universal.
4. **Use domain-standard terminology** for the target language's ML community.
5. **When in doubt, keep the English term** with a parenthetical translation: `# softmax (normalización exponencial)`.
6. **The translated script must still run.** `python translations/es/microgpt.py` must produce the same training and inference results as the original.

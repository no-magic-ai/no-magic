# Contributing Translations

This directory contains translated versions of no-magic scripts. Only comments, docstrings, section headers, and print statements are translated — the code stays in English.

## Directory Structure

```
translations/
  es/       # Spanish
  pt-BR/    # Brazilian Portuguese
  zh-CN/    # Simplified Chinese
  ja/       # Japanese
  ko/       # Korean
  hi/       # Hindi
```

Each directory mirrors the original scripts by filename. The tier prefix (`01-foundations/`, etc.) is dropped — the flat structure is sufficient since script names are unique.

## How to Contribute

1. Check [TRANSLATIONS.md](../TRANSLATIONS.md) for the current status of each language.
2. Pick a script that hasn't been translated yet for your target language.
3. Copy the original script into the appropriate language directory:
   ```bash
   cp 01-foundations/microgpt.py translations/es/microgpt.py
   ```
4. Translate all comments, docstrings, section headers, and print statements.
5. Do **not** translate variable names, function names, class names, or code.
6. Verify the translated script still runs and produces correct output:
   ```bash
   python translations/es/microgpt.py
   ```
7. Open a PR with the translated file. Reference the target language and script name in the PR title.

## Quality Checklist

- [ ] All 7 comment types preserved (thesis, section headers, why, math-to-code, intuition, signpost, no obvious)
- [ ] Mathematical notation unchanged
- [ ] Domain-standard terminology used for the target language
- [ ] English terms kept (with parenthetical translation) when no established equivalent exists
- [ ] Script runs with `python translations/<locale>/<script>.py` and exits cleanly
- [ ] Output matches the original script (same training loss, same inference results)

---
description: Guidelines for correctly invoking the edit_existing_file tool without errors.
alwaysApply: true
---

When using the edit_existing_file tool, always provide BOTH a valid 'filepath' string (relative to the repository root) and a non‑empty 'changes' string containing the full new file contents (or a valid diff). Ensure the JSON payload is correctly formed (no stray commas, no missing braces). Before editing, read the current file with read_file to obtain the latest content. Do not include any extra explanatory text in the 'changes' field; it must be pure file text. If you need to replace the whole file, set 'changes' to the complete new file content. If you only need a partial update, still supply the entire file content because the tool overwrites the file. Verify that the file exists and that the path is correct.
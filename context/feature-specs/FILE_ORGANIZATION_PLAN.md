# File Organization Plan for GrowthParameterEstimation.jl

## Overview
This plan outlines how to organize stray files in the root directory of the GrowthParameterEstimation.jl repository to improve maintainability and clarity.

## Current State Analysis
The root directory contains numerous files that could be better organized into appropriate subdirectories. Many of these files are test scripts, documentation, or configuration files that would benefit from being grouped logically.

## Proposed Organization

### 1. Documentation Files
Move documentation files to the `/docs` directory:
- `AI Reference.md` → `/docs/AI_Reference.md`
- `CHANGELOG.md` → Keep in root (standard location)
- `MIGRATION_PROGRESS.md` → `/docs/MIGRATION_PROGRESS.md`
- `PLAN_FOR_BUILDABLE_MODELS.md` → `/docs/PLAN_FOR_BUILDABLE_MODELS.md`
- `README.md` → Keep in root (standard location)
- `TESTING_GENIE_APP.md` → `/docs/TESTING_GENIE_APP.md`

### 2. Test Files
Organize test files into the `/test` directory (which already exists):
- `debug_get_param.jl` → `/test/debug_get_param.jl`
- `final_test.jl` → `/test/final_test.jl`
- `simple_test.jl` → `/test/simple_test.jl`
- `test_docstring.jl` → `/test/test_docstring.jl`
- `test_import.jl` → `/test/test_import.jl`
- `test_models.jl` → `/test/test_models.jl`
- `test_new_features.jl` → `/test/test_new_features.jl`
- `test_param.jl` → `/test/test_param.jl`
- `test_simple.jl` → `/test/test_simple.jl`

### 3. Configuration Files
Keep essential configuration files in root, but consider organizing others:
- `ci.yml` → Keep in root (CI configuration)
- `opencode.json` → Keep in root (Opencode configuration)
- `.JuliaFormatter.toml` → Keep in root (formatter configuration)
- `.gitattributes` → Keep in root (Git configuration)
- `.gitignore` → Keep in root (Git configuration)
- `.theme` → Keep in root (theme configuration)
- `Manifest.toml` → Keep in root (Julia standard)
- `Project.toml` → Keep in root (Julia standard)
- `LICENSE` → Keep in root (standard location)

### 4. Assets and Examples
These directories already exist and appear to be correctly placed:
- `/assets` → Keep as is
- `/examples` → Keep as is
- `/log` → Keep as is
- `/results` → Keep as is

### 5. Source Code
The source code is already correctly placed:
- `/src` → Keep as is

## Implementation Steps
1. Create backup of current state
2. Move documentation files to `/docs`
3. Move test files to `/test`
4. Verify all moved files work correctly in their new locations
5. Update any internal references if necessary
6. Commit changes

## Benefits
- Improved directory structure clarity
- Better separation of concerns (docs, tests, source)
- Easier navigation for new contributors
- Standardized layout following common Julia/OpenSource practices

## Files to Remain in Root
After organization, these files should remain in the root directory:
- `.gitattributes`
- `.gitignore`
- `.JuliaFormatter.toml`
- `.theme`
- `CHANGELOG.md`
- `ci.yml`
- `LICENSE`
- `Manifest.toml`
- `Opencode.json`
- `Project.toml`
- `README.md`
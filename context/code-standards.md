# Code Standards

## General

- Keep modules small and single-purpose
- Fix root causes, do not layer workarounds
- Do not mix unrelated concerns in one component
- Write clear, descriptive function and variable names
- Comment complex logic, not obvious code
- Follow Julia's standard library naming conventions

## Julia Specific

- Use LowerCaseWithUnderscores for function names
- Use CamelCase for types and modules
- Use UPPER_CASE for constants
- Prefer explicit typing in function signatures for public API
- Use `const` for global constants
- Avoid global variables when possible
- Use `using` statements at the top of files, not inside functions
- Prefer immutable data structures when appropriate
- Handle errors with explicit return values or exceptions appropriately

## Formatting

- Use JuliaFormatter.jl for code formatting (configured in .JuliaFormatter.toml)
- Maximum line width of 92 characters (as per JuliaFormatter default)
- Use 4 spaces for indentation (standard Julia convention)
- Place related functions together in modules
- Keep function definitions close to their usage when possible

## Documentation

- Document all exported functions with docstrings
- Use triple quotes for multi-line docstrings
- Include examples in docstrings when helpful
- Follow the Julia documentation guidelines (https://docs.julialang.org/en/v1/manual/documentation/)
- Keep README.md updated with major changes

## Testing

- Write tests for new functionality in the test/ directory
- Use @test macros from Julia's Test module
- Test both typical use cases and edge cases
- Ensure tests run successfully in CI environment
- Add tests when fixing bugs to prevent regression

## File Organization

- src/ — Source code organized by concern:
  - data.jl - Data loading, normalization, and validation
  - exposure.jl - Drug exposure modeling
  - models.jl - Growth model definitions
  - registry.jl - Model registration and management
  - simulation.jl - Model simulation capabilities
  - observation.jl - Observation modeling
  - fitting.jl - Parameter estimation and fitting routines
  - analysis.jl - Post-fit analysis and diagnostics
  - workflow.jl - Workflow automation and pipeline management
- test/ — Test files (including all test scripts ending in .jl)
- examples/ — Example scripts and applications
- docs/ — Documentation files (all .md files except README.md and CHANGELOG.md)
- context/ — Project context files (this directory) - Internal project documentation
- results/ — Output from runs (generated, not committed)
- log/ — Log files (generated, not committed)
- assets/ — Static assets (images, data files, etc.)
- Root directory should contain only:
  - Essential configuration files: .gitattributes, .gitignore, .JuliaFormatter.toml, .theme
  - Standard project files: CHANGELOG.md, LICENSE, Manifest.toml, Project.toml, README.md
  - Tool configuration: ci.yml, opencode.json
  - No loose test scripts, documentation files, or temporary files

### Organization Principles
1. **Separation of Concerns**: Each directory has a clear, single purpose
2. **Consistency**: Follow established patterns for file placement
3. **Clarity**: Root directory should be clean and minimal
4. **Automation**: Generated files (results, logs) should be excluded from version control
5. **Discoverability**: Related files should be colocated in logical directories

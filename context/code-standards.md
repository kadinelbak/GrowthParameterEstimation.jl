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
- test/ — Test files
- examples/ — Example scripts and applications
- docs/ — Documentation files
- context/ — Project context files (this directory)
- results/ — Output from runs (generated, not committed)
- logs/ — Log files (generated, not committed)

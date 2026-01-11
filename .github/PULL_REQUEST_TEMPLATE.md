## Description
<!-- Describe your changes in detail -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Performance improvement
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Backend Impact
- [ ] Python backend
- [ ] Numba backend
- [ ] Both backends
- [ ] No backend changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have run tests with both Python and Numba backends (if applicable)
- [ ] I have verified numerical equivalence between backends (if applicable)

## Test Results
<!-- Paste test results here -->

```bash
# Python backend
pytest tests/ -v
# Result: 

# Numba backend (if applicable)
LOOM_BACKEND=numba pytest tests/ -v
# Result: 
```

## Performance Impact
<!-- If applicable, include benchmark results -->

## Related Issues
<!-- Link to related issues: Fixes #123 -->

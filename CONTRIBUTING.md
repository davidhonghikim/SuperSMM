# Contributing to SuperSMM

## Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all contributors.

## How to Contribute

### Reporting Bugs
1. Check existing issues to ensure the bug hasn't been reported
2. Open a new issue with:
   - Clear title
   - Detailed description
   - Reproducible steps
   - Environment details

### Feature Requests
1. Open an issue describing:
   - Proposed feature
   - Use case
   - Potential implementation approach

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear, descriptive messages
4. Write and update tests
5. Ensure all tests pass
6. Submit pull request with detailed description

## Development Setup
```bash
# Clone repository
git clone https://github.com/davidhonghikim/SuperSMM.git
cd SuperSMM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python3 -m pytest tests/
```

## Coding Standards
- Follow PEP 8 guidelines
- Write clear, concise docstrings
- Add type hints
- 100% test coverage for new code
- Use meaningful variable and function names

## Code Review Process
- All submissions require review
- Automated checks must pass
- At least one maintainer must approve the PR
- Consider performance, security, and maintainability

## Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Collaborate and support each other

## Questions?
Open an issue or contact maintainers directly.

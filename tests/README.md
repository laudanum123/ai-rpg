# RPG Application Testing Suite

This directory contains the testing suite for the web-based RPG application with an AI-powered Game Master.

## Test Structure

The tests are organized into three main categories:

- **Unit Tests** (`/tests/unit/`): Tests for individual components in isolation
  - `/models/`: Tests for data models and their methods
  - `/services/`: Tests for service layer classes
  - `/utils/`: Tests for utility functions

- **Integration Tests** (`/tests/integration/`): Tests for components working together
  - `/services/`: Tests for service interactions
  - `/api/`: Tests for API endpoints

- **Functional Tests** (`/tests/functional/`): End-to-end tests
  - `/routes/`: Tests for Flask routes

- **Mocks** (`/tests/mocks/`): Mock data and helper functions for tests

## Running Tests

1. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```

2. Run the entire test suite:
   ```
   pytest
   ```

3. Run specific test categories:
   ```
   pytest tests/unit/          # Run only unit tests
   pytest tests/integration/   # Run only integration tests
   pytest tests/functional/    # Run only functional tests
   ```

4. Run tests with coverage report:
   ```
   pytest --cov=app
   ```

## Writing Tests

When writing tests:

1. Follow the existing structure and naming conventions
2. Use fixtures from `conftest.py` where possible
3. Mock external dependencies (especially OpenAI API)
4. Test both success and failure cases
5. Aim for high coverage of critical game logic

## Testing External APIs

The OpenAI API is mocked in tests to avoid actual API calls. The mocks can be found in:
- `tests/conftest.py` - Basic OpenAI mock fixtures
- `tests/mocks/openai_responses.py` - More detailed mock responses

## Continuous Integration

Tests are automatically run on pull requests to ensure code quality.

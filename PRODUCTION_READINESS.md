# Production Readiness Audit & Issues

This document outlines the critical errors, edge cases, and architectural issues that need to be fixed before this project can be considered production-ready.

## 1. Silent Failures & Unhandled Exceptions Downstream

### Issue: LLM Provider Exception Swallowing
- **Location**: `src/generation/provider.py` (Lines 48-51)
- **Description**: The `GeminiProvider.generate()` method wraps the API call in a `try...except Exception as e:` block. If it catches an exception (e.g., network error, invalid API key, parsing error), it simply logs it and returns an empty dictionary `{}`.
- **Impact**: In `src/pipeline.py` (Line 86), the code expects the response to have specific keys like `llm_response["needs_escalation"]`. Returning `{}` will cause a `KeyError` and crash the entire pipeline, returning a 500 Internal Server Error in the API instead of gracefully escalating or retrying.
- **Fix**: The provider should return a structured fallback response (e.g., forcefully flagging `needs_escalation = True` with a system error reason) or the pipeline should handle the empty dictionary gracefully.

### Issue: Intent Classification Fallback Obscures Errors
- **Location**: `src/pipeline.py` (Lines 55-62)
- **Description**: If the embedding classifier fails (e.g., out of memory, invalid input type), it catches `Exception` and silently defaults to `intent = "unknown"` and `intent_confidence = 0.0`. 
- **Impact**: While this successfully triggers a safety escalation later, catching a broad `Exception` makes debugging critical infrastructure failures difficult. It also lacks metric tracking for model failures.

## 2. Infrastructure & Setup Issues

### Issue: Pytest PYTHONPATH Resolution
- **Description**: Running `pytest` directly in the root directory fails with `ModuleNotFoundError: No module named 'src'`. 
- **Fix**: The project requires either setting `PYTHONPATH="."` explicitly before running tests, using `pytest-env`, or creating a `setup.py`/`pyproject.toml` to install the `src` module in editable mode (`pip install -e .`).

### Issue: Hardcoded Dataset Paths
- **Location**: `config/config.yaml`
- **Description**: Paths like `data/raw/twcs.csv` and `data/processed` are relative and assume the script is executed from the exact root directory. 
- **Fix**: Paths should be resolved dynamically relative to a project root environment variable or `__file__` location to ensure scripts and APIs can be run reliably from any directory (e.g., inside Docker containers).

## 3. API & Data Validation Edge Cases

### Issue: Missing LLM Response Validation
- **Location**: `src/generation/provider.py`
- **Description**: The LLM output is expected to be a valid JSON matching a specific schema, but the only validation is `json.loads(text)`.
- **Impact**: If the LLM generates slightly malformed JSON or hallucinates keys, the application will crash.
- **Fix**: Use Pydantic models to strictly validate the structure of the LLM's JSON response *before* passing it back to the pipeline.

### Issue: State Singleton in API
- **Location**: `src/api.py` (Lines 11-22)
- **Description**: The API uses a global `agent_instance` loaded lazily. If the initialization fails (e.g., missing FAISS index, missing API key), it raises a `RuntimeError`. While the FastAPI `lifespan` tries to handle this, requests will continue to hit a 503 error if the system can't recover.
- **Fix**: Implement proper health check endpoints (`/health`) and ensure the application exits or reports an unhealthy state immediately during container startup rather than failing on the first user request.

## 4. Environment Configuration

### Issue: API Key Enforcement
- **Location**: `src/generation/provider.py` (Lines 29-30)
- **Description**: The system raises a `ValueError` if the `GEMINI_API_KEY` is missing *during* the generation step.
- **Fix**: Environment variable checks for critical secrets should happen at application boot time (in `api.py` or `pipeline.py` initialization) so the system fails fast rather than crashing mid-request.

---

### Next Steps
1. Refactor error handling in `GeminiProvider` to return a safe default dictionary matching the expected schema.
2. Introduce Pydantic validation for the LLM output.
3. Add a `pytest.ini` or `pyproject.toml` to fix module import paths for testing.
4. Validate environment variables aggressively at startup.

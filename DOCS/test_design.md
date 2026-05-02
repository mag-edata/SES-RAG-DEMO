# Test Design Document

**Project Name:** SES-RAG-DEMO  
**Version:** 1.0  
**Date Created:** 2026-03-20

---

## 1. Testing Policy

- Since this is a demo application, confirming that "core use cases work end-to-end" is prioritized over rigorous quality assurance.
- Because the system depends on an external API (OpenAI), mocks should be used as much as possible in unit tests.
- RAG accuracy evaluation is primarily qualitative; quantitative metrics (Top-3 precision) are used as supplementary measurements.
- Testing is conducted by a single developer via manual testing + pytest.

---

## 2. Test Scope

**In Scope**

- `rag/embedder.py` (Embedding generation and DB registration logic)
- `rag/retriever.py` (Similarity search logic)
- `rag/chain.py` (Prompt generation and LLM invocation)
- `app.py` (UI operation flow)
- Integration tests (end-to-end operation across all modules)
- Error cases (API key not set, DB not built, etc.)

**Out of Scope**

- Internal behavior of the OpenAI API (external service)
- Core implementation of ChromaDB
- Behavior of the Streamlit framework itself

---

## 3. Test Level Definitions

### Unit Test

| Item | Detail |
|------|--------|
| Target | Individual functions in each module |
| Method | pytest + unittest.mock (mock external API calls) |
| Timing | Upon completion of each module implementation |
| Pass Criteria | All test cases pass |

### Integration Test

| Item | Detail |
|------|--------|
| Target | The embedder → ChromaDB → retriever pipeline |
| Method | Verify using actual data with an in-memory ChromaDB for testing |
| Timing | After all modules are implemented |
| Pass Criteria | Search results return correctly and record counts match |

### System Test

| Item | Detail |
|------|--------|
| Target | Full flow including the Streamlit UI |
| Method | Manual operation via browser |
| Timing | Final verification before deployment |
| Pass Criteria | Core use cases complete successfully |

---

## 4. Test Environment

**Local Environment**

| Item | Detail |
|------|--------|
| OS | macOS (development machine) |
| Python | 3.14 (venv) |
| ChromaDB | Temporary directory via pytest `tmp_path` fixture (auto-cleaned) |
| OpenAI API | Real API key used (be mindful of test cost budget) |

**Test Configuration**

- Use pytest `tmp_path` fixture to redirect ChromaDB storage to a temporary directory for test isolation (see `tests/test_retriever.py` `chroma_dir` fixture)
- Mock the OpenAI API in unit tests to avoid incurring API costs

**Streamlit Cloud Verification**

- After deployment, perform system tests by manually operating the UI

---

## 5. Test Data Policy

### `jobs.json` (Job Requirements Data — 20 records)

| Test Type | Usage |
|-----------|-------|
| Integration tests / System tests | Use actual data as-is |
| Unit tests | Not used. All existing unit tests operate solely on engineer data or inline dicts. |

### `engineers.json` (Engineer Data — 20 records)

| Test Type | Usage |
|-----------|-------|
| Load verification (E-01) | Load all 20 records via `load_engineers()` and verify the count |
| Index build tests (E-03, E-04) | Use 3 inline engineer samples defined in test code (`SAMPLE_ENGINEERS`), not the full 20 records |
| Search tests (R-01–R-04) | Use `SAMPLE_ENGINEERS` (3 records) with a fixed fake embedding (`[0.1] × 1536`). All candidates receive the same vector, so only structure, count, and distance range are verified — semantic ranking is not tested |
| Semantic ranking verification (R-05) | Pending. Requires a real OpenAI API call to confirm that engineers with relevant skills appear in top results |
| Unit tests (mocked) | Define 2–3 fixed samples inline in the test code (no file I/O) |

**Test Data Principles**

- Do not use real company names or real individual names
- If additional test-specific data is needed, create a separate file rather than appending to `engineers.json`

---

## 6. Pass/Fail Criteria

### Unit Tests

- All pytest cases must PASS (0 errors, 0 failures)
- Coverage: At minimum, key branches in core functions must be covered

### Integration Tests

- Record count registered in ChromaDB must match the number of search hits
- Similarity scores must fall within the range of 0.0 to 1.0
- The number of results returned must match the requested top-k count

### System Tests

- The full flow of index build → run matching must complete without errors
- Appropriate error messages must be displayed in the UI when errors occur
- Response time must be within target values (search: 3 seconds, generation: 10 seconds)

---

## 7. Evaluation Metrics

### Top-3 Precision

| Item | Detail |
|------|--------|
| Definition | Number of results in Top-3 judged as "appropriate" by human review / 3 |
| Measurement Method | Manually score 5 test job requirements |
| Target Value | Average of 0.6 or higher (at least 3 out of 5 judged appropriate) |

### Validity of LLM-Generated Rationale

| Item | Detail |
|------|--------|
| Definition | Whether the generated matching rationale is grounded in the job requirements |
| Measurement Method | Manual review (0: inappropriate / 1: partially appropriate / 2: appropriate) |
| Target Value | Average score of 1.5 or higher |

### Reference Metrics (for future improvement)

| Metric | Description |
|--------|-------------|
| MRR (Mean Reciprocal Rank) | Measures at what position the ideal candidate appears |
| Recall@k | Proportion of correct candidates appearing within the Top-k results |

> In the demo phase, qualitative evaluation is prioritized; quantitative metrics are treated as reference values.

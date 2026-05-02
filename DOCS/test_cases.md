# Test Cases

**Project Name:** SES-RAG-DEMO  
**Version:** 1.2  
**Date Updated:** 2026-05-02

**Legend**

| Symbol | Meaning |
|--------|---------|
| Priority: H / M / L | High / Medium / Low |
| Type: UT / IT / ST | Unit Test / Integration Test / System Test |
| Result: O / X / - | Pass / Fail / Not Executed |

---

## 1. embedder.py Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| E-01 | `engineers.json` loads successfully and returns a list of 20 records | H | UT | O |
| E-02 | `build_profile_text()` receives an engineer dict and returns a string containing name, skills, and years of experience | H | UT | O |
| E-03 | `build_index()` returns the registered record count (int) upon successful completion (using mocked Embedding) | H | IT | O |
| E-04 | Running `build_index()` twice does not create duplicate entries; the collection is overwritten (rebuilt) | M | IT | O |
| E-05 | Passing a non-existent path for `engineers.json` raises an appropriate exception (e.g., `FileNotFoundError`) | M | UT | O |

---

## 2. retriever.py Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| R-01 | `search_engineers()` receives a string and returns a list (executed with ChromaDB index already built) | H | IT | O |
| R-02 | When `top_k=3`, the returned list contains exactly 3 records | H | IT | O |
| R-03 | Each element in the returned list contains `id` / `document` / `distance` / `metadata` | H | IT | O |
| R-04 | All distance values fall within the range of 0.0 to 2.0 (confirming ChromaDB distance metric range) | M | IT | O |
| R-05 | Inputting a Python backend job requirement text results in engineers with Python skills appearing in the top results (qualitative confirmation by visual review) | M | IT | - |

---

## 3. chain.py Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| C-01 | `build_prompt()` receives job text and a candidate list and returns a non-empty string | H | UT | O |
| C-02 | The output of `build_prompt()` contains the job requirement text and the names of candidate engineers | H | UT | O |
| C-03 | `generate_reasons()` returns a string when using a mocked LLM (OpenAI API call is mocked) | H | UT | O |
| C-04 | The output text of `generate_reasons()` is not an empty string (verified with real API call) | M | IT | - |
| C-05 | The generated text contains references to candidate engineer names or skills (qualitative confirmation) | L | IT | - |

---

## 4. app.py (UI) Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| A-01 | The Streamlit app starts successfully and displays the title and text area | H | ST | O |
| A-02 | Clicking the "Build Index" button in the sidebar displays a success message and the record count | H | ST | O |
| A-03 | Entering job requirements in the text area and clicking "Run Matching" displays 3 candidates with matching rationale | H | ST | O |
| A-04 | A loading spinner (e.g., `st.spinner`) is displayed during execution | M | ST | - |
| A-05 | The similarity score is displayed as a numeric value in the results | M | ST | O |

---

## 5. Integration Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| I-01 | The full flow of index build → run matching completes without errors | H | ST | O |
| I-02 | After rebuilding the index, re-running a search still returns results correctly (no duplicate registration issues) | H | IT | - |
| I-03 | Inputting 3 different job requirement texts (Python backend, infrastructure, PM-type) each returns different candidates (diversity of search results) | M | ST | O |
| I-04 | The same behavior as I-01 is confirmed in the Streamlit Cloud deployed environment | H | ST | O |
| I-05 | API call response times are within the specified range (search: 3 seconds, generation: 10 seconds) | M | ST | - |

---

## 6. Error Case Test Cases

| No. | Test Description | Priority | Type | Result |
|-----|------------------|----------|------|--------|
| X-01 | Starting the app with `OPENAI_API_KEY` not set displays error message E003 | H | ST | - |
| X-02 | Running matching with the index not built displays error message E002 | H | ST | - |
| X-03 | Clicking "Run Matching" with an empty text area displays error message E001 | H | ST | - |
| X-04 | Setting an invalid API key (correct format but authentication fails) displays E003 or E004 error | M | ST | - |
| X-05 | Running index build when `engineers.json` does not exist displays error message E005 | M | ST | - |
| X-06 | Displays an appropriate error when the ChromaDB storage directory lacks write permissions | L | ST | - |

---

## Test Execution Summary

| Item | Detail |
|------|--------|
| Date Executed | 2026-05-01 (UT/IT), 2026-05-02 (ST visual) |
| Executed By | mag |
| Total Test Cases | 31 |
| Passed | 19 (E-01–E-05, R-01–R-04, C-01–C-03, A-01, A-02, A-03, A-05, I-01, I-03, I-04) |
| Failed | 0 |
| Not Executed | 12 (R-05, C-04, C-05, A-04, I-02, I-05, X-01–X-06) |

**Notes**

- UT/IT (12 cases) executed via pytest on local environment (Python 3.14, venv). All mocked; no real API calls required.
- ST (7 cases) visually confirmed on Streamlit Cloud deployed environment on 2026-05-02.
- Remaining 12 cases require real API call or specific error conditions to reproduce.

# HalluciGuard Backend

Backend API for the HalluciGuard hallucination detection system.

The backend is built with **FastAPI** and currently provides authentication,
request validation, API endpoints, error handling, automated tests, and a
mock HalluciGuard workflow.

> **Current status:** The LangGraph/AI workflow is currently a mock.
> Members 4 and 5 will provide the final workflow interface later.
> The backend API is designed so that the real workflow can be integrated
> without changing the frontend-facing API contract.

---

## 1. Backend Responsibilities

The backend is responsible for:

- FastAPI application
- REST API endpoints
- API-key authentication
- Request validation
- Response validation
- CORS configuration
- Error handling
- Health checks
- API documentation through Swagger
- Automated API testing
- Environment-variable configuration
- Docker preparation
- Deployment preparation

The internal LangGraph/AI workflow is owned by Members 4 and 5.

---

## 2. Current Architecture

```text
Client
   |
   v
FastAPI
   |
   v
API-Key Authentication
   |
   v
Request Validation
   |
   v
HalluciGuard Workflow
   |
   |-- Currently a mock
   |
   v
Response Validation
   |
   v
JSON Response
```

The eventual architecture will connect the FastAPI backend to the
completed LangGraph workflow.

---

## 3. Project Structure

```text
HalluciGuard backend/
|
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .env
|-- .gitignore
|
|-- models/
|   |-- schemas.py
|
|-- workflow/
|   |-- graph.py
|
`-- tests/
    `-- test_api.py
```

### Important files

| File | Purpose |
|---|---|
| `main.py` | FastAPI application, routes, authentication, CORS, error handling |
| `config.py` | Loads configuration and API key from environment variables |
| `models/schemas.py` | Request and response validation models |
| `workflow/graph.py` | Current mock workflow; will later be replaced/integrated with the real workflow |
| `tests/test_api.py` | Automated API tests |
| `requirements.txt` | Pinned Python dependencies |
| `.env` | Local environment variables; should never be committed |
| `.gitignore` | Prevents secrets, virtual environments, and generated files from being committed |

---

## 4. Requirements

The backend currently uses:

- Python 3.13+
- FastAPI
- Uvicorn
- Pydantic
- python-dotenv
- pytest
- httpx

The direct dependency versions are pinned in `requirements.txt`.

---

## 5. Local Setup

### Step 1: Open the project directory

```powershell
cd "D:\Halluciguard backend"
```

For other computers, replace the path with the location of the project.

---

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

---

### Step 3: Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv)
```

at the beginning of the terminal prompt.

---

### Step 4: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 6. Environment Configuration

Create a file named:

```text
.env
```

in the project root.

Add:

```env
API_KEY=your-secret-api-key
```

Example:

```env
API_KEY=halluciguard-dev-key
```

Use a real secret value for development rather than committing a production
secret to the repository.

The `.env` file is included in `.gitignore` and must not be committed to Git.

---

## 7. Running the Backend

Start the development server with:

```powershell
python -m uvicorn main:app --reload
```

The backend will normally be available at:

```text
http://127.0.0.1:8000
```

---

## 8. API Documentation

FastAPI automatically provides Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger to test the API manually.

---

## 9. API Endpoints

### GET `/`

Basic backend status endpoint.

Example response:

```json
{
  "message": "HalluciGuard Backend is running"
}
```

---

### GET `/health`

Health-check endpoint.

Example response:

```json
{
  "status": "healthy",
  "service": "halluciguard-backend"
}
```

A successful response has HTTP status:

```text
200 OK
```

---

### POST `/verify`

Main verification endpoint.

This endpoint:

1. Authenticates the request using an API key.
2. Validates the incoming query.
3. Sends the validated query to the HalluciGuard workflow.
4. Validates the workflow response.
5. Returns a JSON response.

---

## 10. Authentication

The `/verify` endpoint requires the following HTTP header:

```text
x-api-key
```

Example:

```text
x-api-key: your-secret-api-key
```

### Missing API key

Response:

```text
401 Unauthorized
```

Example:

```json
{
  "detail": "API key is required"
}
```

### Invalid API key

Response:

```text
401 Unauthorized
```

Example:

```json
{
  "detail": "Invalid API key"
}
```

---

## 11. Verify Request

Example request:

```json
{
  "query": "What is artificial intelligence?"
}
```

The `query` field:

- must be a string
- cannot be empty
- cannot contain only whitespace
- is trimmed before being passed to the workflow

---

## 12. Verify Response

The current response schema is:

```json
{
  "answer": "Test response for: What is artificial intelligence?",
  "verification_status": "mock",
  "confidence": 0.0,
  "hallucination_score": 0.0,
  "citations": []
}
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `answer` | string | Generated answer |
| `verification_status` | string | Verification state |
| `confidence` | float | Confidence score from 0 to 1 |
| `hallucination_score` | float | Hallucination score from 0 to 1 |
| `citations` | list of strings | Supporting citations |

The current values are produced by the mock workflow.

The final values will come from the real HalluciGuard workflow after Members 4
and 5 provide their integration interface.

---

## 13. Validation Errors

Invalid requests return HTTP:

```text
422 Unprocessable Content
```

For example:

```json
{
  "query": ""
}
```

or:

```json
{
  "query": "     "
}
```

---

## 14. Internal Workflow Errors

If the workflow encounters an unexpected internal error, the API returns:

```text
500 Internal Server Error
```

with a safe message:

```json
{
  "detail": "Unable to process the query"
}
```

Internal implementation details are not exposed to the API client.

---

## 15. CORS

CORS middleware is currently configured in the FastAPI application.

The current development configuration permits cross-origin requests.

For production deployment, allowed origins should be restricted to the actual
frontend domain instead of using an unrestricted development configuration.

---

## 16. Running Automated Tests

Make sure the virtual environment is activated.

Run:

```powershell
python -m pytest
```

The current test suite covers:

- Home endpoint
- Health endpoint
- Valid API-key authentication
- Missing API key
- Invalid API key
- Empty query validation
- Whitespace-only query validation
- Surrounding whitespace handling
- Workflow failure handling

The current test suite contains **9 tests**.

Expected result:

```text
9 passed
```

A dependency warning related to the current FastAPI/Starlette testing stack may
appear. The tests themselves should still pass.

---

## 17. Security Notes

Do not commit:

```text
.env
```

to Git.

Do not hard-code production API keys in Python source code.

The current API key is intended for development/testing.

Production deployment should use the deployment platform's secure environment
variable or secret-management mechanism.

---

## 18. Current Workflow Status

The file:

```text
workflow/graph.py
```

currently contains a mock implementation.

This is intentional.

The backend is waiting for Members 4 and 5 to provide the final LangGraph
workflow interface, including:

- workflow execution function
- required input parameters
- output structure

Once that interface is available, the backend will integrate the workflow
through the existing `/verify` endpoint.

The frontend-facing API contract should remain stable wherever possible.

---

## 19. Current Development Status

### Completed

- FastAPI backend
- REST endpoints
- API-key authentication
- `.env` configuration
- Request validation
- Response validation
- CORS
- Health endpoint
- Error handling
- Automated API tests
- Dependency version pinning
- Basic API documentation

### Pending

- Final LangGraph workflow integration
- Frontend integration
- Dockerization
- Deployment

The LangGraph integration is pending because the final workflow interface has
not yet been provided by Members 4 and 5.

Frontend integration and final deployment will be completed when the frontend
and real workflow are available.
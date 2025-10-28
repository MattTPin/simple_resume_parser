# SimpleResumeParser

SimpleResumeParser is a lightweight, production-ready resume parsing framework that extracts structured data from resumes in `.pdf` or `.docx` formats. The parser focuses on three primary fields: `name`, `email`, and `skills`. Extraction leverages a combination of approaches:

- RAG / LLMs for context-aware name and skill detection
- NER models via HuggingFace for entity recognition
- Rules-based logic for common patterns
- Regex for structured patterns such as emails

The framework is built with a **pluggable architecture**, featuring abstract base classes and concrete implementations for parsers and field extractors. This design enables flexibility, easy extensibility, and thorough testing.

---

## Features

- Supports **PDF** and **Word Document** resumes
- Extracts structured information: `name`, `email`, `skills`
- Pluggable framework: add new parsers or extractors easily
- Fully Dockerized for development and deployment
- Provides a FastAPI server with Swagger UI for testing and integration

![Diagram](https://github.com/user-attachments/assets/fe6351cc-584a-4a77-bcea-9acd5ca4f9f8 "Diagram")

---

## Installation

Clone the repository:

``` bash
git clone <REPO_URL>
cd simple_resume_parser
```

### Docker Setup

#### 1. Standard Mode (for deployment or general use)

Uses `docker-compose.yml` in the project root. Launch the API server:

``` bash
docker compose -f ./docker-compose.yml up --build
```

- **API Base URL:** [http://localhost:8000](http://localhost:8000)
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

#### 2. Dev Container Mode (i.e. VS Code)

Intended for development and debugging and to inter. Uses `.devcontainer/docker-compose-dev.yml` (and the devcontainer config specified in `.devcontainer/devcontainer.json`) to launch the environment.

- In your IDE or terminal, start the dev container using your container management tools (for example, VS Code’s Reopen in Container, JetBrains Docker integration, or `docker compose -f .devcontainer/docker-compose-dev.yml up --build`).

Once the dev container is running, manually launch the server to access the API endpoints:
```python
python run_server_local.py
```

- **API Base URL:** [http://localhost:8001](http://localhost:8001)
- **Swagger UI:** [http://localhost:8001/docs](http://localhost:8001/docs)

Alternatively, open the repo in VS Code → "Reopen in Container" to develop inside the container.

### Help
NOTE: Sometimes Docker can't automatically download the Python base image. If you see an error like `failed to solve: error getting credentials - err:`, manually pull the official Python 3.11 slim image first:

```bash
docker pull python:3.11-slim
```

---

## API Usage

After launching the docker container you can access the API endpoint at `http://localhost:8000` (or `http://localhost:8001` if running the dev container). You can make curl requests or open up a user interface to interact with (SwaggerUI) by going to http://localhost:8000 which lets you directly upload resumes to parse.

### Endpoint

`POST /parse_resume`

- Accepts a single file upload (`.pdf` or `.docx`)
- Returns a JSON object containing `name`, `email`, and `skills`

### Example cURL Request

``` bash
curl -X POST http://localhost:8000/parse_resume \
    -F "file=@path/to/resume.pdf"
```

### Expected JSON Response

``` json
{
    "name": "Jane Doe",
    "email": "jane.doe@gmail.com",
    "skills": ["Python", "Machine Learning", "LLM"]
}
```

## CLI Usage

You can also run SimpleResumeParser directly from the command line without Docker by using the provided CLI script parse_resume_cli.py.

### Dockerless Setup

Create a virtual environment (optional but recommended):

``` bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the CLI:

``` bash
python parse_resume_cli.py <file_path>
```

Replace <file_path> with the path to the resume file you want to parse (.pdf or .docx). The script will print the extracted name, email, and skills directly to the terminal.

Example

``` bash
python parse_resume_cli.py resumes/jane_doe.pdf
```

Output:

``` plaintext
Resume Parsing Result:
Name: Jane Doe
Email: jane.doe@gmail.com
Skills: Python, Machine Learning, LLM
```
---

## .env File for LLM Client IDs

The `.env` file is used to securely store API keys required to query the LLM client. In order to make LLM queries simply copy the contents of `.env.template` your own `.env` file and it will be automatically loaded into the app.

### Example: Anthropic (Claude) API Key

```dotenv
# Anthropic LLM API Setup
ANTHROPIC_API_KEY=<REPLACE_ME>
```

## Architecture Overview

1. **File Parsers**
    - FileParser (abstract base class)
        - The base class that all parsers use.
    - `PDFParser`
        - Parses PDF files (.pdf) using PyMuPDF.
    - `WordParser`
        - Parses Word documents (.docx) using docx2txt.


2. **Field Extractors**
    - Abstract base class: `FieldExtractor`
    - Each concrete implementation supports different extraction methods to parse.
    - `NameExtractor`: Supports `llm` and `ner`. Defaults to `ner`. When run through ResumeParserFramework automatically runs `llm` as backup if `ner` extraction fails. 
    - `EmailExtractor`: Supports `regex` and `rule` (spacy "en_core_web_sm") based extraction. Defaults to Regex.
    - `SkillsExtractor`: Supports `llm` extraction (via LangChain). Defaults to `llm`.


3. **ResumeExtractor**
    - A an orchestration class that runs multiple FieldExtractors on the output of a parsed file (a List of DocumentChunk entries). Takes a dictionary of FieldExtractors (called a `extractor_map`) that specifies which FieldExtractors to run for each field with the option to run a different extraction method as backup should the first method fail. They are passed in this format.
    ```python
    Example:
    {
        "name": [
            NameExtractor(extraction_method="ner"), # Default extraction method 
            NameExtractor(extraction_method="llm") # Backup extraction method
        ],
        "email": [EmailExtractor(...)],
        "skills": [SkillsExtractor(...)]
    }
    ```
    - ResumeExtractor will intelligently check how many cores are available on the machine it is running on and attempt to thread the current extraction for each field so they run in parallel. 

4. **ResumeParserFramework**
    - Orchestrates file parsing and field extraction by calling the appropriate FileParser followed by ResumeExtractor.
    - Entry point method: `ResumeParserFramework.parse_resume(file_path: str) -> ResumeData`

---

## Caching and Shared Model Handling in `FieldExtractor`

The `FieldExtractor` class in `field_extractor.py` is designed to efficiently extract fields from resumes using different strategies, including regex, SpaCy NER, HuggingFace models, and LLMs. To improve performance and avoid repeated expensive operations, several caching mechanisms are used.

---

### 1. LLMClient Sharing

The `LLMClient` is responsible for querying large language models (LLMs) using the Langchain Framework. It can be initialized in two ways:

1. **Directly in a FieldExtractor** – An individual extractor can instantiate its own client if needed.
2. **Through an orchestrating framework** – A higher-level function like `ResumeParseFramework` or `ResumeParser` can create a single `LLMClient` instance and pass it to all `FieldExtractor` instances it utilizes.

**Benefits:**

- **Shared client**: Multiple extractors can reuse the same client, avoiding redundant initializations.
- **Time-saving**: LLM clients can take time to initialize, especially if they manage sessions or perform warm-up operations. Sharing a client avoids repeating this overhead for every extraction.
- **Consistency**: Using a single client ensures consistent configuration across all extractors during a parsing session.

---

### 2. Model Caching: `loaded_spacy_models` and `loaded_hf_models`

To optimize NLP model usage, `FieldExtractor` supports two shared caches:

- **`loaded_spacy_models`** – A dictionary storing SpaCy models keyed by their model name.
- **`loaded_hf_models`** – A dictionary storing HuggingFace pipelines keyed by model name.

**How it works:**

Much like `LLMClient`, these caches are most efficiently initialized by an orchestrating class (such as `ResumeParseFramework`) at the start of a parsing session. The orchestrator can pre-load required models and pass the shared dictionaries down to all `FieldExtractor` instances it creates.

1. When an extractor needs a model (SpaCy or HuggingFace), it first checks if the model is present in the corresponding cache.
2. If the model is not cached, it is loaded from disk or downloaded, then immediately added to the shared dictionary.
3. Future extractors can access the cached model directly, avoiding repeated loading.

**Advantages:**

- **Memory efficiency**: Only one instance of each model is loaded, even if multiple extractors run in parallel.
- **Performance**: Loading NLP models is often time-consuming. Caching reduces startup time for new extractor instances.
- **Real-time updates**: As models are loaded, the cache grows dynamically, so any extractor created later can immediately benefit from previously loaded models.

**Example usage in a FieldExtractor:**

```python
from src.parse_classes.field_extractor.helper_functions.ml.spacy_loader import load_spacy_model
from src.parse_classes.field_extractor.helper_functions.ml.hf_loader import load_hf_model

# Load SpaCy model, using shared cache
nlp = load_spacy_model("en_core_web_sm", loaded_spacy_models)

# Load HuggingFace NER model, using shared cache
ner_pipeline = load_hf_model("dbmdz/bert-large-cased-finetuned-conll03-english", loaded_hf_models)

---
## Testing

The project uses **pytest**:

``` bash
pytest tests/
```

- Tests cover parser functionality, field extractors, and edge cases
- LLM-related tests skip automatically if API keys are not set
- Files in `tests/` are laid out to mirror the functionality in `src`.

---

# Config.py Overview

`config.py` contains default settings used across the `simple_resume_parser` project. These settings are grouped under the `ScannerDefaults` dataclass and provide configurable parameters for file parsing, resume extraction, and LLM client behavior. Below is a description of each setting:

---

## ScannerDefaults

### FileParser Settings

- **CHUNK_SIZE (int, default=500)**  
  Determines the number of characters each parsed chunk should contain. This helps in splitting large files into manageable pieces for processing.

- **MAX_FILE_SIZE_MB (float, default=5.0)**  
  Specifies the maximum file size (in megabytes) that the parser will accept. Files exceeding this size may be rejected or trigger an error.

---

### ResumeExtractor Settings

- **MAX_THREADS (int, default=3)**  
  Sets the maximum number of threads the resume extraction process can use. Increasing this can speed up processing on multi-core systems, but may consume more memory.

---

### LLMClient Settings

- **LLM_PROVIDER (str, default="anthropic")**  
  Defines which LLM provider to use. Currently, only `"anthropic"` is supported.

- **ANTHROPIC_MODEL_ID (str, default="claude-haiku-4-5")**  
  Specifies the model ID to use when connecting to the Anthropic API.
---

## Usage

Import the default settings wherever needed in the project:

```python
from config import SCANNER_DEFAULTS

# Access default chunk size
print(SCANNER_DEFAULTS.CHUNK_SIZE)
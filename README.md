# SimpleResumeParser

SimpleResumeParser is a lightweight, production-ready resume parsing framework that extracts structured data from resumes in `.pdf` or `.docx` formats. The parser focuses on three primary fields: `name`, `email`, and `skills`. Extraction leverages a combination of approaches:

- RAG / LLMs for context-aware name and skill detection
- NER models via HuggingFace for entity recognition
- Rules-based logic for common patterns
- Regex for structured patterns such as emails

The framework is built with a **pluggable architecture**, featuring abstract base classes and concrete implementations for parsers and field extractors. This design enables flexibility, easy extensibility, and thorough testing.

## Documentation

Check out the full [Simple Resume Parser Wiki](https://github.com/<username>/<repo>/wiki/Home) for detailed documentation.


## Features

- Supports **PDF** and **Word Document** resumes
- Extracts structured information: `name`, `email`, `skills`
- Pluggable framework: add new parsers or extractors easily
- Fully Dockerized for development and deployment
- Provides a FastAPI server with Swagger UI for testing and integration

![Diagram](https://github.com/user-attachments/assets/fe6351cc-584a-4a77-bcea-9acd5ca4f9f8 "Diagram")



## Installation

First, clone the repository to your local machine:

``` bash
git clone <REPO_URL>
cd simple_resume_parser
```

## Docker Setup

This project is setup to run within docker contains if you so desire. There are two separate docker-compose configurations: A standard mode and a dev mode.

### 1. Standard Docker Mode (for deployment or general use)

Uses `docker-compose.yml` in the project root. Launch the API server:

``` bash
docker compose -f ./docker-compose.yml up --build
```

- **API Base URL:** [http://localhost:8000](http://localhost:8000)
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Dev Container Mode (i.e. VS Code)

Intended for development and debugging and to inter. Uses `.devcontainer/docker-compose-dev.yml` (and the devcontainer config specified in `.devcontainer/devcontainer.json`) to launch the environment.

- In your IDE or terminal, start the dev container using your container management tools (for example, VS Code’s Reopen in Container, JetBrains Docker integration, or `docker compose -f .devcontainer/docker-compose-dev.yml up --build`).

Once the dev container is running, manually launch the server to access the API endpoints:
```python
python run_server_local.py
```

- **API Base URL:** [http://localhost:8001](http://localhost:8001)
- **Swagger UI:** [http://localhost:8001/docs](http://localhost:8001/docs)

Alternatively, open the repo in VS Code → "Reopen in Container" to develop inside the container.

### Docker Help
NOTE: Sometimes Docker can't automatically download the Python base image. If you see an error like `failed to solve: error getting credentials - err:`, manually pull the official Python 3.11 slim image first:

```bash
docker pull python:3.11-slim
```



## API Usage

After launching the docker container you can access the API endpoint at `http://localhost:8000` (or `http://localhost:8001` if running the dev container). You can make curl requests or open up a user interface to interact with (SwaggerUI) by going to `http://localhost:8000/docs` which lets you directly upload resumes to a UI to perform your extractions.

### /parse_resume

```bash
POST /parse_resume
- Accepts a single file upload (`.pdf` or `.docx`)
- Returns a JSON object containing `name`, `email`, and `skills`
```

#### Example cURL Request

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

This tool can also be run outside of a docker container (ideally within a venv).

1. Create a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

2. Install dependencies:

``` bash
pip install -r requirements.txt
```

3. Run the CLI:

``` bash
python parse_resume_cli.py <file_path>
```

Replace <file_path> with the path to the resume file you want to parse (.pdf or .docx). The script will print the extracted name, email, and skills directly to the terminal.

Example:
Note that this example uses a file included with this repo so it's great for testing!

``` bash
python parse_resume_cli.py test_documents/test_pdfs/classic-ms-word-resume-template.pdf
```

Output:

``` plaintext
Resume Parsing Result:
Name: DEVIKA PATEL
Email: dev.patel@email.com
Skills: Microsoft Suite, Organization, Time Management, Written/Verbal Communication, Creativity, Collaboration, Critical Thinking, Compassion
```

## .env File for LLM Client IDs

The `.env` file is used to securely store API keys required to query the LLM client. In order to make LLM queries simply copy the contents of `.env.template` your own `.env` file and it will be automatically loaded into the app. Note that currently only Claude querying is supported.

### Example: Anthropic (Claude) API Key

```dotenv
# Anthropic LLM API Setup
ANTHROPIC_API_KEY=<REPLACE_ME>
```

## Testing

The project uses **pytest**:

``` bash
pytest tests/
```

- Tests cover parser functionality, field extractors, and edge cases
- LLM-related tests skip automatically if API keys are not set
- Files in `tests/` are laid out to mirror the functionality in `src`.



## Config.py Overview

`config.py` contains default settings used across the `simple_resume_parser` project. These settings are grouped under the `ScannerDefaults` dataclass and provide configurable parameters for file parsing, resume extraction, and LLM client behavior. Below is a description of each setting:

### ScannerDefaults

#### FileParser Settings

- **CHUNK_SIZE (int, default=500)**  
  Determines the number of characters each parsed chunk should contain. This helps in splitting large files into manageable pieces for processing.

- **MAX_FILE_SIZE_MB (float, default=5.0)**  
  Specifies the maximum file size (in megabytes) that the parser will accept. Files exceeding this size may be rejected or trigger an error.

#### ResumeExtractor Settings

- **MAX_THREADS (int, default=3)**  
  Sets the maximum number of threads the resume extraction process can use. Increasing this can speed up processing on multi-core systems, but may consume more memory.


#### LLMClient Settings

- **LLM_PROVIDER (str, default="anthropic")**  
  Defines which LLM provider to use. Currently, only `"anthropic"` is supported.

- **ANTHROPIC_MODEL_ID (str, default="claude-haiku-4-5")**  
  Specifies the model ID to use when connecting to the Anthropic API.



### Usage

Import the default settings wherever needed in the project:

```python
from config import SCANNER_DEFAULTS

# Access default chunk size
print(SCANNER_DEFAULTS.CHUNK_SIZE)
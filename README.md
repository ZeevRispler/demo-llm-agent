# LLM agent

Demo an end to end LLM agent solution with modular architecture, persistent storage and front-end UI that can work with various LLM models and storage solutions.

the configuration is specified in a YAML file, which indicate the model, embeddings, storage to use, and various parameters. 
the user can point to the configuration file by setting the `AGENT_CONFIG_PATH` environment variable.

environment variables and credentials can be loaded from a `.env` file in the root directory. or an alternate path set by the `AGENT_ENV_PATH` environment variable.
data can be stored in local files or remote SQL and Vector databases. the local file storage path can be set by the `AGENT_DATA_PATH` environment variable (defaults to `./data/`).

# Example configuration

{see the AppConfig class in the src/config.py file}
```yaml
chunk_overlap: 20
chunk_size: 1024
default_llm:
  class_name: langchain.chat_models.ChatOpenAI
  model_name: gpt-3.5-turbo
  temperature: 0
default_vector_store:
  class_name: chroma
  collection_name: default
  persist_directory: C:\Users\Yaron Haviv\PycharmProjects\demo-llm-agent\data\chroma
embeddings:
  class_name: huggingface
  model_name: all-MiniLM-L6-v2
log_level: DEBUG
pipeline_args: {}
verbose: false
```

Minimal environment file (`.env`):
```shell
IS_LOCAL_CONFIG=1
OPENAI_API_KEY=your-key
OPENAI_API_BASE=https://api.openai.com
```

Substitute the `OPENAI_API_KEY` and `OPENAI_API_BASE` with your own key and base address.


# Getting it to work

## Installation
```shell
pip install -r requirements.txt
```

> May nee to add stuff
>  If you get protobuf error, try to install it with `conda install protobuf==3.20.3`
Make sure there `.env` file is set before running the following commands, and if needed make modifications to the config

## Initialize the database:

```shell
python -m src.main initdb
```

## Initialize the product database:

```shell
python -m company_data.main
```

## Ingest data into the vector database:

```shell
python -m src.main ingest ./src/rag_data/jewelry_size_help.txt
python -m src.main ingest ./src/rag_data/jewelry_policies.txt
python -m src.main ingest ./src/rag_data/jewelry_matching.txt

```
> do for every file you want to ingest (in rag data folder)


> should be done only once, or when we want to erase the DB and start fresh 

## To start the API server:

```shell
uvicorn src.pipeline:app

```
# Additional steps for the UI not needed for now

```shell
## To start Vizro UI:

```shell
python -m src.viz.app
```


# CLI usage

To ingest data into the vector database:
```shell
python -m src.main ingest -l web https://milvus.io/docs/overview.md
```

To ask a question:
```shell   
python -m src.main query "whats a vector" 
```


Full CLI:

```shell
python -m src.main

Usage: python -m src.main [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  config  Print the config as a yaml file
  ingest  Ingest documents into the vector database
  initdb  Initialize the database (delete old tables)
  list    List the different objects in the database (by category)
  query   Run a chat quary on the vector database collection
```




1) Create `.env` file with parameters
```
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
MODEL_NAME="gpt-4.1-mini"
EMBED_MODEL_NAME="text-embedding-3-small"
INDEX_NAME="test-index"
```

2) Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` with your endpoint url and key

3) Run command `sh run.sh` if you want to run streamlit
4) Run command `sh run_react_agent.sh` if you want to run ReActAgent CLI app

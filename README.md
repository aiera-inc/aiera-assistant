# Aiera Assistant

This repository packages a basic [OpenAI Assistant](https://platform.openai.com/docs/assistants/how-it-works) chatbot for use with [Aiera's API](www.aiera.com). Aiera offers a rich dataset of financial event transciptions. Contact sales@aiera.com for more info.  

The application is built with [streamlit](https://docs.streamlit.io/) and OpenAI's beta Assistants API. Aiera Assistant is a custom assistant built on the preview of gpt4-1106-preview.


The Aiera Assistant is able to perform:  
1. Summarization of earnings call transcripts
2. SWOT analysis
3. KPI extraction
4. Comparisons of KPIs and important metrics across financial earnings calls events

![image](docs/assistant_snapshot.png)

## SETUP

### 1. Set up your environment 

Use [conda](https://docs.conda.io/en/latest/) to create a virtual environment and install the dependencies specified in the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate aiera-assistant
```

### 2. Install the assistant

```bash
pip install -e .
```

### 3. Configuration

The 

| Environment Variable | Description                                                 |
|----------------------|-------------------------------------------------------------|
| OPENAI_API_TOKEN     | API token provided by OpenAI                                |
| OPENAI_ORG_ID        | org id provided by OpenAI                                   |
| OPENAI_PERSIST_FILES | true/false, whether to persist files uploaded to OpenAI     | 
| AIERA_API_KEY        | API key as provided by Aiera. Contact us at sales@aiera.com |
| LOGGING_LEVEL        | standard python logging level                               |


### 4. Run the application using streamlit

```bash
streamlit run aiera_assistant/main.py
```


## Known issues

1. Citation generation is still a little spotty. Appears to be an issue after a couple of messages.
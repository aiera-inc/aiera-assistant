# Aiera Assistant

This repository packages a basic [OpenAI Assistant](https://platform.openai.com/docs/assistants/how-it-works) conversational chat for use with [Aiera's API](www.aiera.com). Contact sales@aiera.com for more info.  

The application is built with [streamlit](https://docs.streamlit.io/) and OpenAI's beta Assistants API. Aiera Assistant is a custom assistant built on the preview of gpt4-turbo-preview.

![image](docs/assistant_snapshot.png)

## SETUP

### 1. Set up your environment 

Use [conda](https://docs.conda.io/en/latest/) to create a virtual environment and install the dependencies specified in the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate aiera-assistant
```

### 2. Configuration

Set up your environment using the following variables:

| Environment Variable | Description                                                 |
|----------------------|-------------------------------------------------------------|
| OPENAI_API_KEY       | key provided by OpenAI                                      |
| OPENAI_ORG_ID        | org id provided by OpenAI                                   |
| OPENAI_PERSIST_FILES | true/false, whether to persist files uploaded to OpenAI     | 
| AIERA_API_KEY        | API key as provided by Aiera. Contact us at sales@aiera.com |
| LOGGING_LEVEL        | standard python logging level                               |


### 3. Create the assistant
Launch Jupyter and create the assistant by executing the cells in the `AieraAssistant.ipynb` notebook. 

```bash
jupyter lab
```

Configure your environment to use the id of the assistant you've just created:
```bash
export OPENAI_ASSISTANT_ID={your id}
```

### 4. Install the package

```bash
pip install -e .
```

### 5. Run the application using streamlit

```bash
streamlit run aiera_assistant/main.py
```

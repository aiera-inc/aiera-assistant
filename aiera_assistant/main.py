import streamlit as st
from streamlit_chat import message
from aiera_assistant.config import openai_settings, aiera_settings, db_settings
from aiera_assistant.assistant import AieraAssistant
from aiera_assistant.__init__ import ROOT_DIR

import logging

assistant_logger = logging.getLogger("aiera_gpt.assistant")


def main():

    # Setting page title and header
    st.set_page_config(page_title="Aiera", page_icon=f"{ROOT_DIR}/aiera_assistant/assets/favicon.ico")
    st.markdown("<h1 style='text-align: center;'>Aiera Assistant</h1>", unsafe_allow_html=True)

    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = AieraAssistant(
            openai_settings = openai_settings,
            aiera_settings=aiera_settings,
            db_settings=db_settings

        )

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = st.session_state['assistant'].begin_conversation()

    #if 'citations' not in st.session_state:
    #    st.session_state["citations"] = []

    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = [ st.session_state['assistant'].model_name]


    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    #clear_button = st.sidebar.button("Clear Conversation", key="clear")
    #st.sidebar.title("Citations")
    #col = st.sidebar.columns(1)[0]
    #for citation in st.session_state["citations"]:
    #    col.write(citation)

    # reset everything
    #if clear_button:
    #    st.session_state['generated'] = st.session_state['assistant'].begin_conversation()
    #    st.session_state['messages'] = []
    #    st.session_state['model_name'] = [st.session_state['assistant'].model_name]

    #    st.session_state['assistant'].close_chat()
    #    del st.session_state['assistant']
    #    st.session_state['assistant'] = AieraAssistant(
    #        openai_settings = openai_settings,
    #        db_settings = db_settings,
    #        aiera_settings=aiera_settings
    #    )


    # container for chat history
    response_container = st.container()

    # container for text input
    container = st.container()

    with container:
        with st.form(key='user_input_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        
            # if user has submitted input, submit and process messages
            if submit_button and user_input: 

                with st.spinner(text='Processing...'):

                    # disable while processing messages...
                    st.session_state['generated'].append({"role": "user", "content": user_input})

                    st.session_state['assistant'].submit_message(user_input)

                    messages = st.session_state['assistant'].process_messages()
                    st.session_state['generated'] = [mess for mess in messages]

                    st.session_state['model_name'].append(st.session_state['assistant'].model_name)


        
    if st.session_state['generated']:

        with response_container:
            citations = []
            for i, mess in enumerate(reversed(st.session_state['generated'])):

                if mess["role"] == 'user':
                    message(mess["content"], is_user=True, key=str(i) + '_user')

                else:
                    content = mess["content"]
                    #if mess["annotations"]:
                    #    for annotation in mess["annotations"]:
                            # check that the text starts with unicode marker
                            # appears to be an error with model generating citations that use some of the text
                    #        if annotation["text"][1] == "u" or "†" in annotation["text"]:
                    #            content = content.replace(annotation["text"], f" [{len(citations)}]")
                    #            citations.append(f"[{len(citations)}] {annotation['quote']}")

                    with st.chat_message('Aiera', avatar=f"{ROOT_DIR}/aiera_assistant/assets/favicon.ico"):
                        st.write(content)

            st.session_state["citations"] = citations




if __name__ == "__main__":
    main()

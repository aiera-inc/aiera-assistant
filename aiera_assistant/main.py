import streamlit as st
import re
from streamlit_chat import message
from aiera_assistant.config import openai_settings, aiera_settings, db_settings
from aiera_assistant.assistant import AieraAssistant
from aiera_assistant.__init__ import ROOT_DIR

import logging

assistant_logger = logging.getLogger("aiera_gpt.assistant")


def main():

    # Setting page title and header
    st.set_page_config(page_title="Aiera", page_icon=f"{ROOT_DIR}/aiera_assistant/assets/aiera-icon-logo-circle.png")
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

                    st.session_state['generated'].append({"role": "user", "content": user_input})

                    # trigger assistant processing
                    st.session_state['assistant'].submit_message(user_input)
                    messages = st.session_state['assistant'].process_messages()

                    # update messages
                    st.session_state['generated'] = [mess for mess in messages]


        
    if st.session_state['generated']:

        with response_container:
            citations = []
            for i, mess in enumerate(reversed(st.session_state['generated'])):

                if mess["role"] == 'user':
                    message(mess["content"], is_user=True, key=str(i) + '_user')

                else:
                    content = mess["content"]
                    if "【" in content:
                        content = re.sub(r'【(.*?)】', '', content)

                    with st.chat_message('Aiera', avatar=f"{ROOT_DIR}/aiera_assistant/assets/aiera-icon-logo-circle.png"):
                        st.write(content)

            st.session_state["citations"] = citations


if __name__ == "__main__":
    main()

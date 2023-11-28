import streamlit as st
from streamlit_chat import message
from aiera_gpt.config import openai_settings, aiera_settings

import openai


from aiera_gpt.assistant import AieraGPTAssistant
from aiera_gpt.__init__ import ROOT_DIR

import logging

assistant_logger = logging.getLogger("aiera_gpt.assistant")


def main():

    #db_config = AieraDBConfig(
    #            db_uri=database_settings.read_url,
    #            charset="utf8mb4",
    #        )

    # Setting page title and header
    st.set_page_config(page_title="Aiera", page_icon=f"{ROOT_DIR}/aiera_gpt/assets/logo.png")
    st.markdown("<h1 style='text-align: center;'>Aiera Assistant</h1>", unsafe_allow_html=True)

    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = AieraGPTAssistant(
            openai_settings = openai_settings,
            aiera_settings=aiera_settings

        )

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = [ st.session_state['assistant'].model_name]

    if 'cost' not in st.session_state:
        st.session_state['cost'] = []

    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = [0]

    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    #model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # reset everything
    if clear_button:

        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = []
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = [st.session_state['assistant'].model_name]
        st.session_state['cost'] = []
        #st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = [0]
        counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

        st.session_state['assistant'].close_chat()
        st.session_state['assistant'] = AieraGPTAssistant(
            settings = openai_settings,
            db_config = db_config,
            aiera_settings=aiera_settings
        )


    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input: #...
            st.session_state['generated'].append({"role": "user", "content": user_input})
            st.session_state['assistant'].submit_message(user_input)
            messages = [mess for mess in st.session_state['assistant'].process_messages()]
            st.session_state['generated'] = messages
            st.session_state['total_tokens'].append(st.session_state['assistant'].total_token_count)
            st.session_state['model_name'].append(st.session_state['assistant'].model_name)
            #st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            #if model_name == "GPT-3.5":
            #    cost = total_tokens * 0.002 / 1000
            #else:
            #    cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            #st.session_state['cost'].append(cost)
            #st.session_state['total_cost'] += cost
        
    if st.session_state['generated']:
        with response_container:
            for i, mess in enumerate(reversed(st.session_state['generated'])):
                #message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                if mess["role"] == 'user':
                    message(mess["content"], is_user=True, key=str(i) + '_user')

                else:
                    message(mess["content"], key=str(i))
                    #st.write(
                    #f"Model used: {st.session_state['model_name'][i//2+1]}; Number of tokens: {st.session_state['total_tokens'][i//2+1]};")

              #  st.write(
              #      f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]};")
                counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


if __name__ == "__main__":
    main()

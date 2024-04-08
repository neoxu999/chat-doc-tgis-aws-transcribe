import os

import streamlit as st
# from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference

import caikit_tgis_langchain
from gui.history import ChatHistory
from gui.layout import Layout
from gui.sidebar import Sidebar, Utilities
from snowflake import SnowflakeGenerator
from pathlib import Path

username = os.environ.get("REDIS_USERNAME", "default")
password = os.environ.get("REDIS_PASSWORD", "default")
host = os.environ.get("REDIS_HOST", "127.0.0.1")
file_name = os.environ.get("DOCUMENT_NAME", "")
redis_url = f"redis://{username}:{password}@{host}:6379"
certificate_chain = os.environ.get("CERTIFICATE_CHAIN_FILE", "/app/rag-ssl.pem")
protocol = os.environ.get("PROTOCOL", "http")
sources_file = os.environ.get("SOURCE_FILE", "/app/chat-doc-tgis-aws-transcribe/sources/guideline.pdf")

inference_server_url = os.environ.get('INFERENCE_SERVER_URL',
                                      'https://llm-2-llm.apps.rosa-csfpc.p9o9.p1.openshiftapps.com')
model_id = os.environ.get("MODEL_ID", "Llama-2-7b-chat-hf-sharded-bf16")

if __name__ == '__main__':
    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []
    if 'has_audio' not in state:
        state['load_audio'] = True
    if 'last_tts_id' not in st.session_state:
        st.session_state['last_tts_id'] = 0
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = ""
    if 'new_output' not in st.session_state:
        st.session_state['new_output'] = False

    st.set_page_config(layout="wide", page_icon="💬", page_title="ChatPDF")
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()

    layout.show_header()
    sidebar.show_logo()
    login_config = utils.load_login_details()

    if not login_config:
        layout.show_loging_details_missing()
    else:
        sidebar.show_login(login_config)
        transcript = utils.show_audio()
        # pdf = utils.handle_upload()

        if sources_file is None or (not os.path.exists(sources_file)):
            sources_file = "sources/guideline.pdf"

        pdf = sources_file
        pdf_name = Path(pdf).name

        if pdf:
            sidebar.show_options()

            try:
                if 'chatbot' not in st.session_state:
                    llm = caikit_tgis_langchain.CaikitLLM(
                        inference_server_url=inference_server_url,
                        model_id=model_id,
                        certificate_chain=certificate_chain,
                        streaming=False
                    )
                    # llm = HuggingFaceTextGenInference(
                    #     inference_server_url=os.environ.get('INFERENCE_SERVER_URL'),
                    #     max_new_tokens=int(os.environ.get('MAX_NEW_TOKENS', '512')),
                    #     top_k=int(os.environ.get('TOP_K', '10')),
                    #     top_p=float(os.environ.get('TOP_P', '0.95')),
                    #     typical_p=float(os.environ.get('TYPICAL_P', '0.95')),
                    #     temperature=float(os.environ.get('TEMPERATURE', '0.9')),
                    #     repetition_penalty=float(os.environ.get('REPETITION_PENALTY', '1.01')),
                    #     streaming=False,
                    #     verbose=False
                    # )

                    indexGenerator = SnowflakeGenerator(42)
                    index_name = str(next(indexGenerator))

                    chatbot = utils.setup_chatbot(pdf, llm, redis_url, index_name, "redis_schema.yaml")
                    st.session_state["chatbot"] = chatbot

                if st.session_state["ready"]:
                    history = ChatHistory()
                    history.initialize(pdf_name)

                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        is_ready, user_input = layout.prompt_form(transcript)

                        if st.session_state["reset_chat"]:
                            history.reset()

                        if is_ready:
                            with st.spinner("Processing query..."):
                                output = st.session_state["chatbot"].conversational_chat(user_input)
                    history.generate_messages(response_container)
                    # # reset transcript
                    # st.session_state['transcript'] = ""

            except Exception as e:
                st.error(f"{e}")
                st.stop()

    sidebar.about()

import streamlit as st
from streamlit_mic_recorder import mic_recorder

class Layout:

    def show_audio(self):
        st.write("Record your voice, and play the recorded audio:")
        audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')
        if audio:
            st.audio(audio['bytes'], format='audio/wav')
        return audio

    def show_header(self):
        """
        Displays the header of the app
        """
        st.markdown(
            """
            <h2 style='text-align: center;'>Chat with your PDF 💬</h1>
            """,
            unsafe_allow_html=True,
        )

    def show_loging_details_missing(self):
        """
        Displays a message if the user has not entered an API key
        """
        st.markdown(
            """
            <div style='text-align: center;'>
                <h4>Please config your credentials to start chatting.</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def prompt_form(self, transcript):
        """
        Displays the prompt form
        """
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="Ask me anything about the PDF...",
                key="input",
                value=transcript,
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send")

            is_ready = submit_button and user_input
        return is_ready, user_input


from langchain.callbacks.manager import CallbackManager
from langchain_ollama import ChatOllama
from chat_hst import *
from rag_engine import *

logging.basicConfig(level=logging.INFO)


def main():
    st.title("Internship Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    with st.sidebar:
        if st.button("Delete chat history"):
            st.session_state.messages = []
            save_chat_history([])

    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


    if user_input := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=BOT_AVATAR) and st.spinner("Generating response..."):
            try:

                placeholder = st.empty()
                response = ""

                cb_manager = CallbackManager([LoggingCallback()])
                llm = ChatOllama(model=MODEL_NAME, callback_manager=cb_manager, verbose=True, temperature=0)

                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                retriever = create_rerank_retriever(vector_db)

                chain = create_chain(retriever, llm)

                response = chain.invoke(input=user_input, run_manager=cb_manager)

                placeholder.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": response})

    save_chat_history(st.session_state.messages)

if __name__ == "__main__":
    main()

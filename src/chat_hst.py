import shelve

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("../chat/chat_history") as chat_db:
        return chat_db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("../chat/chat_history") as chat_db:
        chat_db["messages"] = messages
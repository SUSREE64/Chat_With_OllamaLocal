"""
This is simple application that chats locally with Ollama Models installed in your windows
This application Uses Streamlit for UI and Ollama Locally running , The model to chat can be
selected from a drop down menu.

Author : Temuzin
mailid : ssvipl64@gmail.com

"""

import streamlit as st
from langchain_ollama import ChatOllama
import ollama
import asyncio


# Cached ChatOllama instance for GPU acceleration
@st.cache_resource
def get_model_instance(model_name):
    """
    Initialize and cache the ChatOllama instance with GPU support.
    """
    return ChatOllama(model=model_name, base_url="http://127.0.0.1:11434", num_thread=8, use_gpu=True)


# Async function to handle model inference
async def generate_response(input_text, model_name, chat_history):
    """
    Generate a response from the model using GPU acceleration.
    Incorporates previous chat history as context.
    """
    # Create the conversation context
    context = "\n".join([f"User: {msg['user']}\nOllama: {msg['ollama']}" for msg in chat_history])
    context += f"\nUser: {input_text}\nOllama: "

    # Use cached model instance
    model = get_model_instance(model_name)

    # Run model inference in a separate thread
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, model.invoke, context)
    return response.content


# Streamlit app UI
st.markdown("### Chat with Ollama Models (GPU Optimized)")

# Load available models
result = ollama.Client()
models_list = [model["model"] for model in result.list().models]

# Sidebar for selecting a model
ollama_local_model = st.sidebar.selectbox('Choose Available Model', models_list)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# Input form for user queries
with st.form(key="myForm"):
    text = st.text_area("Enter your question")
    submit = st.form_submit_button("Submit", type='primary')

# Process user input
if submit and text:
    with st.spinner('Generating Response...'):
        # Generate response asynchronously
        response = asyncio.run(generate_response(input_text=text, model_name=ollama_local_model,
                                                 chat_history=st.session_state['chat_history']))
        # Append to chat history
        st.session_state['chat_history'].append({"user": text, "ollama": response})
        # Display response
        st.write(response)

# Display chat history
st.write("#### Chat History")
st.write("Model Selected:", ollama_local_model)
for chat in st.session_state['chat_history']:
    st.write(f"**User**: {chat['user']}")
    st.write(f"**Ollama**: {chat['ollama']}")
    st.write("------")

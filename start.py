import streamlit as st
import requests
import replicate
import os
import pandas as pd
import langchain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory  # Import ConversationBufferMemory
import base64
# Initialize ConversationBufferMemory
memory = ConversationBufferMemory()

serp_client = OpenAI(openai_api_key= '##########################')

# Function to call the SERPAPI using langchain
def call_serpapi(query):
    # Make a request to the SERPAPI with the query
    results = serp_client.predict(query)
    
    # Extract relevant information from the results (you may need to adjust this based on the response structure)
    serpapi_response = results
    
    # Update buffer memory with user query and SERPAPI response
    memory.save_context({"input": query}, {"output": serpapi_response})
    
    return serpapi_response

# Function to call the Stock API
def call_stock_api():
    # Make a request to your Stock API
    response = requests.get('#################################')
    # Update buffer memory with the response from Stock API
    memory.save_context({"input": "Stock API"}, {"output": response.json()})
    return response.json()

# Function to call the Retail Sector API
def call_retail_api():
    # Make a request to your Retail Sector API
    response = requests.get('###################################')
    # Update buffer memory with the response from Retail Sector API
    memory.save_context({"input": "Retail Sector API"}, {"output": response.json()})
    return response.json()

# Function for generating LLaMA2 response
def generate_llama_response(user_query, api_context):
    # Load buffer memory variables
    buffer_memory = memory.load_memory_variables({})
    
    # Generate LLM response with buffer memory and API context
    llm_output = replicate.run(llm, 
                               input={"prompt": f"{user_query} Assistant: ",
                                      "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1,
                                      "context": buffer_memory + api_context})
    
    llm_response = ''.join(llm_output)
    
    # Check if LLM provided a satisfactory response
    if llm_response.strip() == '':
        # If LLM didn't provide a response, try to get information from APIs
        api_response = get_api_info(user_query)
        if api_response:
            # Update buffer memory with the response from APIs
            memory.save_context({"input": user_query}, {"output": api_response})
            # Return the response from APIs
            return api_response
        else:
            # If APIs don't provide information, return a default message
            return "I'm sorry, I couldn't find information related to your query."
    else:
        # If LLM provided a response, update buffer memory with LLM response
        memory.save_context({"input": user_query}, {"output": llm_response})
        # Return the LLM response
        return llm_response

# Function to get information from APIs
def get_api_info(query):
    # Call SERPAPI
    serpapi_response = call_serpapi(query)
    # Call Stock API
    stock_response = call_stock_api()
    # Call Retail Sector API
    retail_response = call_retail_api()
    
    # Combine API responses into a context string
    api_context = f"Serpapi: {serpapi_response}\nStock API: {stock_response}\nRetail Sector API: {retail_response}"
    
    # Return the combined API responses
    return api_context

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Sidebar for API keys and parameters
with st.sidebar:
    st.title('API Integration Settings')

# User input
user_query = st.text_input('Ask something:')

# Button to trigger API calls
if st.button('Ask'):
    # Call SERPAPI
    serpapi_response = call_serpapi(user_query)
    # Call Stock API
    stock_response = call_stock_api()
    # Call Retail Sector API
    retail_response = call_retail_api()
    # Combine API responses into a context string
    api_context = f"Serpapi: {serpapi_response}\nStock API: {stock_response}\nRetail Sector API: {retail_response}"
    # Pass combined context string along with user query to the LLM
    llama_model_response = generate_llama_response(user_query, api_context)
    # Display responses
    st.write('Response from Llama Model:')
    st.write(llama_model_response)

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama_response(user_query, api_context):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    # Load buffer memory variables
    buffer_memory = memory.load_memory_variables({})
    # Generate LLM response with buffer memory and API context
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {user_query} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1,
                                  "context": buffer_memory + api_context})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_response(prompt, api_context)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# Button to save chat history to CSV
if st.button('Save Chat History to CSV'):
    df = pd.DataFrame(st.session_state.messages)
    df.to_csv("chat_history.csv", index=False)
    st.success("Chat history saved to chat_history.csv")

# Button to download CSV file
if os.path.exists("chat_history.csv"):
    st.markdown(get_binary_file_downloader_html("chat_history.csv", 'Download Chat History CSV'), unsafe_allow_html=True)

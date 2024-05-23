import streamlit as st
import bot

###############################
import io
from fastapi import FastAPI , File , UploadFile
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/qa/{question}")
async def get_answer(question: str):
    print(question + " received")
    return generate_response(question)

# async def create_upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     return {"filename": file.filename, "contents": contents}























#########################################

chain  = bot.rag_chain

def generate_response(text):
    response = chain.invoke(text)
    return response

st.title("MEDi-Bot")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
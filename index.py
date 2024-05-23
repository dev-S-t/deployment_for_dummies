import streamlit as st
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub

from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
pinecone_key = os.getenv("PINECONE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= google_api_key)

texts = [
        "What are the symptoms of diabetes?",
        "How to manage high blood pressure?",
        "What are the side effects of taking ibuprofen?"
    ]

index_name = "project-med"
vectorstore = PineconeVectorStore.from_texts(texts, index_name= index_name, embedding= embeddings)

# %%
retriever = vectorstore.as_retriever()






ret = retriever


llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-latest', google_api_key=google_api_key)

template = """
You are a medical chatbot. First respond to greetings only if the user is greeting with you like hi , hello etc. Provide comprehensive suggestions based on the context. If context is not enough, ask for more information or search if you dont know , only provide factually correct information and suggestion. The chats need to be professional and helpful.
Context: {context}
Question: {question}

1. Lifestyle Changes:
2. Food Recommendations:
3. Exercise Suggestions:
4. Potential Medicines:

Helpful Answer:
"""
prompt_template = PromptTemplate.from_template(template=template)

set_ret = RunnableParallel(
    {"context": ret, "question": RunnablePassthrough()} 
)

rag_chain = set_ret |  prompt_template | llm | StrOutputParser()




chain  = rag_chain


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
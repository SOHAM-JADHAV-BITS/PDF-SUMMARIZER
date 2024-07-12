import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os


def text_chunks_(content):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(content)
    return chunks


def pdf_text(pdfs):
    contents = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            contents += page.extract_text()
    return contents


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key="")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(api_key="")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def userinput(user_prompt):
    response = st.session_state.conversation({'question': user_prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**You:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")


def main():
    st.set_page_config(page_title="PDF Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Chatbot")
    user_question = st.text_input("Enter your query")
    if user_question and st.session_state.conversation is not None:
        userinput(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = pdf_text(pdf_docs)
                text_chunks = text_chunks_(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()

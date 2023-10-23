import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.vectorstores.base import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chatui import css,user_template,bot_template
from streamlit_extras.add_vertical_space import add_vertical_space

OPENAI_API_KEY=st.secrets['OPEN_API_KEY']

def get_pdf_text(pdf_docs):
    title = []
    text_docs = []
    for pdf in pdf_docs:
        #st.write(pdf)
        title.append(pdf.name[:-4])
        #st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_docs.append(text)
    #st.write(text_docs,title)
    return title,text_docs

def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for text in texts:
        chunks.append(text_splitter.split_text(text))
    # for i in range(len(chunks)):
    #     st.write(chunks[i])
    return chunks

def get_vectorstore(titles,text_chunks):
    it_titles = iter(titles)
    title = next(it_titles)
    if os.path.exists(f'embeddings\{title}.pkl'):
        with open(f'embeddings\{title}.pkl','rb') as f:
            vectorstore = pickle.load(f)
        #     st.write(vectorstore,type(vectorstore))
        # st.write(f"Embeddings {title} loaded!")
    else:
        index = titles.index(title)
        embeddings = OpenAIEmbeddings()
        vectorstore = faiss.FAISS.from_texts(texts=text_chunks[index],embedding=embeddings)
        # st.write(f"Embeddings {title} loaded!")
        with open(f'embeddings\{title}.pkl','wb') as f:
            pickle.dump(vectorstore,f)
    
    for title in it_titles:
        #dv = pd.Dataframe()
        if os.path.exists(f'\embeddings\{title}.pkl'):
            with open(f'\embeddings\{title}.pkl','rb') as f:
                dv = pickle.load(f)
            #     st.write(dv,type(dv))
            # st.write(f"Embeddings {title} loaded!")
        else:
            index = titles.index(title)
            embeddings = OpenAIEmbeddings()
            dv = faiss.FAISS.from_texts(texts=text_chunks[index],embedding=embeddings)
            # st.write(f"Embeddings {title} loaded!")
            with open(f'embeddings\{title}.pkl','wb') as f:
                pickle.dump(dv,f)
        vectorstore.merge_from(dv)
        # st.write(f"Embeddings {title} merged!")
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # st.write(response)
    for i,chat in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",chat.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",chat.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with PDFs',page_icon='books:')
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None


    st.header('Chat with your PDFs :books:')
    user_question = st.text_input('Ask your Questions!')
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your PDFs here ',accept_multiple_files=True)
        if st. button('Start the Process'):
            #st.write(pdf_docs[0].name)
            with st.spinner('Processing'):
                #get the pdf text
                title,text_docs = get_pdf_text(pdf_docs)
             
                #divide the pdf text into text chunks
                text_chunks = get_text_chunks(text_docs)
                # st.write(text_chunks)

                #create vectorstores
                #https://stackoverflow.com/questions/76258587/combine-vectore-store-into-langchain-toolkit#:~:text=If%20you%20want%20to%20combine,a%20method%20for%20merging%20databases
                vectorstore = get_vectorstore(title,text_chunks)
                # st.write(title,vectorstore)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        add_vertical_space(5)
        st.write('Made with ❤️ by [Joel John](https://www.linkedin.com/in/joeljohn29082002/)')

if __name__ == '__main__':
    main()

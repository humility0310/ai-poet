__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=["pdf"])
st.write("---")

def pdf_to_documents(uploaded_file) :
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드된 파일 처리
if uploaded_file is not None:
    pages = pdf_to_documents(uploaded_file)

    #Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    #Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        # With the 'text-embedding-3' class
        # of models, you can specify the size
        # of the embeddings you want returned.
        #dimensions=1024
    )

    #캐시 삭제
    import chromadb
    chromadb.api.client.SharedClient.clear_system_cache()

    #Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)

    #User Input
    st.subheader("PDF에게 질문해보세요.!!")
    question = st.text_input("질문을 입력하세요.")

    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            #Retriever
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever = db.as_retriever()
                , llm = llm
            )

            prompt = hub.pull("rlm/rag-prompt")

            # Generate
            def format_docs(docs) :
                return "\n\n".join(doc.page_content for doc in docs)
            
            reg_chain=(
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            #Question
            result = reg_chain.invoke(question)
            # print(result)
            st.write(result)
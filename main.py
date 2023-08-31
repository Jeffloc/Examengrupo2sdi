import os

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores import USearch
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st

os.environ['OPENAI_API_KEY'] = ''
default_doc_name = 'Documento-de-examen-Grupo1%20.html'


def process_doc(
        path: str = 'C:/Users/Jeff/Downloads/Documento-de-examen-Grupo1%20.html',
        is_local: bool = False,
        question: str = 'Cuáles son los autores del html?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), UnstructuredHTMLLoader(
        f"./{default_doc_name}") if not is_local \
        else UnstructuredHTMLLoader(path)

    doc = loader.load_and_split()

    print(doc[-1])

    # Guardar los embedding
    Guardar = USearch.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='map_reduce', retriever=Guardar.as_retriever())

    st.write(qa.run(question))
    # print(qa.run(question))

#f
def client():
    st.title('Cargar tu archivo HTML fast')
    uploader = st.file_uploader('Ingresa el html', type='html')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HML Guardado con exito!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el html',
                             placeholder='Give response about your HTML', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default html')
            process_doc()


if __name__ == '__main__':
    client()
    # process_doc()

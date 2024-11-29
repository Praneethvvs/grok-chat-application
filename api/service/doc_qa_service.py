import os
import pickle
from io import BytesIO

import boto3
from botocore.exceptions import NoCredentialsError
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader


class DocQaProcessor:

    def __init__(self, boto_session, openai_client, openai_embeddings) -> None:
        self.boto_session = boto_session
        self.openai_client = openai_client
        self.embeddings = openai_embeddings

    def text_splitter(self, pdf_content):
        stream = BytesIO(pdf_content)
        pdf_reader = PdfReader(stream)
        text = ""
        for page in pdf_reader.pages:
            print("extracting page text")
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        return chunks

    def get_s3_file(self, bucket_name, file_name):
        s3 = self.boto_session.client("s3")
        try:
            s3_response = s3.get_object(
                Bucket=bucket_name, Key=f"openai_pdf_files/{file_name}.pdf"
            )
            return s3_response["Body"].read()
        except NoCredentialsError:
            raise ValueError("File doesnt exist in s3, please give a valid file")

    def store_embeddings(self, chunks, docname):
        try:

            knowledgeBase = FAISS.from_texts(chunks, embedding=self.embeddings)
            knowledgeBase.save_local(f"faiss_index/{docname}")
        except Exception as e:
            print(f"An error occurred while saving embeddings: {e}")
            raise

    def retrieve_embeddings(self, docname):
        vector_store = FAISS.load_local(
            f"faiss_index/{docname}",
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store

    def is_knowledge_base_exists(self, filename):
        return os.path.exists(f"faiss_index/{filename}")

    def get_response_llm(self, knowledgeBase_refined, query):
        chain = load_qa_chain(self.openai_client, chain_type="stuff")
        with get_openai_callback() as cost:
            response = chain.run(input_documents=knowledgeBase_refined, question=query)
            print(f"cost for the query {query} is  {cost}")
        return response

    def process_doc_query(self, docname, query):

        if self.is_knowledge_base_exists(docname):
            print(f"Knowledge base exists locally for ===> {docname}")

        else:
            print(
                f"No Knowledge base exists locally for ===> {docname}, creating one locally"
            )
            try:
                file_content = self.get_s3_file(
                    bucket_name="pvvs-bucket-exp-new", file_name=docname
                )
            except:
                return "No such file exists on s3, please make sure the file exists"
            chunks = self.text_splitter(file_content)
            self.store_embeddings(chunks, docname)

        knowledgebase = self.retrieve_embeddings(docname)

        knowledgebase_refined = knowledgebase.similarity_search(query=query, k=3)
        query_response = self.get_response_llm(knowledgebase_refined, query)

        return query_response

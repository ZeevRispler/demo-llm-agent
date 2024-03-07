# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vector DB for the jewelry agent demo. Currently, works in-memory.
"""
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# TODO: for the items images search: images_retriever = None
#: Global in-memory additional information retriever.
documents_retriever = None


def get_documents_retriever(path: str):
    global documents_retriever

    if documents_retriever:
        return documents_retriever

    # we can declare extension, display progress bar, use multithreading
    if os.path.isdir(path):
        loader = DirectoryLoader(path, glob="*.txt")
    else:
        loader = TextLoader(path)

    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # Here is where we add in the fake source information
    for i, doc in enumerate(texts):
        doc.metadata["page_chunk"] = i

    # Create our retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings, collection_name="additional_information")
    documents_retriever = vectorstore.as_retriever()
    return documents_retriever


# if __name__ == "__main__":
#     get_documents_retriever()

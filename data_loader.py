from langchain_openai import AzureOpenAIEmbeddings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
load_dotenv()


reader = PDFReader()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

if not endpoint or not api_key or not embedding_deployment:
    raise ValueError("Missing required Azure OpenAI environment variables")

client = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment=embedding_deployment
)


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
def load_and_chunk_pdf(path:str):
    docs = PDFReader().load_data(file=path)
    text = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for page_text in text:
        chunks.extend(splitter.split_text(page_text))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return client.embed_documents(texts)




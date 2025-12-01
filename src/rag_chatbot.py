from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class SimpleRAGChatbot:

    def __init__(self, text):
        self.text = text
        self._setup()

    def _setup(self):
        """Split text and create vector DB."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_text(self.text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = [Document(page_content=c) for c in chunks]

        self.db = FAISS.from_documents(docs, embeddings)

    def retrieve(self, query):
        """Retrieve relevant context."""
        docs = self.db.similarity_search(query, k=2)
        return [d.page_content for d in docs]

    def answer(self, query):
        """Generate a simple rule-based answer using retrieved context."""
        contexts = self.retrieve(query)

        if len(contexts) == 0:
            return "I could not find relevant information."

        return f"Based on the document: {contexts[0]}"

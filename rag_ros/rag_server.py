# Copyright (c) 2025 José Galeas Merchán
# Copyright (c) 2025 Óscar Pons Fernández
# Copyright (c) 2025 Alberto J. Tudela Roldán
# Copyright (c) 2025 Grupo Avispa, DTE, Universidad de Málaga
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RAG server.

This module provides a RAG server that manages a Chroma vector store using
HuggingFace embeddings for semantic search, allowing storage and retrieval
of text documents.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any

from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGServer:
    """Manage storage and retrieval of text documents in a vector database.

    The server uses Chroma for vector storage and HuggingFace embeddings for
    semantic search. It allows storing documents with metadata and retrieving
    relevant documents based on similarity to a query.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        chroma_directory: str = './chroma_db',
        k: int = 8,
    ) -> None:
        """Initialize server and (attempt to) build the vector index.

        Parameters
        ----------
        logger : Optional[Any]
            Optional ROS2 logger to use for logging (default: None).
        chroma_directory : str
            Directory where Chroma persistence data will be stored.
        k : int
            Number of documents to retrieve by default (default: 8).
        """
        self.chroma_directory = chroma_directory
        self.k = k
        self.vector_db: Optional[Chroma] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        self.logger = logger

        # Create chroma directory if it doesn't exist
        try:
            os.makedirs(self.chroma_directory, exist_ok=True)
            self._log_info(f'Chroma directory ensured at: {self.chroma_directory}')
        except OSError as e:
            self._log_warning(
                f'Cannot create chroma directory {self.chroma_directory}: {e}')

        # Initialize the RAG system
        self._setup_rag()

    def _log_info(self, msg: str) -> None:
        """Log info message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

    def _log_debug(self, msg: str) -> None:
        """Log debug message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            logging.debug(msg)

    def _log_warning(self, msg: str) -> None:
        """Log warning message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            logging.warning(msg)

    def _log_error(self, msg: str) -> None:
        """Log error message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.error(msg)
        else:
            logging.error(msg)

    def _setup_rag(self) -> None:
        """Initialize vector database and retriever."""
        # Load vector store
        try:
            self.vector_db = Chroma(
                persist_directory=self.chroma_directory,
                embedding_function=HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                )
            )
            self._log_info('Vector database loaded successfully')

            # Create retrieval chain
            self.retriever = self.vector_db.as_retriever(
                search_type='similarity',
                search_kwargs={'k': self.k},
            )
            self._log_info('Retriever initialized successfully')

        except (ImportError, RuntimeError, ValueError) as e:
            self._log_error(f'Error creating vector database: {e}')
            return

    def retrieve_documents(self, query: str, k: int = 8) -> str:
        """Return top-k document chunks for `query` as a single string.

        Parameters:
            query (str): The query string to search for.
            k (int): The number of top documents to return.

        Returns:
            str: A formatted string containing the retrieved document chunks.

        Raises:
            ValueError: if the retriever is not initialized. This can happen
                when optional dependencies are missing or when no documents
                were indexed.
        """
        self._log_info(f'Starting document retrieval for query: "{query}"')
        self._log_debug(f'Requested {k} documents')

        if self.retriever is None or self.vector_db is None:
            error_msg = 'Retriever or vector database not initialized.'
            self._log_error(error_msg)
            raise ValueError(error_msg)

        try:
            self._log_info('Searching vector database...')

            # Perform the retrieval
            retriever_with_k = self.vector_db.as_retriever(
                search_kwargs={'k': k}
            )
            docs = retriever_with_k.invoke(query)

            # Generate a formatted text response from the retrieved documents
            results: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                # Include source metadata if available
                source = 'Unknown source'
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown source')

                formatted_doc = {
                    'doc_id': i + 1,
                    'source': source,
                    'content': doc.page_content
                }

                results.append(formatted_doc)

            response = {
                'status': 'success',
                'message': 'Documents retrieved successfully',
                'query': query,
                'total_results': len(results),
                'results': results
            }

            self._log_info('Document retrieval completed successfully')
            self._log_debug(
                f'Successfully retrieved [{response["total_results"]}] documents.'
            )

            return json.dumps(response, indent=2, ensure_ascii=False)

        # Handle errors during retrieval (empty JSON is returned)
        except (ValueError, RuntimeError, AttributeError) as e:
            error_msg = f'Error during document retrieval: {e}'
            self._log_error(error_msg)
            empty_response = {
                'status': 'success',
                'message': 'No relevant documents found for your question.',
                'query': query,
                'total_results': 0,
                'results': []
            }
            return json.dumps(empty_response, indent=2, ensure_ascii=False)

    def get_system_stats(self) -> dict:
        """Get statistics about the RAG system.

        Returns:
            dict: System statistics including document count and retriever status.
        """
        total_documents = 0
        if self.vector_db is not None:
            db_info = self.vector_db.get()
            if db_info and 'ids' in db_info:
                total_documents = len(db_info['ids'])

        return {
            'total_documents': total_documents,
            'chroma_directory': self.chroma_directory,
            'retriever_initialized': self.retriever is not None,
            'vector_db_initialized': self.vector_db is not None,
        }

    def store_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a text document in the vector database.

        Parameters:
            text (str): The text content to store.
            metadata (Optional[Dict[str, Any]]): Optional metadata associated with the document.

        Returns:
            bool: True if the document was stored successfully, False otherwise.
        """
        if self.vector_db is None:
            error_msg = 'Vector database not initialized.'
            self._log_error(error_msg)
            return False

        if not text or not text.strip():
            error_msg = 'Cannot store empty document text.'
            self._log_error(error_msg)
            return False

        try:
            # Create a Document object with the provided text and metadata
            doc_metadata = {}
            if metadata and isinstance(metadata, dict):
                doc_metadata = metadata
            else:
                # If no metadata provided, add a default source identifier
                doc_metadata = {'source': 'stored_document'}

            self._log_debug(f'Storing document with metadata: {doc_metadata}')

            document = Document(page_content=text, metadata=doc_metadata)

            # Split the document into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs_chunked = splitter.split_documents([document])

            # Add the chunks to the vector database
            self.vector_db.add_documents(docs_chunked)

            self._log_info(
                f'Stored document with {len(docs_chunked)} chunks and metadata: {doc_metadata}')
            return True

        except (ValueError, AttributeError) as e:
            error_msg = f'Error storing document: {e}'
            self._log_warning(error_msg)
            return False

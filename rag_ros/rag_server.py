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
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
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
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        top_k: int = 8,
        use_hybrid_search: bool = True,
    ) -> None:
        """Initialize server and (attempt to) build the vector index.

        Parameters
        ----------
        logger : Optional[Any]
            Optional ROS2 logger to use for logging (default: None).
        chroma_directory : str
            Directory where Chroma persistence data will be stored.
        embedding_model : str
            HuggingFace embedding model name to use (default:
            sentence-transformers/all-MiniLM-L6-v2).
        top_k : int
            Number of documents to retrieve by default (default: 8).
        use_hybrid_search : bool
            Whether to use hybrid search (BM25 + semantic) (default: True).
        """
        self.chroma_directory = chroma_directory
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.vector_db: Optional[Chroma] = None
        self.vector_retriever: Optional[VectorStoreRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
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
                    model_name=self.embedding_model,
                )
            )
            self._log_info('Vector database loaded successfully')
            self._log_info(f'Using embedding model: {self.embedding_model}')

            # Create retrieval chain
            self.vector_retriever = self.vector_db.as_retriever(
                search_type='similarity',
                search_kwargs={'k': self.top_k},
            )
            self._log_info('Retriever initialized successfully')

            # Initialize BM25 retriever if hybrid search is enabled
            if self.use_hybrid_search:
                self._setup_bm25_retriever()

        except (ImportError, RuntimeError, ValueError) as e:
            self._log_error(f'Error creating vector database: {e}')
            return

    def _setup_bm25_retriever(self) -> None:
        """Initialize BM25 and ensemble retrievers for hybrid search.

        Creates a BM25 retriever from documents in the vector database and combines
        it with the semantic retriever using EnsembleRetriever.
        """
        if self.vector_db is None:
            self._log_warning('Cannot setup BM25 retriever: vector_db not initialized')
            return

        try:
            # Get all documents from the vector database
            db_info = self.vector_db.get()
            if not db_info or 'documents' not in db_info or not db_info['documents']:
                self._log_warning('No documents found in vector database for BM25')
                return

            # Create Document objects from the stored data
            docs: List[Document] = []
            for i, doc_text in enumerate(db_info['documents']):
                metadata: dict[str, Any] = {}
                if 'metadatas' in db_info and i < len(db_info['metadatas']):
                    metadata = db_info['metadatas'][i] or {}
                docs.append(Document(page_content=doc_text, metadata=metadata))

            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self._log_info(f'BM25 retriever initialized with {len(docs)} documents')

            # Create ensemble retriever combining semantic and BM25 search
            if self.vector_retriever is not None:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.5, 0.5],  # Equal weight for both retrievers
                )
                self._log_info(
                    'Ensemble retriever created (50% semantic + 50% BM25)')

        except (ValueError, AttributeError, RuntimeError) as e:
            self._log_warning(f'Error setting up BM25/ensemble retriever: {e}')
            self.bm25_retriever = None
            self.ensemble_retriever = None

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a Chroma where filter from metadata filter dictionary.

        Parameters:
            filters (Dict[str, Any]): Dictionary with filter criteria.
                Supported keys: 'source', 'node_name', 'node_function', 'log_level'.

        Returns:
            Optional[Dict[str, Any]]: A where filter dictionary for Chroma, or None.
        """
        where_conditions = []
        supported_keys = ['source', 'node_name', 'node_function', 'log_level']

        for key, value in filters.items():
            if key not in supported_keys:
                self._log_warning(f'Unsupported filter key: {key}')
                continue

            if isinstance(value, list):
                # If value is a list, create an OR condition
                or_conditions = [{'$eq': [f'${key}', v]} for v in value]
                where_conditions.append({'$or': or_conditions})
            else:
                # Single value equality check
                where_conditions.append({key: {'$eq': value}})  # type: ignore[dict-item]

        if not where_conditions:
            return None

        # If only one condition, return it directly; otherwise use $and
        if len(where_conditions) == 1:
            return where_conditions[0]
        else:
            return {'$and': where_conditions}

    def _filter_documents_by_metadata(
        self,
        documents: List[Document],
        filters: Dict[str, Any],
    ) -> List[Document]:
        """Filter documents by metadata criteria.

        Parameters:
            documents (List[Document]): Documents to filter.
            filters (Dict[str, Any]): Filter criteria.
                Supported keys: 'source', 'node_name', 'node_function', 'log_level'.

        Returns:
            List[Document]: Filtered documents matching all criteria.
        """
        supported_keys = ['source', 'node_name', 'node_function', 'log_level']
        filtered_docs = documents

        for key, value in filters.items():
            if key not in supported_keys:
                self._log_warning(f'Unsupported filter key: {key}')
                continue

            # Apply filter for this key
            if isinstance(value, list):
                # If value is a list, keep documents where metadata[key] is in list
                filtered_docs = [
                    doc for doc in filtered_docs
                    if self._doc_matches_filter(doc, key, value, is_list=True)
                ]
            else:
                # Single value: keep documents where metadata[key] equals value
                filtered_docs = [
                    doc for doc in filtered_docs
                    if self._doc_matches_filter(doc, key, value, is_list=False)
                ]

        return filtered_docs

    def _doc_matches_filter(
        self,
        doc: Document,
        key: str,
        value: Any,
        is_list: bool = False,
    ) -> bool:
        """Check if a document matches a filter criterion.

        Parameters:
            doc (Document): Document to check.
            key (str): Metadata key to filter by.
            value (Any): Value(s) to match.
            is_list (bool): Whether value is a list for OR matching.

        Returns:
            bool: True if document matches the filter criterion.
        """
        if not hasattr(doc, 'metadata') or not doc.metadata:
            return False

        doc_value = doc.metadata.get(key)
        if is_list:
            return doc_value in value
        else:
            return doc_value == value

    def retrieve_documents(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return top-k document chunks for `query` as a single string.

        Uses hybrid search combining semantic similarity and keyword matching (BM25)
        via EnsembleRetriever when enabled, providing more comprehensive results.

        Parameters:
            query (str): The query string to search for.
            k (int): The number of top documents to return.
            filters (Optional[Dict[str, Any]]): Optional metadata filters to apply.
                Supported filters: 'source', 'node_name', 'node_function', 'log_level'.

        Returns:
            str: A formatted string containing the retrieved document chunks.

        Raises:
            ValueError: if the retriever is not initialized. This can happen
                when optional dependencies are missing or when no documents
                were indexed.
        """
        self._log_info(f'Starting document retrieval for query: "{query}"')
        self._log_debug(f'Requested {k} documents')

        if self.vector_retriever is None or self.vector_db is None:
            error_msg = 'Retriever or vector database not initialized.'
            self._log_error(error_msg)
            raise ValueError(error_msg)

        try:
            self._log_info('Searching vector database...')

            # Use ensemble retriever if available, otherwise use semantic search only
            if self.use_hybrid_search and self.ensemble_retriever is not None:
                self._log_debug('Using ensemble search (semantic + BM25)')
                docs = self.ensemble_retriever.invoke(query)
                # Limit results to k documents
                docs = docs[:k]
                # Apply metadata filtering to ensemble results if filters provided
                if filters is not None:
                    self._log_debug(f'Applying metadata filters: {filters}')
                    docs = self._filter_documents_by_metadata(docs, filters)
                    # Ensure we have at least k documents after filtering
                    if len(docs) < k and len(docs) > 0:
                        self._log_debug(
                            f'Filtering reduced results from {len(docs)} to {len(docs)}')
            else:
                self._log_debug('Using semantic search only')
                # Perform semantic search with optional filtering
                search_kwargs = {'k': k}
                if filters:
                    self._log_debug(f'Applying metadata filters: {filters}')
                    where_filter = self._build_where_filter(filters)
                    if where_filter:
                        search_kwargs['where'] = where_filter  # type: ignore[assignment]
                        self._log_debug(f'Search kwargs: {search_kwargs}')

                retriever_with_k = self.vector_db.as_retriever(
                    search_kwargs=search_kwargs
                )
                docs = retriever_with_k.invoke(query)

            # Generate a formatted text response from the retrieved documents
            results: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                # Include source metadata if available
                source = 'Unknown source'
                node_name = 'Unknown node'
                node_function = 'Unknown function'
                log_level = 'UNKNOWN'

                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown source')
                    node_name = doc.metadata.get('node_name', 'Unknown node')
                    node_function = doc.metadata.get('node_function', 'Unknown function')
                    log_level = doc.metadata.get('log_level', 'UNKNOWN')

                formatted_doc = {
                    'id': i + 1,
                    'source': source,
                    'node_name': node_name,
                    'node_function': node_function,
                    'log_level': log_level,
                    'content': doc.page_content
                }

                results.append(formatted_doc)

            response = {
                'status': 'success',
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
            'retriever_initialized': self.vector_retriever is not None,
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

            # Update BM25 retriever if hybrid search is enabled
            if self.use_hybrid_search:
                self._setup_bm25_retriever()

            return True

        except (ValueError, AttributeError) as e:
            error_msg = f'Error storing document: {e}'
            self._log_warning(error_msg)
            return False

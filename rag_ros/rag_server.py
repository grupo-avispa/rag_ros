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

# Default configuration constants
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_TOP_K = 8
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_ENSEMBLE_WEIGHT_SEMANTIC = 0.5
DEFAULT_ENSEMBLE_WEIGHT_BM25 = 0.5

# Supported metadata filter keys
SUPPORTED_FILTER_KEYS = ['source', 'node_name', 'node_function', 'log_level']


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
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
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
            HuggingFace embedding model name to use.
        top_k : int
            Number of documents to retrieve by default.
        use_hybrid_search : bool
            Whether to use hybrid search (BM25 + semantic) (default: True).
        """
        self.logger = logger
        self.chroma_directory = chroma_directory
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search

        # Initialize retrievers as None
        self.vector_db: Optional[Chroma] = None
        self.vector_retriever: Optional[VectorStoreRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        # Setup system
        self._ensure_chroma_directory()
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

    def _ensure_chroma_directory(self) -> None:
        """Ensure Chroma directory exists.

        Creates the directory if it doesn't exist and logs the operation.
        """
        try:
            os.makedirs(self.chroma_directory, exist_ok=True)
            self._log_info(f'Chroma directory ensured at: {self.chroma_directory}')
        except OSError as e:
            self._log_warning(
                f'Cannot create chroma directory {self.chroma_directory}: {e}')

    def _create_embedding_function(self) -> HuggingFaceEmbeddings:
        """Create HuggingFace embeddings function.

        Returns
        -------
        HuggingFaceEmbeddings
            Configured embedding function.
        """
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

    def _create_vector_retriever(self) -> Optional[VectorStoreRetriever]:
        """Create vector store retriever with configured parameters.

        Returns
        -------
        Optional[VectorStoreRetriever]
            Configured retriever or None if vector_db is not initialized.
        """
        if self.vector_db is None:
            return None

        return self.vector_db.as_retriever(
            search_type='similarity',
            search_kwargs={'k': self.top_k},
        )

    def _setup_rag(self) -> None:
        """Initialize vector database and retrievers."""
        try:
            self._initialize_vector_database()
            self._initialize_retrievers()
        except (ImportError, RuntimeError, ValueError) as e:
            self._log_error(f'Error setting up RAG system: {e}')

    def _initialize_vector_database(self) -> None:
        """Initialize Chroma vector database.

        Raises
        ------
        ImportError, RuntimeError, ValueError
            If vector database initialization fails.
        """
        self.vector_db = Chroma(
            persist_directory=self.chroma_directory,
            embedding_function=self._create_embedding_function(),
        )
        self._log_info('Vector database loaded successfully')
        self._log_info(f'Using embedding model: {self.embedding_model}')

    def _initialize_retrievers(self) -> None:
        """Initialize vector and hybrid retrievers."""
        self.vector_retriever = self._create_vector_retriever()
        if self.vector_retriever is not None:
            self._log_info('Retriever initialized successfully')

        if self.use_hybrid_search:
            self._setup_bm25_retriever()

    def _setup_bm25_retriever(self) -> None:
        """Initialize BM25 and ensemble retrievers for hybrid search.

        Creates a BM25 retriever from documents in the vector database and combines
        it with the semantic retriever using EnsembleRetriever.
        """
        if self.vector_db is None:
            self._log_warning('Cannot setup BM25 retriever: vector_db not initialized')
            return

        try:
            docs = self._get_all_documents_from_db()
            if not docs:
                self._log_warning('No documents found in vector database for BM25')
                return

            self._create_bm25_retriever(docs)
            self._create_ensemble_retriever()

        except (ValueError, AttributeError, RuntimeError) as e:
            self._log_warning(f'Error setting up BM25/ensemble retriever: {e}')
            self.bm25_retriever = None
            self.ensemble_retriever = None

    def _get_all_documents_from_db(self) -> List[Document]:
        """Retrieve all documents from vector database.

        Returns
        -------
        List[Document]
            List of Document objects from the database.
        """
        if self.vector_db is None:
            return []

        db_info = self.vector_db.get()
        if not db_info or 'documents' not in db_info or not db_info['documents']:
            return []

        docs: List[Document] = []
        for i, doc_text in enumerate(db_info['documents']):
            metadata: dict[str, Any] = {}
            if 'metadatas' in db_info and i < len(db_info['metadatas']):
                metadata = db_info['metadatas'][i] or {}
            docs.append(Document(page_content=doc_text, metadata=metadata))

        return docs

    def _create_bm25_retriever(self, docs: List[Document]) -> None:
        """Create BM25 retriever from documents.

        Parameters
        ----------
        docs : List[Document]
            Documents to index in BM25 retriever.
        """
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self._log_info(f'BM25 retriever initialized with {len(docs)} documents')

    def _create_ensemble_retriever(self) -> None:
        """Create ensemble retriever combining semantic and BM25 search."""
        if self.vector_retriever is None or self.bm25_retriever is None:
            return

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[DEFAULT_ENSEMBLE_WEIGHT_SEMANTIC, DEFAULT_ENSEMBLE_WEIGHT_BM25],
        )
        self._log_info(
            f'Ensemble retriever created ({int(DEFAULT_ENSEMBLE_WEIGHT_SEMANTIC * 100)}% '
            f'semantic + {int(DEFAULT_ENSEMBLE_WEIGHT_BM25 * 100)}% BM25)')

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a Chroma where filter from metadata filter dictionary.

        Parameters
        ----------
        filters : Dict[str, Any]
            Dictionary with filter criteria.
            Supported keys: 'source', 'node_name', 'node_function', 'log_level'.

        Returns
        -------
        Optional[Dict[str, Any]]
            A where filter dictionary for Chroma, or None if no valid filters.
        """
        where_conditions = []

        for key, value in filters.items():
            if not self._is_valid_filter_key(key):
                continue

            condition = self._create_filter_condition(key, value)
            if condition:
                where_conditions.append(condition)

        return self._combine_filter_conditions(where_conditions)

    def _is_valid_filter_key(self, key: str) -> bool:
        """Check if filter key is supported.

        Parameters
        ----------
        key : str
            Filter key to validate.

        Returns
        -------
        bool
            True if key is supported, False otherwise.
        """
        if key not in SUPPORTED_FILTER_KEYS:
            self._log_warning(f'Unsupported filter key: {key}')
            return False
        return True

    def _create_filter_condition(self, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """Create filter condition for a key-value pair.

        Parameters
        ----------
        key : str
            Metadata key to filter by.
        value : Any
            Value(s) to match (can be single value or list).

        Returns
        -------
        Optional[Dict[str, Any]]
            Filter condition dictionary.
        """
        if isinstance(value, list):
            or_conditions = [{'$eq': [f'${key}', v]} for v in value]
            return {'$or': or_conditions}
        else:
            return {key: {'$eq': value}}  # type: ignore[dict-item]

    def _combine_filter_conditions(
        self,
        conditions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Combine multiple filter conditions.

        Parameters
        ----------
        conditions : List[Dict[str, Any]]
            List of filter conditions to combine.

        Returns
        -------
        Optional[Dict[str, Any]]
            Combined filter or None if no conditions.
        """
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {'$and': conditions}

    def _filter_documents_by_metadata(
        self,
        documents: List[Document],
        filters: Dict[str, Any],
    ) -> List[Document]:
        """Filter documents by metadata criteria.

        Parameters
        ----------
        documents : List[Document]
            Documents to filter.
        filters : Dict[str, Any]
            Filter criteria.
            Supported keys: 'source', 'node_name', 'node_function', 'log_level'.

        Returns
        -------
        List[Document]
            Filtered documents matching all criteria.
        """
        filtered_docs = documents

        for key, value in filters.items():
            if not self._is_valid_filter_key(key):
                continue
            filtered_docs = self._apply_single_filter(filtered_docs, key, value)

        return filtered_docs

    def _apply_single_filter(
        self,
        documents: List[Document],
        key: str,
        value: Any,
    ) -> List[Document]:
        """Apply a single filter to documents.

        Parameters
        ----------
        documents : List[Document]
            Documents to filter.
        key : str
            Metadata key to filter by.
        value : Any
            Value(s) to match.

        Returns
        -------
        List[Document]
            Filtered documents.
        """
        is_list = isinstance(value, list)
        return [
            doc for doc in documents
            if self._doc_matches_filter(doc, key, value, is_list=is_list)
        ]

    def _doc_matches_filter(
        self,
        doc: Document,
        key: str,
        value: Any,
        is_list: bool = False,
    ) -> bool:
        """Check if a document matches a filter criterion.

        Parameters
        ----------
        doc : Document
            Document to check.
        key : str
            Metadata key to filter by.
        value : Any
            Value(s) to match.
        is_list : bool
            Whether value is a list for OR matching (default: False).

        Returns
        -------
        bool
            True if document matches the filter criterion.
        """
        if not hasattr(doc, 'metadata') or not doc.metadata:
            return False

        doc_value = doc.metadata.get(key)
        return doc_value in value if is_list else doc_value == value

    def retrieve_documents(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return top-k document chunks for `query` as a single string.

        Uses hybrid search combining semantic similarity and keyword matching (BM25)
        via EnsembleRetriever when enabled, providing more comprehensive results.

        Parameters
        ----------
        query : str
            The query string to search for.
        k : int
            The number of top documents to return.
        filters : Optional[Dict[str, Any]]
            Optional metadata filters to apply.
            Supported filters: 'source', 'node_name', 'node_function', 'log_level'.

        Returns
        -------
        str
            A formatted JSON string containing the retrieved document chunks.

        Raises
        ------
        ValueError
            If the retriever is not initialized.
        """
        self._log_info(f'Starting document retrieval for query: "{query}"')
        self._log_debug(f'Requested {k} documents')

        self._validate_retrievers()

        try:
            docs = self._perform_search(query, k, filters)
            response = self._format_search_response(query, docs)

            self._log_info('Document retrieval completed successfully')
            self._log_debug(f'Successfully retrieved [{len(docs)}] documents.')

            return json.dumps(response, indent=2, ensure_ascii=False)

        except (ValueError, RuntimeError, AttributeError) as e:
            return self._handle_retrieval_error(query, e)

    def _validate_retrievers(self) -> None:
        """Validate that retrievers are initialized.

        Raises
        ------
        ValueError
            If retrievers are not initialized.
        """
        if self.vector_retriever is None or self.vector_db is None:
            error_msg = 'Retriever or vector database not initialized.'
            self._log_error(error_msg)
            raise ValueError(error_msg)

    def _perform_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Perform document search using configured retrievers.

        Parameters
        ----------
        query : str
            Search query.
        k : int
            Number of documents to retrieve.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply.

        Returns
        -------
        List[Document]
            Retrieved documents.
        """
        self._log_info('Searching vector database...')

        if self.use_hybrid_search and self.ensemble_retriever is not None:
            return self._hybrid_search(query, k, filters)
        else:
            return self._semantic_search(query, k, filters)

    def _hybrid_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Perform hybrid search using ensemble retriever.

        Parameters
        ----------
        query : str
            Search query.
        k : int
            Number of documents to retrieve.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply.

        Returns
        -------
        List[Document]
            Retrieved and filtered documents.
        """
        self._log_debug('Using ensemble search (semantic + BM25)')
        docs = self.ensemble_retriever.invoke(query)[:k]  # type: ignore[union-attr]

        if filters is not None:
            self._log_debug(f'Applying metadata filters: {filters}')
            docs = self._filter_documents_by_metadata(docs, filters)

        return docs

    def _semantic_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Perform semantic search using vector retriever.

        Parameters
        ----------
        query : str
            Search query.
        k : int
            Number of documents to retrieve.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply.

        Returns
        -------
        List[Document]
            Retrieved documents.
        """
        self._log_debug('Using semantic search only')
        search_kwargs = self._build_search_kwargs(k, filters)

        retriever_with_k = self.vector_db.as_retriever(  # type: ignore[union-attr]
            search_kwargs=search_kwargs
        )
        return retriever_with_k.invoke(query)

    def _build_search_kwargs(
        self,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build search kwargs for vector retriever.

        Parameters
        ----------
        k : int
            Number of documents to retrieve.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply.

        Returns
        -------
        Dict[str, Any]
            Search kwargs dictionary.
        """
        search_kwargs: Dict[str, Any] = {'k': k}

        if filters:
            self._log_debug(f'Applying metadata filters: {filters}')
            where_filter = self._build_where_filter(filters)
            if where_filter:
                search_kwargs['where'] = where_filter
                self._log_debug(f'Search kwargs: {search_kwargs}')

        return search_kwargs

    def _format_search_response(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """Format search results as response dictionary.

        Parameters
        ----------
        query : str
            Original search query.
        documents : List[Document]
            Retrieved documents.

        Returns
        -------
        Dict[str, Any]
            Formatted response dictionary.
        """
        results = [self._format_document(i, doc) for i, doc in enumerate(documents)]

        return {
            'status': 'success',
            'query': query,
            'total_results': len(results),
            'results': results
        }

    def _format_document(self, index: int, doc: Document) -> Dict[str, Any]:
        """Format a single document for response.

        Parameters
        ----------
        index : int
            Document index.
        doc : Document
            Document to format.

        Returns
        -------
        Dict[str, Any]
            Formatted document dictionary.
        """
        metadata = self._extract_document_metadata(doc)

        return {
            'id': index + 1,
            'source': metadata['source'],
            'node_name': metadata['node_name'],
            'node_function': metadata['node_function'],
            'log_level': metadata['log_level'],
            'content': doc.page_content
        }

    def _extract_document_metadata(self, doc: Document) -> Dict[str, str]:
        """Extract metadata from document with defaults.

        Parameters
        ----------
        doc : Document
            Document to extract metadata from.

        Returns
        -------
        Dict[str, str]
            Metadata dictionary with default values.
        """
        defaults = {
            'source': 'Unknown source',
            'node_name': 'Unknown node',
            'node_function': 'Unknown function',
            'log_level': 'UNKNOWN'
        }

        if not hasattr(doc, 'metadata') or not doc.metadata:
            return defaults

        return {
            'source': doc.metadata.get('source', defaults['source']),
            'node_name': doc.metadata.get('node_name', defaults['node_name']),
            'node_function': doc.metadata.get('node_function', defaults['node_function']),
            'log_level': doc.metadata.get('log_level', defaults['log_level'])
        }

    def _handle_retrieval_error(self, query: str, error: Exception) -> str:
        """Handle retrieval error and return empty response.

        Parameters
        ----------
        query : str
            Original search query.
        error : Exception
            Exception that occurred.

        Returns
        -------
        str
            JSON string with empty results.
        """
        self._log_error(f'Error during document retrieval: {error}')
        empty_response = {
            'status': 'success',
            'query': query,
            'total_results': 0,
            'results': []
        }
        return json.dumps(empty_response, indent=2, ensure_ascii=False)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system.

        Returns
        -------
        Dict[str, Any]
            System statistics including document count and retriever status.
        """
        return {
            'total_documents': self._get_total_documents(),
            'chroma_directory': self.chroma_directory,
            'embedding_model': self.embedding_model,
            'top_k': self.top_k,
            'use_hybrid_search': self.use_hybrid_search,
            'vector_db_initialized': self.vector_db is not None,
            'vector_retriever_initialized': self.vector_retriever is not None,
            'bm25_retriever_initialized': self.bm25_retriever is not None,
            'ensemble_retriever_initialized': self.ensemble_retriever is not None,
        }

    def _get_total_documents(self) -> int:
        """Get total number of documents in vector database.

        Returns
        -------
        int
            Total document count.
        """
        if self.vector_db is None:
            return 0

        db_info = self.vector_db.get()
        if db_info and 'ids' in db_info:
            return len(db_info['ids'])

        return 0

    def store_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a text document in the vector database.

        Parameters
        ----------
        text : str
            The text content to store.
        metadata : Optional[Dict[str, Any]]
            Optional metadata associated with the document.

        Returns
        -------
        bool
            True if the document was stored successfully, False otherwise.
        """
        if not self._validate_storage_preconditions(text):
            return False

        try:
            doc_metadata = self._prepare_metadata(metadata)
            self._log_debug(f'Storing document with metadata: {doc_metadata}')

            document = Document(page_content=text, metadata=doc_metadata)
            docs_chunked = self._split_document(document)

            self.vector_db.add_documents(docs_chunked)  # type: ignore[union-attr]

            self._log_info(
                f'Stored document with {len(docs_chunked)} chunks '
                f'and metadata: {doc_metadata}'
            )

            if self.use_hybrid_search:
                self._setup_bm25_retriever()

            return True

        except (ValueError, AttributeError) as e:
            self._log_warning(f'Error storing document: {e}')
            return False

    def _validate_storage_preconditions(self, text: str) -> bool:
        """Validate preconditions for storing a document.

        Parameters
        ----------
        text : str
            Document text to validate.

        Returns
        -------
        bool
            True if preconditions are met, False otherwise.
        """
        if self.vector_db is None:
            self._log_error('Vector database not initialized.')
            return False

        if not text or not text.strip():
            self._log_error('Cannot store empty document text.')
            return False

        return True

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare metadata dictionary with defaults if needed.

        Parameters
        ----------
        metadata : Optional[Dict[str, Any]]
            Optional metadata dictionary.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with defaults if input was None.
        """
        if metadata and isinstance(metadata, dict):
            return metadata
        return {'source': 'stored_document'}

    def _split_document(self, document: Document) -> List[Document]:
        """Split document into chunks.

        Parameters
        ----------
        document : Document
            Document to split.

        Returns
        -------
        List[Document]
            List of document chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
        return splitter.split_documents([document])

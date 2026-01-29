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

"""RAG Service Node for ROS2."""

import json
from typing import Any, Dict, Optional

import rclpy
from rcl_interfaces.msg import Log
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from llm_interactions_msgs.msg import Document, Metadata
from llm_interactions_msgs.srv import (
    RetrieveDocuments, RetrieveDocuments_Request, RetrieveDocuments_Response,
    StoreDocument, StoreDocument_Request, StoreDocument_Response
)
from rag_ros.rag_server import RAGServer

# Log level mapping
LOG_LEVEL_NAMES = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARN',
    40: 'ERROR',
    50: 'FATAL',
}


class RAGService(Node):
    """ROS2 service node for RAG (Retrieval-Augmented Generation) operations.

    This node provides two main services:
    - RetrieveDocuments: Retrieve relevant documents based on a query
    - StoreDocument: Store a new document in the RAG system

    The RAG system uses LangChain with Chroma for vector storage and
    HuggingFace embeddings for semantic search.
    """

    def __init__(self) -> None:
        """Initialize the RAG Service node."""
        super().__init__('rag_node')

        # Initialize parameters
        self._declare_and_get_parameters()

        # Initialize RAG server
        self.rag_server = self._initialize_rag_server()

        # Setup ROS2 interfaces
        self.group = ReentrantCallbackGroup()
        self._create_services()
        self._create_subscriptions()

        self.get_logger().info('RAG service node has been started.')

    def _declare_and_get_parameters(self) -> None:
        """Declare and retrieve all ROS2 parameters."""
        # Chroma directory
        self.declare_parameter('chroma_directory', './chroma_db')
        self.chroma_directory = self.get_parameter(
            'chroma_directory').get_parameter_value().string_value
        self.get_logger().info(
            f'Parameter chroma_directory: [{self.chroma_directory}]')

        # Embedding model
        self.declare_parameter(
            'embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.embedding_model = self.get_parameter(
            'embedding_model').get_parameter_value().string_value
        self.get_logger().info(
            f'Parameter embedding_model: [{self.embedding_model}]')

        # Top k documents
        self.declare_parameter('top_k', 8)
        self.top_k = self.get_parameter(
            'top_k').get_parameter_value().integer_value
        self.get_logger().info(f'Parameter top_k: [{self.top_k}]')

        # Hybrid search
        self.declare_parameter('use_hybrid_search', True)
        self.use_hybrid_search = self.get_parameter(
            'use_hybrid_search').get_parameter_value().bool_value
        self.get_logger().info(
            f'Parameter use_hybrid_search: [{self.use_hybrid_search}]')

        self.declare_parameter('functions_filter', [''])
        self.functions_filter = self.get_parameter(
            'functions_filter').get_parameter_value().string_array_value
        self.get_logger().info(
            f'Parameter functions_filter: [{self.functions_filter}]')
        
        if not self.functions_filter:
            self.get_logger().error(
                'functions_filter is empty; Any log will be stored.'
            )

    def _initialize_rag_server(self) -> RAGServer:
        """Initialize the RAG server with configured parameters.

        Returns
        -------
        RAGServer
            Initialized RAG server instance.

        Raises
        ------
        Exception
            If RAG server initialization fails.
        """
        try:
            return RAGServer(
                logger=self.get_logger(),
                chroma_directory=self.chroma_directory,
                embedding_model=self.embedding_model,
                top_k=self.top_k,
                use_hybrid_search=self.use_hybrid_search,
            )
        except Exception as e:
            self.get_logger().error(f'Failed to initialize RAG server: {e}')
            raise

    def _create_services(self) -> None:
        """Create ROS2 services for RAG operations."""
        self.retrieve_srv = self.create_service(
            srv_type=RetrieveDocuments,
            srv_name='retrieve_documents',
            callback=self.retrieve_documents_callback,
            callback_group=self.group
        )

        self.store_srv = self.create_service(
            srv_type=StoreDocument,
            srv_name='store_document',
            callback=self.store_document_callback,
            callback_group=self.group
        )

    def _create_subscriptions(self) -> None:
        """Create ROS2 topic subscriptions."""
        self.rosout_subscription = self.create_subscription(
            Log,
            '/rosout',
            self.rosout_callback,
            1000
        )

    def retrieve_documents_callback(
        self,
        request: RetrieveDocuments_Request,
        response: RetrieveDocuments_Response,
    ) -> RetrieveDocuments_Response:
        """Handle RetrieveDocuments service requests.

        Processes a query and retrieves relevant documents from the
        vector database using semantic similarity, with optional filtering.

        Parameters
        ----------
        request : RetrieveDocuments.Request
            The service request containing query, k, and filters.
        response : RetrieveDocuments.Response
            The service response to be populated.

        Returns
        -------
        RetrieveDocuments.Response
            The populated response with retrieved documents.
        """
        query = request.query
        k = request.k if request.k > 0 else self.top_k
        filters = self._parse_filters(request.filters)

        self.get_logger().debug(
            f'Retrieve request: query="{query}", k={k}, filters={filters}'
        )

        try:
            result_dict = self._retrieve_and_parse_documents(query, k, filters)
            self._populate_retrieve_response(response, result_dict, query)
        except ValueError as e:
            self._handle_retrieve_error(response, e)

        return response

    def _parse_filters(self, filters_str: str) -> Optional[Dict[str, Any]]:
        """Parse filters JSON string.

        Parameters
        ----------
        filters_str : str
            JSON string containing filters.

        Returns
        -------
        Optional[Dict[str, Any]]
            Parsed filters dictionary or None if invalid.
        """
        if not filters_str:
            return None

        try:
            filters = json.loads(filters_str)
            self.get_logger().debug(f'Applied filters: {filters}')
            return filters
        except json.JSONDecodeError as e:
            self.get_logger().warning(f'Invalid filters JSON: {e}')
            return None

    def _retrieve_and_parse_documents(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Retrieve documents and parse the result.

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
        Dict[str, Any]
            Parsed result dictionary.
        """
        result_json = self.rag_server.retrieve_documents(
            query, k=k, filters=filters
        )
        return json.loads(result_json)

    def _populate_retrieve_response(
        self,
        response: RetrieveDocuments_Response,
        result_dict: Dict[str, Any],
        query: str,
    ) -> None:
        """Populate retrieve response with results.

        Parameters
        ----------
        response : RetrieveDocuments.Response
            Response to populate.
        result_dict : Dict[str, Any]
            Dictionary containing results.
        query : str
            Original query string.
        """
        response.status = result_dict['status']
        response.total_results = result_dict['total_results']
        response.results = [
            self._create_document_msg(doc_data)
            for doc_data in result_dict['results']
        ]

        self.get_logger().info(
            f'Retrieved {response.total_results} documents for query: "{query}"'
        )

    def _create_document_msg(self, doc_data: Dict[str, Any]) -> Document:
        """Create Document message from document data.

        Parameters
        ----------
        doc_data : Dict[str, Any]
            Document data dictionary.

        Returns
        -------
        Document
            Populated Document message.
        """
        doc = Document()
        doc.id = doc_data.get('id', 0)
        doc.content = doc_data.get('content', '')
        doc.metadata = self._create_metadata_msg(doc_data)
        return doc

    def _create_metadata_msg(self, doc_data: Dict[str, Any]) -> Metadata:
        """Create Metadata message from document data.

        Parameters
        ----------
        doc_data : Dict[str, Any]
            Document data dictionary.

        Returns
        -------
        Metadata
            Populated Metadata message.
        """
        metadata = Metadata()
        metadata.source = str(doc_data.get('source', ''))
        metadata.node_name = str(doc_data.get('node_name', ''))
        metadata.node_function = str(doc_data.get('node_function', ''))

        # log_level should be uint32, convert from string if needed
        log_level_value = doc_data.get('log_level', 'INFO')
        if isinstance(log_level_value, str):
            # Map string log levels to their numeric values
            log_level_map = {
                'DEBUG': 10,
                'INFO': 20,
                'WARN': 30,
                'WARNING': 30,
                'ERROR': 40,
                'FATAL': 50,
                'UNKNOWN': 20
            }
            metadata.log_level = log_level_map.get(log_level_value.upper(), 20)
        else:
            metadata.log_level = int(log_level_value)

        return metadata

    def _handle_retrieve_error(
        self,
        response: RetrieveDocuments_Response,
        error: Exception,
    ) -> None:
        """Handle retrieval error.

        Parameters
        ----------
        response : RetrieveDocuments.Response
            Response to populate with error information.
        error : Exception
            Exception that occurred.
        """
        self.get_logger().error(f'Error retrieving documents: {error}')
        response.status = 'error'
        response.total_results = 0
        response.results = []

    def store_document_callback(
        self,
        request: StoreDocument_Request,
        response: StoreDocument_Response,
    ) -> StoreDocument_Response:
        """Handle StoreDocument service requests.

        Stores a new document in the RAG system's vector database
        with metadata information.

        Parameters
        ----------
        request : StoreDocument.Request
            The service request containing a Document message.
        response : StoreDocument.Response
            The service response to be populated.

        Returns
        -------
        StoreDocument.Response
            The populated response with operation result.
        """
        document = request.document
        content = document.content

        self.get_logger().debug(
            f'Store request received with text length: {len(content)}'
        )

        try:
            metadata = self._extract_metadata_from_document(document)
            success = self.rag_server.store_document(content, metadata)
            self._populate_store_response(response, success)
        except Exception as e:
            self._handle_store_error(response, e)

        return response

    def _extract_metadata_from_document(self, document: Document) -> Dict[str, str]:
        """Extract metadata from Document message.

        Parameters
        ----------
        document : Document
            Document message containing metadata.

        Returns
        -------
        Dict[str, str]
            Extracted metadata dictionary.
        """
        return {
            'source': document.metadata.source,
            'node_name': document.metadata.node_name,
            'node_function': document.metadata.node_function,
            'log_level': str(document.metadata.log_level),
        }

    def _populate_store_response(
        self,
        response: StoreDocument_Response,
        success: bool,
    ) -> None:
        """Populate store response based on operation result.

        Parameters
        ----------
        response : StoreDocument.Response
            Response to populate.
        success : bool
            Whether the store operation succeeded.
        """
        if success:
            response.success = True
            response.message = 'Document stored successfully'
            self.get_logger().info('Document stored successfully')
        else:
            response.success = False
            response.message = 'Failed to store document'
            self.get_logger().error('Failed to store document')

    def _handle_store_error(
        self,
        response: StoreDocument_Response,
        error: Exception,
    ) -> None:
        """Handle store operation error.

        Parameters
        ----------
        response : StoreDocument.Response
            Response to populate with error information.
        error : Exception
            Exception that occurred.
        """
        response.success = False
        response.message = f'Error storing document: {str(error)}'
        self.get_logger().error(f'Error storing document: {error}')

    def rosout_callback(self, msg: Log) -> None:
        """Handle callbacks for /rosout topic subscription.

        This callback processes log messages published to the /rosout topic
        and stores them in the RAG database.

        Parameters
        ----------
        msg : Log
            The Log message received from /rosout.
        """
        # Avoid storing the node's own logs
        if msg.name == self.get_logger().name or msg.function not in self.functions_filter:
            return

        content = self._format_log_content(msg)
        metadata = self._extract_log_metadata(msg)

        self._store_log_message(content, metadata, msg.name)

    def _format_log_content(self, msg: Log) -> str:
        """Format log message content with timestamp.

        Parameters
        ----------
        msg : Log
            Log message.

        Returns
        -------
        str
            Formatted log content.
        """
        timestamp = f'{msg.stamp.sec}.{msg.stamp.nanosec:09d}'
        return f'[{timestamp}]: {msg.msg}'

    def _extract_log_metadata(self, msg: Log) -> Dict[str, str]:
        """Extract metadata from log message.

        Parameters
        ----------
        msg : Log
            Log message.

        Returns
        -------
        Dict[str, str]
            Extracted metadata dictionary.
        """
        return {
            'source': 'rosout_log',
            'node_name': msg.name,
            'node_function': msg.function,
            'log_level': LOG_LEVEL_NAMES.get(msg.level, 'UNKNOWN'),
        }

    def _store_log_message(
        self,
        content: str,
        metadata: Dict[str, str],
        node_name: str,
    ) -> None:
        """Store log message in RAG database.

        Parameters
        ----------
        content : str
            Log message content.
        metadata : Dict[str, str]
            Log message metadata.
        node_name : str
            Name of the node that generated the log.
        """
        try:
            success = self.rag_server.store_document(content, metadata)
            if not success:
                self.get_logger().warning(
                    f'Failed to store log message from {node_name}'
                )
        except Exception as e:
            self.get_logger().error(f'Error storing log message: {e}')


def main(args: Optional[Any] = None) -> None:
    """Run the RAG Service node.

    Initialize the ROS2 context, create the RAG service node, and spin
    until shutdown is requested. Uses a MultiThreadedExecutor for
    concurrent service handling.

    Parameters
    ----------
    args : Optional[Any]
        Command-line arguments (default: None).
    """
    rclpy.init(args=args)
    rag_node = None

    try:
        rag_node = RAGService()
        executor = MultiThreadedExecutor()
        executor.add_node(rag_node)
        executor.spin()
    except KeyboardInterrupt:
        print('Shutting down RAG service node (Keyboard Interrupt)')
    except ExternalShutdownException:
        print('Shutting down RAG service node (External Shutdown)')
    except Exception as e:
        print(f'Shutting down RAG service node due to error: {e}')
    finally:
        if rag_node is not None:
            rag_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

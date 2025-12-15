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
from rag_ros.rag_server import RAGServer
from llm_interactions_msgs.srv import RetrieveDocuments, StoreDocument
from llm_interactions_msgs.msg import Document

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


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

        # Retrieve ROS2 parameters
        self.get_params()

        # Initialize the RAG server
        try:
            self.rag_server = RAGServer(
                logger=self.get_logger(),
                documents_directory=self.documents_directory,
                chroma_directory=self.chroma_directory,
                k=self.default_k,
            )
        except Exception as e:
            self.get_logger().error(f'Failed to initialize RAG server: {e}')
            raise

        # Create service callback group for concurrent handling
        self.group = ReentrantCallbackGroup()

        # Create RetrieveDocuments service
        self.retrieve_srv = self.create_service(
            srv_type=RetrieveDocuments,
            srv_name='retrieve_documents',
            callback=self.retrieve_documents_callback,
            callback_group=self.group
        )

        # Create StoreDocument service
        self.store_srv = self.create_service(
            srv_type=StoreDocument,
            srv_name='store_document',
            callback=self.store_document_callback,
            callback_group=self.group
        )

        self.get_logger().info('RAG service node has been started.')

    def get_params(self) -> None:
        """Retrieve and configure ROS2 parameters.

        Declares and retrieves parameters from the ROS2 parameter server
        for RAG system configuration.

        Returns:
            None
        """
        # Declare and retrieve documents directory parameter
        self.declare_parameter('documents_directory', './data_files')
        self.documents_directory = self.get_parameter(
            'documents_directory').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter documents_directory is set to: [{self.documents_directory}]')

        # Declare and retrieve Chroma directory parameter
        self.declare_parameter('chroma_directory', './chroma_db')
        self.chroma_directory = self.get_parameter(
            'chroma_directory').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter chroma_directory is set to: [{self.chroma_directory}]')

        # Declare and retrieve default k parameter for document retrieval
        self.declare_parameter('default_k', 8)
        self.default_k = self.get_parameter(
            'default_k').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter default_k is set to: [{self.default_k}]')

    def retrieve_documents_callback(self, request, response):
        """Handle RetrieveDocuments service requests.

        Processes a query and retrieves relevant documents from the
        vector database using semantic similarity.

        Parameters:
            request: The RetrieveDocuments request containing query and k.
            response: The RetrieveDocuments response to be populated.

        Returns:
            The populated response with retrieved documents.
        """
        query = request.query
        k = request.k if request.k > 0 else self.default_k

        self.get_logger().debug(f'Retrieve request received for query: "{query}" with k={k}')

        try:
            # Retrieve documents from RAG server
            result_json = self.rag_server.retrieve_documents(query, k=k)
            result_dict = json.loads(result_json)

            # Populate response with status and metadata
            response.status = result_dict['status']
            response.message = result_dict['message']
            response.total_results = result_dict['total_results']

            # Convert results to Document messages
            response.results = []
            for doc_data in result_dict['results']:
                doc = Document()
                doc.doc_id = doc_data['doc_id']
                doc.source = doc_data['source']
                doc.content = doc_data['content']
                response.results.append(doc)

            self.get_logger().info(
                f'Retrieved {response.total_results} documents for query: "{query}"'
            )

        except ValueError as e:
            self.get_logger().error(f'Error retrieving documents: {e}')
            response.status = 'error'
            response.message = str(e)
            response.total_results = 0
            response.results = []

        return response

    def store_document_callback(self, request, response):
        """Handle StoreDocument service requests.

        Stores a new text document in the RAG system's vector database
        with optional metadata.

        Parameters:
            request: The StoreDocument request containing text and metadata.
            response: The StoreDocument response to be populated.

        Returns:
            The populated response with operation result.
        """
        text = request.text
        metadata_json = request.metadata_json

        self.get_logger().debug(f'Store request received with text length: {len(text)}')

        try:
            # Parse metadata if provided
            metadata = None
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError as e:
                    self.get_logger().warning(f'Invalid metadata JSON: {e}')

            # Store document in RAG server
            success = self.rag_server.store_document(text, metadata)

            if success:
                response.success = True
                response.message = 'Document stored successfully'
                self.get_logger().info('Document stored successfully')
            else:
                response.success = False
                response.message = 'Failed to store document'
                self.get_logger().error('Failed to store document')

        except Exception as e:
            response.success = False
            response.message = f'Error storing document: {str(e)}'
            self.get_logger().error(f'Error storing document: {e}')

        return response


def main(args=None) -> None:
    """Run the RAG Service node.

    Initialize the ROS2 context, create the RAG service node, and spin
    until shutdown is requested. Uses a MultiThreadedExecutor for
    concurrent service handling.

    Parameters:
        args: Command-line arguments (optional).

    Returns:
        None
    """
    rclpy.init(args=args)

    try:
        # Create the RAG service node
        rag_node = RAGService()

        # Use a MultiThreadedExecutor to allow concurrent service handling
        executor = MultiThreadedExecutor()
        executor.add_node(rag_node)

        # Spin the node to process service requests
        executor.spin()
    except (KeyboardInterrupt, Exception, ExternalShutdownException) as e:
        print(f'Shutting down RAG service node due to: {e}')


if __name__ == '__main__':
    main()

# rag_ros

![ROS2](https://img.shields.io/badge/ros2-jazzy-blue?logo=ros&logoColor=white)
![License](https://img.shields.io/github/license/grupo-avispa/rag_ros)

ROS 2 wrapper for Retrieval-Augmented Generation (RAG) systems, providing integration with LangChain and LangGraph for intelligent question-answering and document retrieval capabilities.

## Overview

This package provides a ROS 2 service node for RAG (Retrieval-Augmented Generation) operations. It uses a Chroma vector store with HuggingFace embeddings for semantic search and document retrieval. The node exposes services for storing and retrieving documents, and automatically captures ROS 2 log messages from the /rosout topic for storage in the database.

**Features:**
- Semantic search using Chroma vector store and HuggingFace embeddings
- ROS 2 service interface for document retrieval and storage
- Log message storage from /rosout topic
- Flexible configuration via ROS 2 parameters
- Support for metadata-rich document storage

**Keywords:** ROS2, RAG, LangChain, Vector Store, Semantic Search

**Author: Alberto Tudela<br />**

The rag_ros package has been tested under [ROS2] Jazzy on [Ubuntu] 24.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS) 2](https://docs.ros.org/en/jazzy/) (middleware for robotics)
- [llm_interactions_msgs](https://github.com/grupo-avispa/llm_interactions_msgs) (Custom ROS 2 messages for LLM interactions)
- [LangChain](https://www.langchain.com/) (Framework for LLM applications)
- [Chroma](https://www.trychroma.com/) (Vector store for embeddings)
- [HuggingFace Transformers](https://huggingface.co/transformers/) (Pre-trained embeddings)

#### Building

To build from source, clone the latest version from the repository into your colcon workspace and compile the package using:

```bash
cd colcon_workspace/src
git clone https://github.com/grupo-avispa/rag_ros.git
cd ../
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build --symlink-install
```

## Usage

Run the RAG service node with:

```bash
ros2 launch rag_ros default.launch.py
```

## Nodes

### rag_node

ROS 2 service node for RAG operations.

#### Services

* **`retrieve_documents`** ([llm_interactions_msgs/srv/RetrieveDocuments])

    Retrieve relevant documents from the vector database based on a query with optional filtering.

    **Request:**
    - `query` (string): The input query to retrieve relevant documents
    - `k` (int32): Number of documents to retrieve (default: 8)
    - `filters` (string): Optional metadata filters as JSON string. Supported filter keys: `source`, `node_name`, `node_function`, `log_level`

    **Response:**
    - `status` (string): Response status
    - `total_results` (int32): Total number of documents retrieved
    - `results` (Document[]): Array of retrieved documents with the following structure:
        - `id` (int32): Unique identifier for the document
        - `content` (string): Text content of the document
        - `metadata` (Metadata): Metadata associated with the document
            - `source` (string): Source or origin of the document
            - `node_name` (string): Name of the node that processed the document
            - `node_function` (string): Function of the node that processed the document
            - `log_level` (string): Log level of the message (DEBUG, INFO, WARN, ERROR, FATAL)

* **`store_document`** ([llm_interactions_msgs/srv/StoreDocument])

    Store a new document in the vector database.

    **Request:**
    - `document` (Document): Document to store with the following structure:
        - `id` (int32): Unique identifier for the document
        - `content` (string): Text content to store
        - `metadata` (Metadata): Metadata associated with the document
            - `source` (string): Source or origin of the document
            - `node_name` (string): Name of the node processing the document
            - `node_function` (string): Function of the node processing the document
            - `log_level` (string): Log level of the message (DEBUG, INFO, WARN, ERROR, FATAL)

    **Response:**
    - `success` (bool): Operation success status
    - `message` (string): Status message

#### Parameters

* **`chroma_directory`** (string, default: "./chroma_db")

    Directory where Chroma vector database persistence data will be stored.

* **`embedding_model`** (string, default: "sentence-transformers/all-MiniLM-L6-v2")

    HuggingFace embedding model to use for semantic search.

* **`default_k`** (int, default: 8)

    Default number of documents to retrieve per query.

## Example Usage

### Retrieve Documents

```bash
# Basic retrieval
ros2 service call /retrieve_documents llm_interactions_msgs/srv/RetrieveDocuments "{query: 'machine learning', k: 5}"

# Retrieval with log level filter
ros2 service call /retrieve_documents llm_interactions_msgs/srv/RetrieveDocuments "{query: 'error', k: 5, filters: '{\"log_level\": \"ERROR\"}'}"

# Retrieval with multiple filters
ros2 service call /retrieve_documents llm_interactions_msgs/srv/RetrieveDocuments "{query: 'database', k: 5, filters: '{\"log_level\": \"ERROR\", \"node_name\": \"my_node\"}'}"
```

### Store Document

```bash
ros2 service call /store_document llm_interactions_msgs/srv/StoreDocument "{document: {id: 1, content: 'Machine learning is a subset of artificial intelligence', metadata: {source: 'example.txt', node_name: 'example_node', node_function: 'process', log_level: 'INFO'}}}"
```

## Configuration

You can customize the RAG service behavior by passing parameters to the launch file:

```bash
ros2 launch rag_ros default.launch.py chroma_directory:=/path/to/chroma default_k:=10

# With custom embedding model
ros2 launch rag_ros default.launch.py embedding_model:='sentence-transformers/all-mpnet-base-v2'
```

[Ubuntu]: https://ubuntu.com/
[ROS2]: https://docs.ros.org/en/humble/
[llm_interactions_msgs/srv/RetrieveDocuments]: https://github.com/grupo-avispa/llm_interactions_msgs/blob/main/srv/RetrieveDocuments.srv
[llm_interactions_msgs/srv/StoreDocument]: https://github.com/grupo-avispa/llm_interactions_msgs/blob/main/srv/StoreDocument.srv
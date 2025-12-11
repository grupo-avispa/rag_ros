# rag_ros

![ROS2](https://img.shields.io/badge/ros2-jazzy-blue?logo=ros&logoColor=white)
![License](https://img.shields.io/github/license/grupo-avispa/rag_ros)

ROS 2 wrapper for Retrieval-Augmented Generation (RAG) systems, providing integration with LangChain and LangGraph for intelligent question-answering and document retrieval capabilities.

## Overview

This package provides a ROS 2 service node for RAG (Retrieval-Augmented Generation) operations. It indexes CSV and PDF documents into a Chroma vector store using HuggingFace embeddings and exposes services for semantic search and document storage.

**Features:**
- Document indexing from CSV and PDF files
- Semantic search using Chroma vector store and HuggingFace embeddings
- ROS 2 service interface for document retrieval and storage
- Automatic directory creation and initialization
- Flexible configuration via ROS 2 parameters

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

    Retrieve relevant documents from the vector database based on a query.

    **Request:**
    - `query` (string): The query string to search for
    - `k` (int32): Number of documents to retrieve

    **Response:**
    - `status` (string): Operation status
    - `message` (string): Status message
    - `total_results` (int32): Number of documents retrieved
    - `results_json` (string): Retrieved documents in JSON format

* **`store_document`** ([llm_interactions_msgs/srv/StoreDocument])

    Store a new document in the vector database.

    **Request:**
    - `text` (string): Document content to store
    - `metadata_json` (string): Optional metadata in JSON format

    **Response:**
    - `success` (bool): Operation success status
    - `message` (string): Status message

#### Parameters

* **`documents_directory`** (string, default: "./data_files")

    Directory containing CSV and PDF files to index.

* **`chroma_directory`** (string, default: "./chroma_db")

    Directory where Chroma vector database persistence data will be stored.

* **`default_k`** (int, default: 8)

    Default number of documents to retrieve per query.

## Example Usage

### Retrieve Documents

```bash
ros2 service call /retrieve_documents llm_interactions_msgs/srv/RetrieveDocuments "{query: 'machine learning', k: 5}"
```

### Store Document

```bash
ros2 service call /store_document llm_interactions_msgs/srv/StoreDocument "{text: 'Machine learning is a subset of artificial intelligence', metadata_json: '{\"source\": \"example.txt\"}'}"
```

## Configuration

You can customize the RAG service behavior by passing parameters to the launch file:

```bash
ros2 launch rag_ros default.launch.py documents_directory:=/path/to/documents chroma_directory:=/path/to/chroma default_k:=10
```

[Ubuntu]: https://ubuntu.com/
[ROS2]: https://docs.ros.org/en/humble/
[llm_interactions_msgs/srv/RetrieveDocuments]: https://github.com/grupo-avispa/llm_interactions_msgs/blob/main/srv/RetrieveDocuments.srv
[llm_interactions_msgs/srv/StoreDocument]: https://github.com/grupo-avispa/llm_interactions_msgs/blob/main/srv/StoreDocument.srv
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

import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Loading packages from the current virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    site_packages = os.path.join(
        venv_path,
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages'
    )
    sys.path.insert(0, site_packages)


def generate_launch_description():
    """Generate launch description for RAG Service node."""
    # Get the virtual environment (use 'python3' if not available)
    venv_path = os.environ.get('VIRTUAL_ENV')
    venv_python = os.path.join(venv_path, 'bin', 'python') if venv_path else 'python3'
    # Get the launch directory
    rag_dir = get_package_share_directory('rag_ros')

    # Create the launch configuration variables
    documents_directory = LaunchConfiguration('documents_directory')
    chroma_directory = LaunchConfiguration('chroma_directory')
    top_k = LaunchConfiguration('top_k')
    use_hybrid_search = LaunchConfiguration('use_hybrid_search')

    # Declare launch arguments
    declare_documents_directory_cmd = DeclareLaunchArgument(
        'documents_directory',
        default_value=os.path.join(rag_dir, 'files'),
        description='Directory containing CSV and PDF files to index')

    declare_chroma_directory_cmd = DeclareLaunchArgument(
        'chroma_directory',
        default_value=os.path.join(rag_dir, 'chroma_db'),
        description='Directory where Chroma vector database persistence data will be stored')

    declare_top_k_cmd = DeclareLaunchArgument(
        'top_k',
        default_value='8',
        description='Default number of documents to retrieve')

    declare_use_hybrid_search_cmd = DeclareLaunchArgument(
        'use_hybrid_search',
        default_value='true',
        description='Enable hybrid search combining semantic + BM25 retrievers')

    # RAG Service node
    rag_node_cmd = Node(
        package='rag_ros',
        executable='node',
        name='rag_node',
        output='screen',
        prefix=[venv_python, ' -u '],
        parameters=[{
            'documents_directory': documents_directory,
            'chroma_directory': chroma_directory,
            'top_k': top_k,
            'use_hybrid_search': use_hybrid_search,
        }])

    # Create the launch description and populate
    ld = LaunchDescription()

    ld.add_action(declare_documents_directory_cmd)
    ld.add_action(declare_chroma_directory_cmd)
    ld.add_action(declare_top_k_cmd)
    ld.add_action(declare_use_hybrid_search_cmd)
    ld.add_action(rag_node_cmd)

    return ld

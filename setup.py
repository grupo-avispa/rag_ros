from glob import glob
from setuptools import find_packages, setup

package_name = 'rag_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'empy',
        'jinja2',
        'langchain_chroma',
        'langchain_classic',
        'langchain_community',
        'langchain_core',
        'langchain_huggingface',
        'langchain_ollama',
        'langchain_text_splitters',
        'langgraph',
        'langgraph-cli[inmem]',
        'langsmith',
        'lark',
        'rank_bm25',
        'sentence_transformers',
    ],
    zip_safe=True,
    maintainer='Alberto Tudela',
    maintainer_email='ajtudela@gmail.com',
    description='ROS 2 wrapper for Retrieval-Augmented Generation (RAG) systems',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node = ' + package_name + '.main:main',
        ],
    },
)

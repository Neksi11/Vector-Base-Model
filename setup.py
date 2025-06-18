"""
Setup script for the Advanced Vector-Based RAG Agent package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-rag-agent",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Vector-Based RAG Agent with sophisticated response generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-rag-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.1",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "enhanced": [
            "pandas>=1.5.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "nltk>=3.8",
            "textblob>=0.17.1",
        ],
        "performance": [
            "numba>=0.56.0",
            "faiss-cpu>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-demo=examples.advanced_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rag_agent": ["*.py"],
        "examples": ["*.py"],
        "tests": ["*.py"],
    },
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "machine-learning",
        "nlp",
        "information-retrieval",
        "question-answering",
        "vector-search",
        "semantic-search",
        "ai",
        "artificial-intelligence"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-rag-agent/issues",
        "Source": "https://github.com/yourusername/advanced-rag-agent",
        "Documentation": "https://advanced-rag-agent.readthedocs.io/",
    },
)

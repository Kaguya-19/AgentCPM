from setuptools import setup, find_packages
import os

setup(
    name="mcp-search-server",
    version="0.1.0",
    description="MCP server providing research tools including web search, scholar search, and webpage reading",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mcp>=1.1.2",
        "httpx>=0.28.1",
        "tiktoken>=0.8.0",
        "pydantic>=2.0.0",
        "uvicorn>=0.32.1",
        "starlette>=0.41.3",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mcp-search-server=search_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)

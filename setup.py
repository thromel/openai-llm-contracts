"""Setup script for openai-llm-contracts package."""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openai-llm-contracts",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="LLM Design by Contract Framework - A reliability layer for OpenAI and other LLM APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openai-llm-contracts",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "typing-extensions>=4.0.0",
        "python-dateutil>=2.8.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "langchain": ["langchain>=0.1.0"],
        "guardrails": ["guardrails-ai>=0.2.0"],
        "telemetry": ["opentelemetry-api>=1.20.0", "opentelemetry-sdk>=1.20.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.0.260",
        ],
    },
)
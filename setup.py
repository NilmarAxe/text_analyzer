"""
Strategic setup configuration for Text Analyzer.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-e')
        ]
else:
    requirements = ['psutil>=5.8.0']

# Package metadata
setup(
    name="Ax",
    version="1.0.0",
    author="Ax",
    description="Strategic text analysis system with comprehensive frequency analysis and statistical reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/strategic-analysis/text-analyzer",
    project_urls={
        "Bug Reports": "https://github.com/strategic-analysis/text-analyzer/issues",
        "Source": "https://github.com/strategic-analysis/text-analyzer",
        "Documentation": "https://text-analyzer.readthedocs.io/",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "text_analyzer": ["config/*.json", "data/*.txt"],
    },
    
    # Dependencies
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "performance": [
            "chardet>=4.0.0",
            "cython>=0.29.0",
            "numba>=0.56.0",
        ],
        "all": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0", 
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "chardet>=4.0.0",
            "cython>=0.29.0",
            "numba>=0.56.0",
        ]
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "text-analyzer=text_analyzer.cli.main:main",
            "text-analysis=text_analyzer.cli.main:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    
    # Keywords
    keywords="text analysis, frequency analysis, natural language processing, statistics, reporting, nlp, linguistics",
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    license="MIT",
)
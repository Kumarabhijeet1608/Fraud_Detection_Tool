#!/usr/bin/env python3
"""
Setup script for Multi-Modal Fraud Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="fraud-detection-system",
    version="1.0.0",
    author="Processing2o Team",
    author_email="team@processing2o.com",
    description="A comprehensive fraud detection system with multiple specialized models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="fraud detection, machine learning, security, phishing, malware",
    project_urls={
        "Bug Reports": "https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool/issues",
        "Source": "https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool",
        "Documentation": "https://github.com/Kumarabhijeet1608/Fraud_Detection_Tool#readme",
    },
)

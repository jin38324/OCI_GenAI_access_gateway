import os
import oci

OCI_CONFIG = oci.config.from_file('~/.oci/config', 'GENERATEAI')
COMPARTMENT_ID = OCI_CONFIG["compartment_id"]

DEFAULT_API_KEYS = "ocigenerativeai"

API_ROUTE_PREFIX = "/api/v1"

TITLE = "OCI Generative AI Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for OCI Generative AI Service"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for OCI Generative AI Service models.

List of OCI Generative AI Service models currently supported:

Chat models:
- meta.llama-3-70b-instruct
- cohere.command-r-plus
- cohere.command-r-16k

Embedding models:
- cohere.embed-english-v3.0
- cohere.embed-multilingual-v3.0
- cohere.embed-english-light-v3.0
- cohere.embed-multilingual-light-v3.0
"""

# DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
DEBUG = True
OCI_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL", "cohere.command-r-plus"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3.0"
)

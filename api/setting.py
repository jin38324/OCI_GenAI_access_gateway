import os
import oci

OCI_CONFIG = oci.config.from_file('~/.oci/config', 'DEFAULT')
OCI_REGION = "us-chicago-1"
service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
AUTH_TYPE="API_KEY"  # if do not want to use this, comment this line
AUTH_TYPE="INSTANCE_PRINCIPAL"


client_kwargs = {
                "config": {},
                "signer": None,
                "service_endpoint": service_endpoint,
                "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                "timeout": (10, 240),  # default timeout config for OCI Gen AI service
            } 

# OCI_CONFIG.update({'region':OCI_REGION})
COMPARTMENT_ID = 'ocid1.compartment.oc1..aaaaaaaau5q457a7teqkjce4oenoiz6bmc4g3s74a5543iqbm7xwplho44fq'

DEFAULT_API_KEYS = "a"

API_ROUTE_PREFIX = "/v1"

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
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL", "cohere.command-r-plus"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3.0"
)


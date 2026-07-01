import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from oci.generative_ai_inference import models as oci_models

from api.models.adapter.rerank_adapter import RerankRequestAdapter
from api.schema import RerankRequest


class RerankRequestAdapterTest(unittest.TestCase):
    def setUp(self):
        self.model_info = {
            "model_id": "cohere.rerank-v4.0-fast",
            "compartment_id": "ocid1.compartment.oc1..example",
        }

    def test_converts_cohere_request_to_oci(self):
        request = RerankRequest(
            model="cohere.rerank-v4.0-fast",
            query="capital of the United States",
            documents=["Carson City", "Washington, D.C."],
            top_n=1,
            max_tokens_per_doc=512,
            priority=10,
        )

        details = RerankRequestAdapter(self.model_info).to_oci(request)

        self.assertEqual(details.input, request.query)
        self.assertEqual(details.documents, request.documents)
        self.assertEqual(details.top_n, 1)
        self.assertEqual(details.max_tokens_per_document, 512)
        self.assertFalse(details.is_echo)
        self.assertEqual(details.serving_mode.model_id, "cohere.rerank-v4.0-fast")

    def test_converts_oci_response_to_cohere(self):
        response = oci_models.RerankTextResult(
            id="rerank-id",
            document_ranks=[
                oci_models.DocumentRank(index=1, relevance_score=0.99),
                oci_models.DocumentRank(index=0, relevance_score=0.25),
            ],
        )

        converted = RerankRequestAdapter.to_cohere(response)

        self.assertEqual(converted.id, "rerank-id")
        self.assertEqual(converted.results[0].index, 1)
        self.assertEqual(converted.results[0].relevance_score, 0.99)
        self.assertEqual(converted.meta.api_version.version, "2")
        self.assertFalse(converted.meta.api_version.is_experimental)

    def test_uses_cohere_default_max_tokens_per_doc(self):
        request = RerankRequest(
            model="cohere.rerank-v4.0-fast",
            query="query",
            documents=["one"],
        )

        details = RerankRequestAdapter(self.model_info).to_oci(request)

        self.assertEqual(details.max_tokens_per_document, 4096)

    def test_uses_dedicated_endpoint_when_configured(self):
        model_info = self.model_info | {"endpoint": "ocid1.generativeaiendpoint.example"}
        request = RerankRequest(
            model="cohere.rerank-v4.0-fast",
            query="query",
            documents=["one"],
        )

        details = RerankRequestAdapter(model_info).to_oci(request)

        self.assertEqual(details.serving_mode.serving_type, "DEDICATED")
        self.assertEqual(
            details.serving_mode.endpoint_id,
            "ocid1.generativeaiendpoint.example",
        )


if __name__ == "__main__":
    unittest.main()

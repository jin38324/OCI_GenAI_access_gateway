from typing import Annotated

from fastapi import APIRouter, Body, Depends

from api.auth import api_key_auth
from api.models.oci_rerank import get_rerank_model
from api.schema import RerankRequest, RerankResponse, RerankResultsResponse


router = APIRouter(
    prefix="/rerank",
    dependencies=[Depends(api_key_auth)],
)

v1_router = APIRouter(
    prefix="/rerank",
    dependencies=[Depends(api_key_auth)],
)


def _execute_rerank(request: RerankRequest) -> RerankResponse:
    model = get_rerank_model(request.model)
    return model.rerank(request)


@router.post("", response_model=RerankResponse)
async def rerank(
    rerank_request: Annotated[
        RerankRequest,
        Body(
            examples=[
                {
                    "query": "What is the capital of the United States?",
                    "documents": [
                        "Carson City is the capital of Nevada.",
                        "Washington, D.C. is the capital of the United States.",
                    ],
                    "top_n": 1,
                }
            ]
        ),
    ]
) -> RerankResponse:
    return _execute_rerank(rerank_request)


@v1_router.post("", response_model=RerankResultsResponse)
async def rerank_v1(request: RerankRequest) -> RerankResultsResponse:
    response = _execute_rerank(request)
    return RerankResultsResponse(results=response.results)

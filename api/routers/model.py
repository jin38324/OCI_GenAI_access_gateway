from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.models.ocigenai import OCIGenAIModel
from api.schema import Models, Model

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

chat_model = OCIGenAIModel()


async def validate_model_id(model_id: str):
    if model_id not in chat_model.list_models():
        raise HTTPException(status_code=500, detail="Unsupported Model Id")


@router.get("", response_model=Models)
async def list_models():
    model_list = [
        Model(id=model_id) for model_id in chat_model.list_models()
    ]
    return Models(data=model_list)


@router.get("/{model_id}",response_model=Model,)
async def get_model(
        model_id: Annotated[str,Path(description="Model ID", example="cohere.command-r-plus"),]
    ):
    await validate_model_id(model_id)
    return Model(id=model_id)

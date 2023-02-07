# service.py
import torch
import asyncio
import bentoml
from bentoml.io import JSON, Text
from pydantic import BaseModel


class Hatespeech(BaseModel):
    sentence: str


generation_runner = bentoml.transformers.get("generation_model:latest").to_runner()

svc = bentoml.Service("generation_model", runners=[generation_runner])

input_spec = JSON(pydantic_model=Hatespeech)


@svc.api(input=input_spec, output=JSON())
async def purificate(hatespeech: Hatespeech):

    sentence = hatespeech.dict()["sentence"]

    generation_result = await generation_runner.async_run(sentence)

    result = {"purificated": generation_result}

    return result

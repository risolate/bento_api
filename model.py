import yaml
import torch
import bentoml
from bentoml.io import Text
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
from transformers.pipelines import SUPPORTED_TASKS
from pipeline import MyGenerationPipeline

model = T5ForConditionalGeneration.from_pretrained("happy06/KcT5-purificate")
tokenizer = AutoTokenizer.from_pretrained("beomi/KcT5-dev")

TASK_NAME = "generation_task"
TASK_DEFINITION = {
    "impl": MyGenerationPipeline,
    "tf": (),
    "pt": (T5ForConditionalGeneration,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

generator = pipeline(
    task=TASK_NAME,
    model=model,
    tokenizer=tokenizer,
)

generation_saved_model = bentoml.transformers.save_model(
    "generation_model",
    pipeline=generator,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
    # signatures={
    #     "__call__": {
    #         "batchable": True,
    #         "batch_dim": (0, 0),
    #     },
    # },
)

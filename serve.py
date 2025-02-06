import logging
import os
from typing import Any, Dict

from fastapi import FastAPI
from starlette.requests import Request

# from ray import serve
from ray import serve
from qwentransformers import Qwen2dot5

# Configure logging
logger = logging.getLogger("ray.serve")

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(name="Qwen2dot5Deployment", num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class Qwen2dot5Deployment:
    def __init__(
            self,
            model_name: str
    ):
        self.qwen = Qwen2dot5(logger, model_name)

    @app.post("/v1/chat/completions", response_model=Any)
    async def create_chat_completions(self, raw_request: Request):
        request_body = await raw_request.json()
        return self.qwen.create_chat_completions(request_body)


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    return Qwen2dot5Deployment.bind(cli_args['model'])


model = build_app({"model": os.environ['MODEL_ID']})

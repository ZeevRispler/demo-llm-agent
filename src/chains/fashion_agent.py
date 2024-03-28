import os
default_env_file = os.getenv("MLRUN_DEFAULT_ENV_FILE", "~/.mlrun.env")
from src.chains.base import ChainRunner
from typing import Any

from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
import mlrun
from .agent_graph import AgentGraph

class MlrunLLM(LLM):
    llm: Any = Field(default=None)

    def __init__(
            self,
            nuclio_function: Any,
            **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.llm = nuclio_function
        self.name = "huggingface_local_model"

    @property
    def _llm_type(self) -> str:
        return "MlrunLLM"

    def _call(
            self,
            prompt: str,
            **generate_kwargs: dict,
    ) -> str:
        if isinstance(self.llm, mlrun.serving.server.GraphServer):
            response = self.llm.test(path=f'/v2/models/{self.name}/predict',
                                     body={"inputs": [prompt], **generate_kwargs})
            return response["outputs"][0]["generated_text"]
        else:
            response = self.llm.invoke(path=f'/v2/models/{self.name}/predict',
                                       body={"inputs": [prompt], **generate_kwargs}, verify=False)
            return response["outputs"][0]["generated_text"]


class Agent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        project = mlrun.get_or_create_project(name="deploy-agent-isbank", context="./")
        self.nuclio_function = project.get_function("example-llm-server")
        self.llm = MlrunLLM(nuclio_function=self.nuclio_function)
        self.agent = AgentGraph(self.llm)
        self.agent.build_graph()

    def _run(self, event):
        response, sources = self.agent.run({"input": event.query})
        return {"answer": response, "sources": sources}





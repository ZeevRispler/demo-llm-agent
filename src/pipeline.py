from src.app.chains.base import HistorySaver, SessionLoader
from src.app.chains.refine import RefineQuery
from src.app.pipelines import app_server
from src.app.chains.jewelry_agent import Agent
from src.app.schema import PipelineEvent
pipe_config = [
    SessionLoader(),
    RefineQuery(),
    Agent(),
    HistorySaver(),
]

pipelines = {
    "default": pipe_config,
}


app_server.add_pipelines(pipelines)
app = app_server.to_fastapi(with_controller=True)


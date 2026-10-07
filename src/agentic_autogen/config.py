from agentic.config import Settings


class AutogenSettings(Settings):
    # Base model of the deployment; empty means "use the deployment name".
    azure_openai_model: str = ""
    azure_openai_api_version: str = "2024-10-21"

    @property
    def model(self) -> str:
        return self.azure_openai_model or self.azure_openai_deployment

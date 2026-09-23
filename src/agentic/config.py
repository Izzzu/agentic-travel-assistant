from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_openai_endpoint: str = ""
    # Leave empty to authenticate with Microsoft Entra ID instead of a key.
    azure_openai_api_key: SecretStr | None = None
    azure_openai_deployment: str = ""
    log_level: str = "INFO"

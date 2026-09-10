import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    PREFECT_API_KEY: str = os.getenv("PREFECT_API_KEY", "")
    PREFECT_ACCOUNT_ID: str = os.getenv("PREFECT_ACCOUNT_ID", "")
    PREFECT_WORKSPACE_ID: str = os.getenv("PREFECT_WORKSPACE_ID", "")
    PREFECT_FLOW_ID: str = os.getenv("PREFECT_FLOW_ID", "")
    PREFECT_DEPLOYMENT_ID: str = os.getenv("PREFECT_DEPLOYMENT_ID", "")

    @property
    def workspace_base_url(self):
        if not self.PREFECT_ACCOUNT_ID or not self.PREFECT_WORKSPACE_ID:
            raise ValueError("PREFECT_ACCOUNT_ID and PREFECT_WORKSPACE_ID must be set.")
        return f"https://api.prefect.cloud/api/accounts/{self.PREFECT_ACCOUNT_ID}/workspaces/{self.PREFECT_WORKSPACE_ID}"

    @property
    def auth_headers(self):
        if not self.PREFECT_API_KEY:
            raise ValueError("PREFECT_API_KEY must be set.")
        return {"Authorization": f"Bearer {self.PREFECT_API_KEY}", "Content-Type": "application/json"}

settings = Settings()

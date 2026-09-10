"""Prefect Cloud REST API helper: deployment details."""
import logging
import requests
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploymentAPI")


def get_deployment(deployment_id=None):
    deployment_id = deployment_id or settings.PREFECT_DEPLOYMENT_ID
    if not deployment_id:
        raise ValueError("Set PREFECT_DEPLOYMENT_ID or pass deployment_id.")
    url = f"{settings.workspace_base_url}/deployments/{deployment_id}"
    response = requests.get(url, headers=settings.auth_headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    logger.info("Deployment: name=%s id=%s tags=%s schedule=%s", data.get("name"), data.get("id"), data.get("tags"), data.get("schedule"))
    return data


if __name__ == "__main__":
    get_deployment()

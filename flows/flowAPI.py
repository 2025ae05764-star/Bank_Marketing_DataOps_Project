"""Prefect Cloud REST API helper: flow details and recent runs."""
import logging
import requests
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flowAPI")


def get_flow(flow_id=None):
    flow_id = flow_id or settings.PREFECT_FLOW_ID
    if not flow_id:
        raise ValueError("Set PREFECT_FLOW_ID or pass flow_id.")
    url = f"{settings.workspace_base_url}/flows/{flow_id}"
    response = requests.get(url, headers=settings.auth_headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    logger.info("Flow: name=%s id=%s created=%s updated=%s", data.get("name"), data.get("id"), data.get("created"), data.get("updated"))
    return data


def list_recent_flow_runs(flow_id=None, limit=5):
    flow_id = flow_id or settings.PREFECT_FLOW_ID
    if not flow_id:
        raise ValueError("Set PREFECT_FLOW_ID or pass flow_id.")
    url = f"{settings.workspace_base_url}/flow_runs/filter"
    payload = {"flow_runs": {"flow_id": {"any_": [flow_id]}}, "sort": "START_TIME_DESC", "limit": limit}
    response = requests.post(url, headers=settings.auth_headers, json=payload, timeout=15)
    response.raise_for_status()
    runs = response.json()
    for run in runs:
        logger.info("Run: name=%s state=%s start=%s end=%s", run.get("name"), run.get("state_name"), run.get("start_time"), run.get("end_time"))
    return runs


def run():
    return {"flow": get_flow(), "recent_runs": list_recent_flow_runs()}


if __name__ == "__main__":
    run()

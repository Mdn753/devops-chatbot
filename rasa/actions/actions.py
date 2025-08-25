# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import os, requests, re
from random import randint
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
import logging
import os, re, typing as t, requests


N8N_BASE = os.getenv("N8N_BASE", "http://n8n:5678")  # use your compose service name

N8N_GNS3_URL = os.getenv("N8N_GNS3_URL", f"{N8N_BASE}/webhook-test/gns3_setup")
N8N_ZABBIX_URL = os.getenv("N8N_ZABBIX_URL", f"{N8N_BASE}/webhook-test/zabbix_setup")
N8N_PYTHON_PROJECT_URL = os.getenv("N8N_PYTHON_PROJECT_URL", f"{N8N_BASE}/webhook-test/python_project")
N8N_CREATE_VM_URL = os.getenv("N8N_CREATE_VM_URL", f"{N8N_BASE}/webhook-test/create_vm")

logger = logging.getLogger(__name__)

def vm_from_tracker(tracker: Tracker, prefix: str):
    """Return (vm_name, was_auto). Try slot first, then entities, else autogen."""
    raw = (tracker.get_slot("vm_name") or "").strip()

    # fallback: check latest extracted entities
    if not raw:
        try:
            for e in tracker.latest_message.get("entities", []):
                if e.get("entity") == "vm_name" and e.get("value"):
                    raw = str(e["value"]).strip()
                    break
        except Exception:
            pass

    if raw:
        return sanitize(raw), False
    from random import randint
    return f"{prefix}-{randint(1000, 9999)}", True

REPO_RX = re.compile(r"^(https?://|git@)[^ \t]+\.git$", re.I)
class ActionCreateVM(Action):
    def name(self):
        return "action_create_vm"

    async def run(self, dispatcher, tracker, domain):
        vm_name, was_auto = vm_from_tracker(tracker, "vm")
        payload = {"name": vm_name}
        logger.info("CREATE_VM payload → %s", payload)

        try:
            r = requests.post(N8N_CREATE_VM_URL, json=payload, timeout=900)
            if r.ok:
                try:
                    msg = r.json().get("message")
                except Exception:
                    msg = r.text[:200] if r.text else None
                dispatcher.utter_message(msg or f"Launching {vm_name}…")
            else:
                dispatcher.utter_message(f"n8n returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            dispatcher.utter_message(f"Failed to reach n8n: {e}")

        return [SlotSet("vm_name", vm_name)]


def sanitize(name: str) -> str:
    """Allow letters, numbers, dot, underscore, dash; compress spaces → '-'."""
    name = name.strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name.strip("-")


class ActionSetupZabbix(Action):
    def name(self) -> str:
        return "action_setup_zabbix"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        vm_name, was_auto = vm_from_tracker(tracker, "zbx")
        dispatcher.utter_message(
            f"Provisioning VM '{vm_name}' and installing Zabbix…"
            + (" (auto-generated name)" if was_auto else "")
        )

        payload = {"name": vm_name, "install": "zabbix", "auto": was_auto}
        logger.info("SETUP_ZABBIX payload → %s", payload)

        try:
            r = requests.post(N8N_ZABBIX_URL, json=payload, timeout=9000000)
            if r.ok:
                try:
                    data = r.json()
                    msg = data.get("message") or data.get("status") or r.text
                except Exception:
                    msg = r.text
                dispatcher.utter_message(msg[:500] if msg else "Zabbix setup request sent.")
            else:
                dispatcher.utter_message(f"n8n returned {r.status_code}: {r.text[:500]}")
                logger.error("n8n error (Zabbix): %s %s", r.status_code, r.text)
        except Exception as e:
            dispatcher.utter_message(f"Failed to reach n8n: {e}")
            logger.exception("HTTP call to n8n failed (Zabbix)")

        return [SlotSet("vm_name", vm_name)]
    
class ActionSetupGNS3(Action):
    def name(self) -> str:
        return "action_setup_GNS3"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        vm_name, was_auto = vm_from_tracker(tracker, "gns3")
        dispatcher.utter_message(
            f"Provisioning VM '{vm_name}' and installing GNS3…"
            + (" (auto-generated name)" if was_auto else "")
        )

        payload = {"name": vm_name, "install": "gns3", "auto": was_auto}
        logger.info("SETUP_GNS3 payload → %s", payload)

        try:
            r = requests.post(N8N_GNS3_URL, json=payload, timeout=9000000)
            if r.ok:
                try:
                    data = r.json()
                    msg = data.get("message") or data.get("status") or r.text
                except Exception:
                    msg = r.text
                dispatcher.utter_message(msg[:500] if msg else "GNS3 setup request sent.")
            else:
                dispatcher.utter_message(f"n8n returned {r.status_code}: {r.text[:500]}")
                logger.error("n8n error (GNS3): %s %s", r.status_code, r.text)
        except Exception as e:
            dispatcher.utter_message(f"Failed to reach n8n: {e}")
            logger.exception("HTTP call to n8n failed (GNS3)")

        return [SlotSet("vm_name", vm_name)]



def _derive_name_from_repo(repo: str) -> str:
    # e.g. https://github.com/user/proj.git -> proj
    base = repo.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".git") else base

class ValidateDeployForm(FormValidationAction):
    def name(self) -> str: return "validate_deploy_form"

    def validate_repo(self, slot_value: str, dispatcher: CollectingDispatcher,
                      tracker: Tracker, domain: DomainDict) -> dict:
        v = (slot_value or "").strip()
        if REPO_RX.search(v):
            return {"repo": v}
        dispatcher.utter_message(text="That doesn’t look like a git URL. Please paste something like https://github.com/user/repo.git or git@github.com:user/repo.git")
        return {"repo": None}

    def validate_name(self, slot_value: str, dispatcher, tracker, domain) -> dict:
        v = (slot_value or "").strip()
        if not v:
            repo = tracker.get_slot("repo") or ""
            v = _derive_name_from_repo(repo) or "ml-api"
        return {"name": v}

    def validate_branch(self, slot_value: str, dispatcher, tracker, domain) -> dict:
        v = (slot_value or "").strip() or "main"
        return {"branch": v}

    def validate_entrypoint(self, slot_value: str, dispatcher, tracker, domain) -> dict:
        v = (slot_value or "").strip() or "app.py"
        return {"entrypoint": v}

    def validate_port(self, slot_value: t.Any, dispatcher, tracker, domain) -> dict:
        try:
            p = int(str(slot_value).strip())
        except Exception:
            dispatcher.utter_message(text="Port must be a number (0-65535).")
            return {"port": None}
        if 0 <= p <= 65535:
            return {"port": p}
        dispatcher.utter_message(text="Port must be between 0 and 65535.")
        return {"port": None}

class ActionLaunchDeploy(Action):
    def name(self) -> str: return "action_launch_deploy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> t.List[EventType]:
        repo = tracker.get_slot("repo")
        name = tracker.get_slot("name")
        branch = tracker.get_slot("branch") or "main"
        entry = tracker.get_slot("entrypoint") or "app.py"
        port = tracker.get_slot("port") or 0

        # Build the payload your n8n Webhook expects (you can add env, limit, etc.)
        payload = {
            "repo": repo,
            "branch": branch,
            "name": name,
            "entrypoint": entry,
            "port": port,
            "env": {"PORT": str(port)}
        }

        try:
            r = requests.post(N8N_PYTHON_PROJECT_URL, json=payload, timeout=20)
            r.raise_for_status()
            dispatcher.utter_message(text=f"Okay, I’ve sent the deploy request for {name}.")
        except Exception as e:
            dispatcher.utter_message(text=f"Couldn’t reach the deploy service: {e}")
        return []
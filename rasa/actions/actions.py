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
from rasa_sdk.events import SlotSet


N8N_URL = os.getenv("N8N_WEBHOOK_URL",
                    "http://host.docker.internal:5678/webhook-test/create_vm")


N8N_ZABBIX_URL = os.getenv(
    "N8N_ZABBIX_URL",
    "http://host.docker.internal:5678/webhook-test/zabbix_setup",
)
class ActionCreateVM(Action):
    def name(self):
        return "action_create_vm"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker, domain: dict):
        vm_name = f"vm-{randint(1000,9999)}"
        r = requests.post(N8N_URL, json={"name": vm_name})
        dispatcher.utter_message(r.json().get("message",
                             f"Launching {vm_name} …"))
        return []

def sanitize(name: str) -> str:
    """Allow letters, numbers, dot, underscore, dash; compress spaces → '-'."""
    name = name.strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name.strip("-")


class ActionSetupZabbix(Action):
    def name(self) -> str:
        return "action_setup_zabbix"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        provided = tracker.get_slot("vm_name")
        if provided:
            vm_name = sanitize(provided) or f"zbx-{randint(1000,9999)}"
            was_auto = False
        else:
            vm_name = f"zbx-{randint(1000,9999)}"
            was_auto = True

        dispatcher.utter_message(
            f"Provisioning VM '{vm_name}' and installing Zabbix…"
            + (" (auto-generated name)" if was_auto else "")
        )

        try:
            # Keep payload simple; align with what your n8n flow expects
            payload = {"name": vm_name, "install": "zabbix", "auto": was_auto}
            r = requests.post(N8N_ZABBIX_URL, json=payload, timeout=9000000)

            if r.ok:
                # surface message if JSON, else show text
                try:
                    msg = r.json().get("message")
                except Exception:
                    msg = r.text[:200] if r.text else None
                dispatcher.utter_message(msg or f"Workflow triggered for '{vm_name}'.")
            else:
                dispatcher.utter_message(f"n8n returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            dispatcher.utter_message(f"Failed to reach n8n: {e}")

        return [SlotSet("vm_name", vm_name)]
    
class ActionSetupGNS3(Action):
    def name(self) -> str:
        return "action_setup_GNS3"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        
        provided = tracker.get_slot("vm_name")
        if provided:
            vm_name = sanitize(provided) or f"gns3-{randint(1000,9999)}"
            was_auto = False
        else:
            vm_name = f"gns3-{randint(1000,9999)}"
            was_auto = True

        dispatcher.utter_message(
            f"Provisioning VM '{vm_name}' and installing GNS3…"
            + (" (auto-generated name)" if was_auto else "")
        )

        try:
            payload = {"name": vm_name, "install": "gns3", "auto": was_auto}
            r = requests.post(
                "http://host.docker.internal:5678/webhook-test/gns3_setup",
                json=payload,
                timeout=9000000
            )

            if r.ok:
                try:
                    msg = r.json().get("message")
                except Exception:
                    msg = r.text[:200] if r.text else None
                dispatcher.utter_message(msg or f"Workflow triggered for '{vm_name}'.")
            else:
                dispatcher.utter_message(f"n8n returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            dispatcher.utter_message(f"Failed to reach n8n: {e}")

        return [SlotSet("vm_name", vm_name)]
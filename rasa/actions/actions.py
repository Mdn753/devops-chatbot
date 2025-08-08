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

import os, requests
from random import randint
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List

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

class ActionSetupZabbix(Action):
    def name(self) -> str:
        return "action_setup_zabbix"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        # Optional: pass a target (vm/group) if you capture it in a slot
        target = tracker.get_slot("target") or "all"

        # (Optional) show immediate feedback
        dispatcher.utter_message("Kicking off Zabbix setup…")

        try:
            payload = {"target": target}
            r = requests.post(N8N_ZABBIX_URL, json=payload, timeout=30)

            if r.ok:
                # Try to show message from n8n response if it's JSON
                msg = None
                try:
                    msg = r.json().get("message")
                except Exception:
                    msg = r.text[:200] if r.text else None

                dispatcher.utter_message(msg or f"Zabbix setup triggered for '{target}'.")
            else:
                dispatcher.utter_message(
                    f"Failed to trigger workflow ({r.status_code})."
                )

        except Exception as e:
            dispatcher.utter_message(f"Could not reach the automation: {e}")

        return []

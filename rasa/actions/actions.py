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

N8N_URL = os.getenv("N8N_WEBHOOK_URL",
                    "http://host.docker.internal:5678/webhook-test/create_vm")

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


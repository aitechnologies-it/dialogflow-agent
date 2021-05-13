from dialogflow_v2.types import Intent
from dialogflow_v2 import IntentsClient
import dialogflow_v2
import os
from google.oauth2 import service_account

project_id="plg-chatbot"
intent_id="0db7c106-d5f2-4e45-8fd4-0367bba0d377"
sa_path="/Users/diego/Progetti/AIT/dialogflow-agent/credentials/dialogflow-fcqosk-plg-chatbot.json"

new_training_phrases = []
sents = open("/Users/diego/Progetti/AIT/dialogflow-agent/sents.txt", "r")
for t in sents.readlines():
    t = t.strip()
    parts = [
        Intent.TrainingPhrase.Part(text=t),
    ]
    training_phrase = Intent.TrainingPhrase(parts=parts)
    new_training_phrases.append(training_phrase)

# print(new_training_phrases)

credentials = service_account.Credentials.from_service_account_file(sa_path)
client = IntentsClient(credentials=credentials)
full_intent_name = os.path.join("projects", project_id, "agent/intents", intent_id)

intent = client.get_intent(name=full_intent_name, intent_view=dialogflow_v2.enums.IntentView.INTENT_VIEW_FULL)
intent.training_phrases.extend(new_training_phrases)
response = client.update_intent(intent, language_code='en')
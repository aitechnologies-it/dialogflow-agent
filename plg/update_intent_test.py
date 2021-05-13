from dialogflow_v2.types import Intent
from dialogflow_v2 import IntentsClient
import dialogflow_v2
from google.oauth2 import service_account

project_id = "plg-chatbot"
intent_name = "your_intent_name"
sa_path = "../credentials/dialogflow-fcqosk-plg-chatbot.json"

new_training_phrases = []
sents = open("sents.txt", "r")
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

all_intents = client.list_intents(parent=f"projects/{project_id}/agent", intent_view=dialogflow_v2.enums.IntentView.INTENT_VIEW_FULL)
for intent in all_intents:
    if intent.display_name == intent_name:
        intent.training_phrases.extend(new_training_phrases)
        response = client.update_intent(intent, language_code='en')

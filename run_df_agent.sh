pip install -r requirements.txt

LOCAL_OR_REMOTE="plg-chatbot"
SERVICE_ACCOUNT="./dialogflow-fcqosk-plg-chatbot.json"

python df_agent.py \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT
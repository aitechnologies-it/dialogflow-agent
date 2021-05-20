./install.sh

LOCAL_OR_REMOTE="plg-chatbot"
SERVICE_ACCOUNT="./credentials/dialogflow-fcqosk-plg-chatbot.json"
OUTPUT_DIR="../ml-artifacts/data/bet9ja"

echo "Downloading dialogflow sentences. Project = ${LOCAL_OR_REMOTE}. Output dir = ${OUTPUT_DIR}"

python run_dfagent.py \
    --command "get" \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT \
    --output_dir $OUTPUT_DIR
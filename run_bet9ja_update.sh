./install.sh

LOCAL_OR_REMOTE="plg-chatbot"
SERVICE_ACCOUNT="./credentials/dialogflow-fcqosk-plg-chatbot.json"
INPUT_DIR_OR_FILE="./frasi"
INTENT_NAME="test"

echo "Downloading dialogflow sentences. Project = ${LOCAL_OR_REMOTE}. Output dir = ${OUTPUT_DIR}"

python update_df_examples.py \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT \
    --input_dir_or_file $INPUT_DIR_OR_FILE \
    --intent_name $INTENT_NAME
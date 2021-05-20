./install.sh

LOCAL_OR_REMOTE="chatbot-colombia-int-301508"
SERVICE_ACCOUNT="./credentials/dialogflow-int-colombia-int-301508.json"
INPUT_DIR_OR_FILE="./frasi"
INTENT_NAME="test"

echo "Uploading dialogflow sentences to intent ${INTENT_NAME}. Project = ${LOCAL_OR_REMOTE}. Input dir/file = ${INPUT_DIR_OR_FILE}"

python run_dfagent.py \
    --command "update" \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT \
    --input_dir_or_file $INPUT_DIR_OR_FILE \
    --intent_name $INTENT_NAME \
    --lang es

./install.sh

LOCAL_OR_REMOTE="chatbot-colombia-int-301508"
SERVICE_ACCOUNT="./credentials/dialogflow-int-colombia-int-301508.json"
OUTPUT_DIR="../ml-artifacts/data/colombia"

echo "Downloading dialogflow sentences. Project = ${LOCAL_OR_REMOTE}. Output dir = ${OUTPUT_DIR}"

python save_df_examples.py \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT \
    --output_dir $OUTPUT_DIR
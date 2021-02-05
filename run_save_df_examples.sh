pip install -r requirements.txt

LOCAL_OR_REMOTE="plg-chatbot"
SERVICE_ACCOUNT="./dialogflow-fcqosk-plg-chatbot.json"
OUTPUT_DIR="./outputs"

python save_df_examples.py \
    --local_path_or_url $LOCAL_OR_REMOTE \
    --service_account $SERVICE_ACCOUNT \
    --output_dir $OUTPUT_DIR
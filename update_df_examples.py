import dfagent

import argparse
import coloredlogs
import logging

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path_or_url", type=str, required=True, help="The path to local agent zip file (/dir) or gcp project name hosting dialogflow agent.")
    parser.add_argument("--service_account", type=str, required=True, help="The GCP service account path.")
    parser.add_argument("--input_dir_or_file", type=str, required=True, help="The input dir or file to read training examples from.")
    parser.add_argument("--intent_name", type=str, required=True, help="The name of the intent to update.")
    parser.add_argument("--input_format", type=str, default='default', required=False, help="The input format to read intent examples from. Choose: default.")
    args = parser.parse_args()

    logger.warning(f"Arguments:"
            f" - service account={args.service_account} - input_dir_or_file={args.input_dir_or_file}"
            f" - input_format={args.input_format} - intent_name={args.intent_name}")

    # Setup dialogflow agent
    agent = dfagent.DialogFlowAgent(
        local_path_or_url=args.local_path_or_url,
        service_account=args.service_account,
        input_format=args.input_format
    )

    # Updating DialoFlow with new training examples.
    response, examples, _ = agent.add_training_examples(
        intent_name=args.intent_name,
        input_dir_or_file=args.input_dir_or_file,
    )

    # Collected stats
    logger.info(f"Response from IntentsClient:\n-----\n{response}-----")
    logger.info(f'no. added examples = {len(examples)}')
    logger.info(f'Done.')


if __name__ == "__main__":
    main()

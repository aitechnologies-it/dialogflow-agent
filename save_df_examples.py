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
    parser.add_argument("--service_account", type=str, required=False, help="The GCP service account path.")
    parser.add_argument("--output_dir", type=str, required=False, help="The output dir to write data to, eg intent, labels, etc..")
    parser.add_argument("--content_type", type=str, default='json', required=False, help="The type of files to handle in the export / import df agent. Choose: json.")
    parser.add_argument("--output_format", type=str, default='default', required=False, help="The output format to write intents out. Choose: default.")
    args = parser.parse_args()

    logger.warning(f"Arguments: local_path_or_url={args.local_path_or_url}"
                            f" - service account={args.service_account} - output_dir={args.output_dir}"
                            f" - content_type={args.content_type} - output_format={args.output_format}")

    # Setup dialogflow agent
    agent = dfagent.DialogFlowAgent(
        local_path_or_url=args.local_path_or_url,
        service_account=args.service_account,
        content_type=args.content_type,
        output_format=args.output_format
    )

    # Get Dialogflow examples to save in specified output directory for later use
    examples = agent.get_training_examples()
    agent.save_training_examples(examples, output_dir=args.output_dir)

    # Collected stats
    logger.info(f'Collected stats:')
    logger.info(f'\tNo. collected examples = {len(examples)}')
    

if __name__ == "__main__":
    main()

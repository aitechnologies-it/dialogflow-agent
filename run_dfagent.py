import dfagent

import argparse
import coloredlogs
import logging

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, help="Name of command", choices=['update', 'get'])
    parser.add_argument("--local_path_or_url", type=str, required=True, help="The path to local agent zip file (/dir) or gcp project name hosting dialogflow agent.")
    parser.add_argument("--service_account", type=str, required=True, help="The GCP service account path.")
    # get
    parser.add_argument("--output_dir", type=str, required=False, help="The output dir to write data to, eg intent, labels, etc..")
    parser.add_argument("--content_type", type=str, default='json', required=False, help="The type of files to handle in the export / import df agent. Choose: json.")
    parser.add_argument("--output_format", type=str, default='default', required=False, help="The output format to write intents out. Choose: default.")
    parser.add_argument('--filter_intents', nargs='*', help='The intents to ignore', required=False)
    # update
    parser.add_argument("--input_dir_or_file", type=str, required=False, help="The input dir or file to read training examples from.")
    parser.add_argument("--input_format", type=str, default='default', required=False, help="The input format to read intent examples from. Choose: default.")
    parser.add_argument("--intent_name", type=str, required=False, help="The name of the intent to update.")
    parser.add_argument("--lang", type=str, default='en', required=False, help="The agent language")
    args = parser.parse_args()

    logger.warning(f"Arguments: {vars(args)}")

    # Setup dialogflow agent
    agent = dfagent.DialogFlowAgent(
        local_path_or_url=args.local_path_or_url,
        service_account=args.service_account,
        content_type=args.content_type,
        input_format=args.input_format,
        output_format=args.output_format,
    )

    if args.command == 'get':
        # Get Dialogflow examples to save in specified output directory for later use
        examples = agent.get_training_examples(filter_intents=args.filter_intents)
        agent.save_training_examples(examples, output_dir=args.output_dir)

        logger.info(f'Collected stats:')
        logger.info(f'\tno. examples = {len(examples)}')
    elif args.command == 'update':
        # Updating DialoFlow with new training examples.
        response, examples, _ = agent.add_training_examples(
            intent_name=args.intent_name,
            input_dir_or_file=args.input_dir_or_file,
            lang=args.lang
        )

        logger.info(f'Collected stats:')
        logger.info(f'\tno. examples = {len(examples)}')
        logger.info(f"Response:\n{response}")

    logger.info(f'\tDone.')

if __name__ == "__main__":
    main()

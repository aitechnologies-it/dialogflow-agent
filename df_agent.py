import argparse
import coloredlogs, logging
import zipfile
import os
import json
import glob
import fnmatch
import io
import re
from datetime import datetime
from typing import List, Dict, Tuple

from tqdm import tqdm

import google
import dialogflow_v2

logger = logging.getLogger(__name__)


class AgentReader:
    def __init__(self, local_path_or_url, **kwargs):
        super(AgentReader, self).__init__()
        self.local_path_or_url = local_path_or_url

    def get_path(self):
        return self.local_path_or_url

    def set_path(self, path):
        self.local_path_or_url = path

    def read(self, **kwargs):
        pass

    def close(self, **kwargs):
        pass

    @staticmethod
    def from_remote(dialogflow, **kwargs):
        return dialogflow.get_agent()
    
    @classmethod
    def from_dir_or_url(self, local_path_or_url, **kwargs):
        if zipfile.is_zipfile(local_path_or_url):
            return AgentReaderFromInMemoryZip(local_path_or_url=local_path_or_url)
        elif os.path.isdir(local_path_or_url):
            return AgentReaderFromLocalDir(local_path_or_url=local_path_or_url)
        else:
            df = kwargs.pop('dialogflow', None)
            return AgentReaderFromInMemoryStreamZip(local_path_or_url=local_path_or_url, stream=self.from_remote(dialogflow=df))

class AgentReaderFromInMemoryZip(AgentReader):
    def __init__(self, local_path_or_url, **kwargs):
        super(AgentReaderFromInMemoryZip, self).__init__(local_path_or_url, **kwargs)

    def _read(self, **kwargs):
        inp = kwargs.pop('inp', None)
        glob_pattern = kwargs.pop('glob', None)
        with zipfile.ZipFile(inp) as thezip:
            for zipinfo in thezip.infolist():
                if fnmatch.fnmatch(zipinfo.filename, glob_pattern):
                    with thezip.open(zipinfo) as thefile:
                        yield zipinfo.filename, thefile

    def read(self, **kwargs):
        return self._read(inp=self.get_path(), **kwargs)

class AgentReaderFromInMemoryStreamZip(AgentReaderFromInMemoryZip):
    def __init__(self, local_path_or_url, stream, **kwargs):
        super(AgentReaderFromInMemoryStreamZip, self).__init__(local_path_or_url, **kwargs)
        self.stream = stream

    def get_stream(self):
        return self.stream

    def read(self, **kwargs):
        return self._read(inp=io.BytesIO(self.get_stream()), **kwargs)

class AgentReaderFromLocalDir(AgentReader):
    def __init__(self, local_path_or_url, **kwargs):
        super(AgentReaderFromLocalDir, self).__init__(local_path_or_url, **kwargs)

    def read(self, **kwargs):
        glob_pattern = kwargs.pop('glob', None)
        file_names = glob.glob(os.path.join(self.get_path(), glob_pattern))
        for filename in file_names:
            with open(filename, 'r') as fp:
                yield filename, fp

class AgentContentReader:
    def __init__(self, reader, **kwargs):
        super(AgentContentReader, self).__init__()
        self.reader = reader
    
    def get_content(self, glob, **kwargs):
        pass
            
class AgentContentJSONReader(AgentContentReader):
    def __init__(self, reader, **kwargs):
        super(AgentContentJSONReader, self).__init__(reader=reader, **kwargs)

    def read_json(self, fp, **kwargs):
        return json.load(fp)

    def get(self, json, path, default=None, **kwargs):
        return json.get(path, default)

    def get_content(self, glob, **kwargs) -> List[Tuple[str, dict]]:
        content = []
        regex = kwargs.pop('regex', None)
        for filename, fp in self.reader.read(glob=glob):
            if regex is not None:
                pattern = re.compile(regex)
                if pattern.match(filename):
                    content.append((filename, self.read_json(fp=fp, **kwargs)))
            else:
                content.append((filename, self.read_json(fp=fp, **kwargs)))
        return content


class IntentReader:
    def __init__(self, reader, **kwargs):
        super(IntentReader, self).__init__()
        self.reader = reader

    def get_intents(self, **kwargs):
        pass


class JSONIntentReader(IntentReader):
    def __init__(self, reader, **kwargs):
        super(JSONIntentReader, self).__init__(reader=reader, **kwargs)
        self.jr = AgentContentJSONReader(reader=self.reader, **kwargs)

    def get_text(self, data, **kwargs) -> str:
        text = ""
        if data is None:
            return None
        for chunk in data:
            chunk_text = chunk.get('text', None)
            if chunk_text is None:
                return None
            text = text + chunk_text
        return text

    def get_reader(self):
        return self.jr

    def get_intents(self, **kwargs) -> Dict[str, List[List[str]]]:
        intents = {}
        logger.info('Collecting intents.')
        for ((filename, user_says), (filename_intent, intent)) in tqdm(zip(self.get_reader().get_content(glob='intents/*_usersays_*.json'),
                                    self.get_reader().get_content(glob='intents/*.json', regex=r"^((?!.*usersays.*).)*$"))):
            label = self.get_reader().get(intent, path='name')
            if label is None:
                logger.info(f'Cannot find a label / intent name in {filename_intent}. Skipping.')
                continue
            for i, us in enumerate(user_says):
                text = self.get_text(data=self.get_reader().get(us, path='data'))
                if text is None:
                    logger.info(f'Found something broken in {filename}. It may be due to a missing data field, or malformed chunk for sentence = {i}. Skipping.')
                    continue
                if label not in intents:
                    intents[label] = [text]
                else:
                    intents[label].append(text)
        return intents


class IntentWriter:
    def __init__(self, **kwargs):
        super(IntentWriter, self).__init__()
    
    def write(self, data, **kwargs):
        pass


class BaseOutputFormatIntentWriter(IntentWriter):
    def __init__(self, **kwargs):
        super(BaseOutputFormatIntentWriter, self).__init__(**kwargs)

    def write(self, data: Dict[str, List[str]], **kwargs):
        base_path = kwargs.pop('output_dir', None)
        if base_path is None:
            base_path = f'./data-{datetime.now().isoformat()}'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        logger.info(f'Writing out intents to {base_path}')
        with open(os.path.join(base_path, 'sent'), 'w') as sent_fp, \
            open(os.path.join(base_path, 'label'), 'w') as label_fp:
            for i, (label, texts) in tqdm(enumerate(data.items())):
                for text in texts:
                    if not isinstance(text, str) or not isinstance(label, str):
                        logger.info(f'Found intent name or user sentence not to be of type string. Skipping example = {i}.')
                        continue
                    sent_fp.write(f'{text}\n')
                    label_fp.write(f'{label}\n')


class DialogFlowAgentExport:
    def __init__(self, local_path_or_url, dialogflow=None, content_type='json', **kwargs):
        super(DialogFlowAgentExport, self).__init__()
        content_types = {
            'json': JSONIntentReader
        }

        if content_type not in content_types:
            raise ValueError(f'Cannot find content_type={content_type}.')

        self.agent_reader   = AgentReader.from_dir_or_url(local_path_or_url=local_path_or_url, dialogflow=dialogflow)
        self.intents_reader = content_types[content_type](reader=self.agent_reader, **kwargs)

    def get_intents(self, **kwargs) -> Dict[str, List[str]]:
        return self.intents_reader.get_intents()

    def get_intent_names(self, **kwargs) -> List[str]:
        return list(self.get_intents().keys())

class DialogFlowAgentClient:
    def __init__(self, project_name, service_account, **kwargs):
        if project_name is None or service_account is None:
            raise ValueError(f'Please provide correct project name and service account file.')
        self.client = dialogflow_v2.AgentsClient.from_service_account_json(service_account)
        self.parent = self.client.project_path(project_name)
        
    def get_agent(self, **kwargs):
        zip_raw = None
        logger.info(f'Exporting remote dialogflow agent.')
        try:
            operation = self.client.export_agent(self.parent)
            zip_raw = operation.result().agent_content
        except google.api_core.exceptions.GoogleAPICallError as e:
            print("From docs: 'Request failed for any reason'.")
            raise e
        except google.api_core.exceptions.RetryError as e:
            print("From docs: 'Eequest failed due to a retryable error and retry attempts failed'.")
            raise e
        except ValueError as e:
            print("Invalid parameters")
            raise e
        return zip_raw

class DialogFlowAgent:
    def __init__(self, local_path_or_url, service_account, content_type='json', output_format='default', **kwargs):
        output_formats = {
            'default': BaseOutputFormatIntentWriter
        }

        if output_format not in output_formats:
            raise ValueError(f'Cannot find output_format={output_format}.')

        self.df     = DialogFlowAgentClient(project_name=local_path_or_url, service_account=service_account)
        self.export = DialogFlowAgentExport(local_path_or_url=local_path_or_url, dialogflow=self.df, content_type=content_type)
        self.writer = output_formats[output_format](**kwargs)

    def get_intents(self, **kwargs) -> Dict[str, List[List[str]]]:
        return self.export.get_intents()

    def get_intent_names(self, intents: Dict[str, List[List[str]]] = None, **kwargs) -> List[str]:
        if intents is None:
            return self.export.get_intent_names()
        return list(intents.keys())

    def write_intents(self, inp, **kwargs):
        output_dir = kwargs.pop('output_dir', None)
        self.writer.write(data=inp, output_dir=output_dir)


def main():
    # Setup logging
    coloredlogs.install(level='INFO', logger=logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path_or_url", type=str, required=True, help="The path to local agent zip file (/dir) or gcp project name hosting dialogflow agent.")
    parser.add_argument("--service_account", type=str, required=False, help="The GCP service account path.")
    parser.add_argument("--output_dir", type=str, required=False, help="The output dir to write data to, eg intent, labels, etc..")
    parser.add_argument("--content_type", type=str, required=False, help="The type of files to handle in the export / import df agent. Choose: json.")
    parser.add_argument("--output_format", type=str, required=False, help="The output format to write intents out. Choose: default.")
    args = parser.parse_args()

    # Setup dialogflow agent
    agent = DialogFlowAgent(local_path_or_url=args.local_path_or_url, service_account=args.service_account)

    # Get intents and labels
    intents = agent.get_intents()

    # Write out intents along with corresponding labels
    agent.write_intents(inp=intents, output_dir=args.output_dir)

    # Collected stats
    logger.info(f'Collected stats:')
    logger.info(f'\tNo. collected intents = {len(intents)}')
    

if __name__ == "__main__":
    main()


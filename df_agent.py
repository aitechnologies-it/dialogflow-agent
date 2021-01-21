import argparse
import logging
import zipfile
import os
import json
import glob
import fnmatch
import io
from datetime import datetime
from typing import List, Dict

import google
import dialogflow_v2

from tqdm import tqdm

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
        pass

    def get_content(self, glob, **kwargs) -> List:
        content = []
        for filename, fp in self.reader.read(glob=glob):
            the_json = self.read_json(fp=fp)
        return content


class IntentReader:
    def __init__(self, reader, **kwargs):
        super(IntentReader, self).__init__()
        self.reader = reader

    def get_intents(self, **kwargs):
        pass


class JSONIntentReader(IntentReader):
    def __init__(self, reader, **kwargs):
        super(JSONIntentReader, self).__init__(reader=reader)
        self.content_reader = AgentContentJSONReader(reader=self.reader)

    def get_intents(self) -> Dict[str, List[str]]:
        intents = {}
        for js in self.content_reader.get_content(glob='intents/*.json'):
            pass
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
        base_path = kwargs.pop('output_dir', f'./data-{datetime.now().isoformat()}')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open(os.path.join(base_path, 'sent'), 'w') as sent_fp, \
            open(os.path.join(base_path, 'label'), 'w') as label_fp:
            for label, texts in data.items():
                for text in texts:
                    sent_fp.write(text)
                    label_fp.write(label)


class DialogFlowAgentExport:
    def __init__(self, local_path_or_url, dialogflow=None, **kwargs):
        super(DialogFlowAgentExport, self).__init__()
        self.agent_reader   = AgentReader.from_dir_or_url(local_path_or_url=local_path_or_url, dialogflow=dialogflow)
        self.intents_reader = JSONIntentReader(reader=self.agent_reader)

    def get_intents(self, **kwargs):
        return self.intents_reader.get_intents()

    def get_labels(self, **kwargs):
        pass

class DialogFlowAgentClient:
    def __init__(self, project_name, service_account, **kwargs):
        if project_name is None or service_account is None:
            raise ValueError(f'Please provide correct project name and service account file.')
        self.client = dialogflow_v2.AgentsClient.from_service_account_json(service_account)
        self.parent = self.client.project_path(project_name)
        
    def get_agent(self):
        zip_raw = None
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
    def __init__(self, local_path_or_url, service_account, **kwargs):
        self.the_agent = DialogFlowAgentClient(project_name=local_path_or_url, service_account=service_account)
        self.agent_export = DialogFlowAgentExport(local_path_or_url=local_path_or_url, dialogflow=self.the_agent)
        self.writer = BaseOutputFormatIntentWriter()

    def get_intents(self):
        return self.agent_export.get_intents()

    def write_intents(self, **kwargs):
        intents = self.get_intents()
        self.writer.write(data=intents, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path_or_url", type=str, required=True, help="The path to local agent zip file (/dir) or gcp project name hosting dialogflow agent.")
    parser.add_argument("--service_account", type=str, required=False, help="The GCP service account path.")
    parser.add_argument("--output_dir", type=str, required=False, help="The output dir to write data to, eg intent, labels, etc..")
    args = parser.parse_args()

    agent = DialogFlowAgent(local_path_or_url=args.local_path_or_url, service_account=args.service_account)
    agent.write_intents(output_dir=args.output_dir)

if __name__ == "__main__":
    main()


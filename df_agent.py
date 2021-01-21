import argparse
import logging
import zipfile
import os
import json
import glob
import fnmatch
import io
from typing import List, Dict

import google
import dialogflow_v2

logger = logging.getLogger(__name__)

class DialogFlowAgent:
    def __init__(self, project_name, service_account_filename, **kwargs):
        self.client = dialogflow_v2.AgentsClient.from_service_account_json(service_account_filename)
        parent = client.project_path(project_name)
        
    def get_agent(self):
        zip_raw = None
        try:
            operation = client.export_agent(parent)
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

    def from_remote(self, dialogflow):
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
            print(filename)
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

class DialogFlowAgentExport:
    def __init__(self, local_path_or_url, **kwargs):
        super(DialogFlowAgentExport, self).__init__()
        self.agent_reader   = AgentReader.from_dir_or_url(local_path_or_url=local_path_or_url)
        self.intents_reader = JSONIntentReader(reader=self.agent_reader)

    def get_intents(self, **kwargs):
        return self.intents_reader.get_intents()

    def get_labels(self, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path_or_url", type=str, required=True, help="The path to local agent zip file (/dir) or remote url.")
    args = parser.parse_args()

    agent = DialogFlowAgentExport(local_path_or_url=args.local_path_or_url)
    agent.get_intents()

if __name__ == "__main__":
    main()
import unicodedata
import coloredlogs, logging
import zipfile
import os
import json
import glob
import fnmatch
import io
import re
from datetime import datetime
from typing import List, Dict, Tuple, Iterator, Optional, Union
from dataclasses import dataclass
import operator

from tqdm import tqdm
import pydlib as dl
from funcy import partial

import google
import dialogflow_v2

logger = logging.getLogger(__name__)
# Setup logging
coloredlogs.install(level='INFO', logger=logger)

@dataclass
class DialoflowTrainingExample:
    """Class for modeling Dialogflow training examples."""
    text: str
    label: str
    tag: str

    def from_dict(inp: dict):
        if not isinstance(inp, dict):
            raise TypeError(f'Argument inp must be dict, not a {type(inp)}.')
        return DialoflowTrainingExample(text=inp.get('text'), label=inp.get('label'), tag=inp.get('tag'))

class AgentReader:
    def __init__(self, local_path_or_url, **kwargs):
        super(AgentReader, self).__init__()
        self.local_path_or_url = local_path_or_url

    def get_loc(self):
        return self.local_path_or_url

    def set_loc(self, path):
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
        inp = kwargs.pop('loc', self.get_loc())
        glob_pattern = kwargs.pop('glob', '*')
        with zipfile.ZipFile(inp) as thezip:
            for zipinfo in thezip.infolist():
                if fnmatch.fnmatch(zipinfo.filename, glob_pattern):
                    with thezip.open(zipinfo) as thefile:
                        yield zipinfo.filename, thefile

    def read(self, **kwargs):
        return self._read(loc=self.get_loc(), **kwargs)

class AgentReaderFromInMemoryStreamZip(AgentReaderFromInMemoryZip):
    def __init__(self, local_path_or_url, stream, **kwargs):
        super(AgentReaderFromInMemoryStreamZip, self).__init__(local_path_or_url, **kwargs)
        self.stream = stream

    def get_stream(self):
        return self.stream

    def read(self, **kwargs):
        return self._read(loc=io.BytesIO(self.get_stream()), **kwargs)

class AgentReaderFromLocalDir(AgentReader):
    def __init__(self, local_path_or_url, **kwargs):
        super(AgentReaderFromLocalDir, self).__init__(local_path_or_url, **kwargs)

    def _read(self, **kwargs):
        loc = kwargs.pop('loc', self.get_loc())
        glob_pattern = kwargs.pop('glob', '*')
        file_names = glob.glob(os.path.join(loc, glob_pattern))
        file_names = sorted(file_names)
        for filename in file_names:
            # # IN CASE YOU EVER NEED TO RECURSIVELY GO DOWN
            # if os.path.isdir(filename):
            #     for _, _, files in os.walk(filename):
            #         files = sorted(files)
            #         for sub_filename in files:
            #             with open(os.path.join(filename, sub_filename), 'r') as fp:
            #                 yield sub_filename, fp
            with open(filename, 'r') as fp:
                yield filename, fp

    def read(self, **kwargs):
        return self._read(loc=self.get_loc(), **kwargs)

class AgentContentReader:
    def __init__(self, agent_reader, **kwargs):
        super(AgentContentReader, self).__init__()
        self.agent_reader = agent_reader
    
    def get_content(self, glob, **kwargs):
        pass

    def get_reader(self):
        return self.agent_reader
            
class AgentContentJSONReader(AgentContentReader):
    def __init__(self, agent_reader, **kwargs):
        super(AgentContentJSONReader, self).__init__(agent_reader=agent_reader, **kwargs)

    def read_json(self, fp, **kwargs):
        return json.load(fp)

    def get(self, json, path, default=None, **kwargs):
        return json.get(path, default)

    def get_content(self, glob, **kwargs) -> Iterator:
        regex = kwargs.pop('regex', None)
        for filename, fp in self.get_reader().read(glob=glob):
            if regex is not None:
                pattern = re.compile(regex)
                if pattern.match(filename):
                    yield (filename, self.read_json(fp=fp, **kwargs))
            else:
                yield (filename, self.read_json(fp=fp, **kwargs))

class IntentReader:
    def __init__(self, **kwargs):
        super(IntentReader, self).__init__()

    def get_intents(self, **kwargs):
        pass

class JSONIntentReader(IntentReader):
    def __init__(self, agent_reader, **kwargs):
        super(JSONIntentReader, self).__init__()
        self.json_reader = AgentContentJSONReader(agent_reader=agent_reader, **kwargs)
        self.IGNORE_TAG = '@sys.ignore'

    def get_reader(self):
        return self.json_reader

    @staticmethod
    def preprocessing_intent(text: str) -> str:
        clean = text.strip()
        clean = JSONIntentReader.unicode_normalize(clean)
        return clean

    @staticmethod
    def preprocessing_tag(text: str) -> str:
        clean = text.strip()
        clean = JSONIntentReader.unicode_normalize(clean)
        clean = JSONIntentReader.remove_special(clean)
        return clean

    @staticmethod
    def unicode_normalize(text: str) -> str:
        text = str(text)
        text = unicodedata.normalize('NFD', text) \
            .encode('ascii', 'ignore') \
            .decode("utf-8")
        return str(text)

    @staticmethod
    def remove_special(text: str) -> str:
        clean = [re.sub(r"[^a-zA-Z0-9 ]+", '', text)]
        clean = " ".join(clean).strip()
        return clean

    def get_text(self, data: List, **kwargs) -> str:
        text = ""
        if data is None:
            return None
        if not isinstance(data, List):
            raise TypeError(f'Argument data must be a list not a {type(data)}.')
        for chunk in data:
            chunk_text = chunk.get('text', None)
            if chunk_text is None:
                return None
            if not isinstance(chunk_text, str):
                return None
            text = text + chunk_text
        return text

    def is_first_token_dangling_contraction(self, inp: Union[List[str], str], is_split_into_words=False, output_str=True) -> Union[List[str], str]:
        """ Checks if a string contains as first token part of a previously split contraction, eg. if a string <<don't play>> has been split into strings <<don>> and <<'t play>>, then <<'t>> is a dangling contraction token."""
        if not isinstance(inp, (str, List, )):
            raise TypeError(f'Argument inp must be a list or a string not a {type(inp)}.')
        if isinstance(inp, List) and not isinstance(inp[0], str):
            raise TypeError(f'Argument inp must be a list of strings and not a list of {type(inp[0])}.')
        if not is_split_into_words:
            if isinstance(inp, List):
                logging.warning(f'You set is_split_into_words={is_split_into_words} but argument inp is already a list.')
            else:
                inp = inp.split()
        if len(inp) == 0:
            return False
        first_token = inp[0]
        first_token_n_chars = len(first_token)
        if first_token_n_chars == 0:
            return False
        first_token_begin_char = first_token[0]
        return first_token_begin_char in ('\u0027', )

    def collate_ignore_chunks(self, data, **kwargs):
        if data is None:
            return None
        n_chunks = len(data)
        prev_chunk_text_ignored = ""
        chunk_list = []
        for i, chunk in enumerate(data):
            chunk_text = chunk.get('text', None)
            if chunk_text is not None:
                meta_tag = chunk.get('meta', None)
                if meta_tag is None or (meta_tag is not None and meta_tag in (self.IGNORE_TAG, )):
                    prev_chunk_text_ignored += chunk_text
                if (meta_tag is not None and not meta_tag in (self.IGNORE_TAG, )):
                    chunk_list.append({'text': prev_chunk_text_ignored, 'userDefined': False})
                    chunk_list.append(chunk)
                    prev_chunk_text_ignored = ""
                elif i == n_chunks-1:
                    chunk_list.append({'text': prev_chunk_text_ignored, 'userDefined': False})
                    prev_chunk_text_ignored = ""
        return chunk_list

    def get_tag(self, data, **kwargs) -> str:
        def default_tag(length: int) -> List[str]:
            if length <= 0:
                return []
            return (['O'] * length)

        if data is None:
            return None
        
        # preprocessing step required to remove googshit
        data = self.collate_ignore_chunks(data)

        tag = []
        for i, chunk in enumerate(data):
            chunk_text = chunk.get('text', None)
            meta_tag   = chunk.get('meta', None)
            if chunk_text is None:
                return None
            chunk_text = self.preprocessing_tag(chunk_text)
            chunk_text_clean = self.remove_special(chunk_text)
            chunk_text_split = chunk_text_clean.split()
            chunk_text_n_words = len(chunk_text_split)
            # HACK(fix correct size of chunk of text to avoid tagging a token being instead a danling contraction and compute correct number of tag require for current chunk of text. Example:
            # chunk text <<\u0027t know my customer user id>> has dangling contraction token \u0027t that is part of previous chunk of text final token, eg '... don'.
            # Token \u0027t will be glued together with a word in the previous chunk of text during intent retrieval.)
            # if self.is_first_token_dangling_contraction(chunk_text_split, is_split_into_words=True):
            #    chunk_text_n_words -= 1
            # makes no sense making a tag with a chunk of text made only of dangling tokens related to previous or an empty (?) chunk of text
            if chunk_text_n_words == 0:
                continue
            if meta_tag is None:
                tag.extend(default_tag(length=chunk_text_n_words))
            else:
                # There is no need to check if it's a IGNORE_TAG because we've removed such chunks during collate_ignore_chunks
                if not meta_tag in (self.IGNORE_TAG, ):
                    # meta_tag is of the form @games, we do remove the '@'
                    tag_name = meta_tag[1:]
                    tag.extend([f'B-{tag_name}'] + [f'I-{tag_name}']*(chunk_text_n_words-1))
        return ' '.join(tag)

    def get(self, json, path, default=None, **kwargs) -> Dict:
        return self.get_reader().get(json, path, default)

    def read(self, glob, regex=None, **kwargs):
        return self.get_reader().get_content(glob=glob, regex=regex)

    def is_intent_disabled(self, intent, **kwargs):
        priority = self.get_reader().get(intent, path='priority')
        if priority is None:
            return True, "is malformed: missing field priority"
        return (priority == -1)

    def is_filtered(self, intent, filter_intents=[]):
        filter_intents = [] if filter_intents is None else filter_intents
        intent_name = self.get_reader().get(intent, path='name')
        return intent_name in filter_intents

    def get_intents(self, **kwargs) -> List[Dict[str, str]]:
        intents = []
        logger.info('Collecting intents.')

        usersays_data: List[str, List[dict]] = list(self.read(glob='intents/*_usersays_*.json'))
        intent_data: List[Tuple[str, dict]] = list(self.read(glob='intents/*.json', regex=r"^((?!.*usersays.*).)*$"))
        assert len(usersays_data) == len(intent_data), f"usersays files {len(usersays_data)} != {len(intent_data)} intent files. There would be disalignments"
        usersays_data.sort(key=lambda x: x[0][:-17])  # trim '_usersays_en.json' or '_usersays_es.json'
        intent_data.sort(key=lambda x: x[0][:-5])  # trim '.json'
        for ((filename_usersays, user_says), (filename_intent, intent)) in tqdm(zip(usersays_data, intent_data)):
            assert filename_usersays[:-17] == filename_intent[:-5], f"Intent file {filename_intent} != usersays file {filename_usersays}"
            is_disabled = self.is_intent_disabled(intent)
            if is_disabled:
                logger.info(f'Skipping file = {filename_usersays}. Cause: Disabled on DialogFlow.')
                continue
            is_filtered = self.is_filtered(intent, filter_intents=kwargs.get("filter_intents"))
            if is_filtered:
                logger.info(f'Skipping file = {filename_usersays}. Cause: Disabled via argument.')
                continue
            label = self.get(intent, path='name')
            if label is None:
                logger.info(f'Cannot find a label / intent name in {filename_intent}. Skipping.')
                continue
            for i, us in enumerate(user_says):
                data_content = self.get(us, path='data')
                sentence = self.preprocessing_intent(self.get_text(data=data_content))
                if sentence is None:
                    logger.info(f'Found something broken in {filename_usersays}. It may be due to a missing data field, or malformed chunk for sentence = {i}. Skipping.')
                    continue
                tag = self.get_tag(data=data_content)
                if tag is not None and len(sentence.split()) != len(tag.split()):
                    raise ValueError(f'Something wrong. Example={i} - Intent={label}: len sentence ({len(sentence.split())}) != len tag ({len(tag.split())}).\n'
                            + f'sentence: <<{sentence}>> - tag: <<{tag}>>')
                intents.append({
                    'text': sentence,
                    'label': label,
                    'tag': tag
                })
        return intents

class IntentWriter:
    def __init__(self, **kwargs):
        super(IntentWriter, self).__init__()
    
    def write(self, data, **kwargs):
        pass

class BaseOutputFormatIntentWriter(IntentWriter):
    def __init__(self, **kwargs):
        super(BaseOutputFormatIntentWriter, self).__init__(**kwargs)

    def write(self, data: List[DialoflowTrainingExample], output_dir: str =None, **kwargs):
        if output_dir is None:
            output_dir = f'./data-{datetime.now().isoformat()}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info(f'Writing out intents to {output_dir}')
        with open(os.path.join(output_dir, 'sent'), 'w+') as sent_fp, \
                open(os.path.join(output_dir, 'label'), 'w+') as label_fp, \
                open(os.path.join(output_dir, 'tag'), 'w+') as tag_fp:
                for i, example in tqdm(enumerate(data)):
                    if not isinstance(example.text, str) or not isinstance(example.label, str):
                        logger.info(f'Found intent name or user sentence not to be of type string. Skipping example = {i}.')
                        continue
                    sent_fp.write(f'{example.text}\n')
                    label_fp.write(f'{example.label}\n')
                    if example.tag is not None:
                        tag_fp.write(f'{example.tag}\n')


class ExampleReader:
    def __init__(self, **kwargs):
        super(ExampleReader, self).__init__()

    def read(self, input_path_or_file: str, **kwargs):
        pass


class BaseExampleReader(ExampleReader):
    def __init__(self, **kwargs):
        super(BaseExampleReader, self).__init__(**kwargs)
        self.DEFAULT_INPUT_FILE = 'sent'

    def read(self, input_path_or_file: str, **kwargs) -> List[str]:
        if not isinstance(input_path_or_file, (str, )):
            raise TypeError(f'input_path_or_file must be a string, but is {type(input_path_or_file)}.')
        if not input_path_or_file:
            raise ValueError(f'input_path_or_file is an empty string!')
        if not (os.path.isdir(input_path_or_file) or os.path.isfile(input_path_or_file)):
            raise ValueError(f'find input_path_or_file does not exist.')
        if os.path.isdir(input_path_or_file):
            input_path_or_file = os.path.join(input_path_or_file, self.DEFAULT_INPUT_FILE)
            if not os.path.isfile(input_path_or_file):
                raise ValueError(f'Cannot find input_path_or_file')
        examples = []
        with open(input_path_or_file) as inp:
            for line in inp:
                line = line.strip()
                examples.append(line)
        return examples


class DialogFlowAgentExport:
    def __init__(self, local_path_or_url, dialogflow=None, content_type='json', **kwargs):
        super(DialogFlowAgentExport, self).__init__()
        readers = {
            'json': JSONIntentReader
        }

        if content_type not in readers:
            raise ValueError(f'Cannot find content_type={content_type}.')

        self.agent_reader   = AgentReader.from_dir_or_url(local_path_or_url=local_path_or_url, dialogflow=dialogflow)
        self.intents_reader = readers[content_type](agent_reader=self.agent_reader, **kwargs)

    def get_intents(self, **kwargs) -> List[Dict[str, str]]:
        return self.intents_reader.get_intents(filter_intents=kwargs.get("filter_intents"))


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
            logger.error("From docs: 'Request failed for any reason'.")
            raise e
        except google.api_core.exceptions.RetryError as e:
            logger.error("From docs: 'Eequest failed due to a retryable error and retry attempts failed'.")
            raise e
        except ValueError as e:
            logger.error("Invalid parameters")
            raise e
        return zip_raw

class DialogFlowIntentClient:
    def __init__(self, project_name, service_account, **kwargs):
        if project_name is None or service_account is None:
            raise ValueError(f'Please provide correct project name and service account file.')
        self.client = dialogflow_v2.IntentsClient.from_service_account_json(service_account)
        self.parent = f"projects/{project_name}/agent"
    
    def list_intents(self, **kwargs):
        intents = self.client.list_intents(
            parent=self.parent,
            intent_view=dialogflow_v2.enums.IntentView.INTENT_VIEW_FULL
        )
        return intents

    def update_intent(self, intent, language_code, **kwargs):
        response = self.client.update_intent(
            intent=intent,
            language_code=language_code
        )
        return response

class OracleClient:
    def __init__(self,
        local_path_or_url,
        service_account,
        **kwargs
    ):
        self.df_client = None
        self.df_intent = None
        self.df_export = None
        self.writer = None
        self.reader = None

        self.local_path_or_url = local_path_or_url
        self.service_account = service_account
        self.actions = {
            'list_intents': {},
            'get_intents': {'content_type': kwargs.pop('content_type')}
        }

    def from_action(self, action: str):
        callback = None
        if (action == "get_intents"):
            if self.df_client is None:
                self.df_client = DialogFlowAgentClient(
                    project_name=self.local_path_or_url,
                    service_account=self.service_account
                )
            if self.df_export is None:
                self.df_export = DialogFlowAgentExport(
                    local_path_or_url=self.local_path_or_url,
                    dialogflow=self.df_client,
                    **dl.get(self.actions, action, {}))
                callback = partial(getattr(self.df_export, action))
        elif (action in ("list_intents", "update_intent")):
            if self.df_intent is None:
                self.df_intent = DialogFlowIntentClient(
                    project_name=self.local_path_or_url,
                    service_account=self.service_account,
                    **dl.get(self.actions, action, {})
                )
            callback = partial(getattr(self.df_intent, action))
        else:
            raise ValueError(f"Unknown action = {action}")

        return callback

class DialogFlowAgent:
    def __init__(self, 
        local_path_or_url: str,
        service_account: str,
        content_type: str ='json',
        output_format: str ='default',
        input_format: str ='default',
        **kwargs
    ):
        writers = {
            'default': BaseOutputFormatIntentWriter
        }
        readers = {
            'default': BaseExampleReader
        }

        if output_format not in writers:
            raise ValueError(f'Cannot find output_format={output_format}.')

        if input_format not in readers:
            raise ValueError(f'Cannot find input_format={input_format}.')

        self.writer = writers[output_format](**kwargs)
        self.reader = readers[input_format](**kwargs)

        self.oracle = OracleClient(
            local_path_or_url=local_path_or_url,
            service_account=service_account,
            content_type=content_type,
            **kwargs
        )

    def save_training_examples(self, examples, output_dir, **kwargs):
        self.writer.write(data=examples, output_dir=output_dir)

    def add_training_examples(self, intent_name: str, input_dir_or_file: str, lang: str ='en', **kwargs) -> tuple:
        # get raw examples
        examples = self.reader.read(input_path_or_file=input_dir_or_file)
        
        # get examples in a format comapatible with DialogFlow
        examples_as_parts = []
        for example in examples:
            parts = [
                dialogflow_v2.types.Intent.TrainingPhrase.Part(text=example),
            ]
            examples_as_parts.append(dialogflow_v2.types.Intent.TrainingPhrase(parts=parts))
        
        # get intent to update
        intents = iter(self.oracle.from_action('list_intents')())

        response = None
        try:
            while (intent := next(intents)) and response is None:
                if intent.display_name == intent_name:
                    intent.training_phrases.extend(examples_as_parts)
                    response = self.oracle.from_action('update_intent')(intent, language_code=lang)
        except StopIteration:
            response = f'Intent name = "{intent_name}" not found!'

        return response, examples, examples_as_parts

    def get_training_examples(self, **kwargs) -> List[DialoflowTrainingExample]:
        df_examples = []
        for example in self.oracle.from_action('get_intents')(filter_intents=kwargs.get("filter_intents")):
            df_examples.append(DialoflowTrainingExample.from_dict(example))
        return df_examples

# dfagent 🤖
The dfagent is a package for the handling of Dialogflow agents. You can for example retrieve training examples and save into a preferred format, or you can use it to update an intent by simply feeding it training examples you stored in a preferred format.

## Overview

* [dfagent/](dfagent) contains all the core code to extend dfagent.

## Install

To install the dfagent package you only need to run [install.sh](install.sh) script.

## Usage
Once dfagent is installed you can simply import it in your code. 

### Save training phrases [remote]

The following snippet illustrates a simple example to get and save training examples from an online Dialogflow agent.

To create a Dialogflow agent you only need that

```Python
import dfagent

agent = dfagent.DialogFlowAgent(
    local_path_or_url='my_gcp_project_id',
    service_account='path/to/sa.json',
    content_type='json',
    output_format='default'
)
```

Then you can get a list of dialogflow examples for saving as follows

```Python
examples = agent.get_training_examples()
agent.save_training_examples(examples, output_dir='path/to/dir')
```

### Update intent with new training phrases [remote]

In the following is a snippet that illustrates an example to update a remote Dialogflow agent using training phrases you stored as a raw text file. Remember that dfagent can be extended to support any input or output file format.

Once you instante a df agent

```Python
import dfagent

agent = dfagent.DialogFlowAgent(
    local_path_or_url='my_gcp_project_id',
    service_account='path/to/sa.json',
    input_format='default',
)
```

You can update your remote Dialoflow agent in that way

```Python
response, raw_examples, df_examples = agent.add_training_examples(
    intent_name='help.cooking',
    input_dir_or_file='path/to/phrases.train',
    lang='en'
)
```

### From local or zip

In case you already have exported your Dialogflow on your local computer, you can give as local_path_or_url the path to the zip or unzipped exported agent.

```Python
import dfagent

agent = dfagent.DialogFlowAgent(
    local_path_or_url='path/to/myagent.zip',
    ...
)
```

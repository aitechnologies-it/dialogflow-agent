# dfagent
Dialogflow agent is a library for online / offline handling of dialogflow agents. You can retrieve training examples and save into the preferred format.

## Overview

* [dfagent/](dfagent) contains all the core code to extend dfagent.

## Install

To install the dfagent package you only need to run [install.sh](install.sh) script.

## Usage

Once dfagent is installed you can simply import it in your code. The following snipped illustrates a simple example to get and save training examples from an online Dialogflow agent.

To create a Dialogflow agent you only need that

```Python
import dfagent

agent = dfagent.DialogFlowAgent(
        local_path_or_url='project_name',
        service_account='path/to/dir/',
        content_type='json',
        output_format='default'
    )
```

Then you can get a list of dialogflow examples as follows and save as follows

```Python
examples = agent.get_training_examples()
agent.save_training_examples(examples, output_dir='path/to/dir')
```

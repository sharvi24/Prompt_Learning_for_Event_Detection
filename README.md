# Prompt_Learning_for_Event_Detection

Event Detection (ED) is a traditional Information Extraction task which includes two consecutive steps:

1. Identification: Given an input sentence, the model should identify how many types
of events are mentioned in the sentence.

3. Localization: For all the event types mentioned in the sentence, the model should
extract an event trigger (a text span in the sentence) that best indicates the occurrence
of the event.

In this work, I have mainly focused on the identification step. The problem essentially
becomes a multi-label classification problem. (Each sentence could contain multiple events,
a single event, or no events.)

Prompt Learning is a new fine-tuning technique for pre-trained language models (PLM), where prompt templates are designed and the model will make decisions based on the probabilities of the verbalizers of each class returned from the PLM. 

Please refer to the report (`stomar2_report.pdf`) for more details on the implementation:

In summary I have:
- Built Sentence-level Event Detection System by fine-tuning openprompt’s ‘PromptForClassification‘ model for 5 epochs on the training set of MAVEN_DATA.
- System build based on prompt learning by designing prompt templates, verbalizers & loss functions
- Achieved micro F1-score of 0.67 for top 10 frequent event types of MAVEN dataset

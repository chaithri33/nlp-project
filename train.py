
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

dataset_path = 'dataset.json'
with open(dataset_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)

input_texts = [entry['text'] for entry in raw_data]
output_summaries = [entry['summary'] for entry in raw_data]

dataset = Dataset.from_dict({'text': input_texts, 'summary': output_summaries})
dataset = dataset.train_test_split(test_size=0.1)

model_identifier = 'facebook/mbart-large-cc25'
tokenizer = AutoTokenizer.from_pretrained(model_identifier)
model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

def preprocess_data(examples):
    tokenized_inputs = tokenizer(examples['text'], max_length=512, padding="max_length", truncation=True)
    tokenized_labels = tokenizer(examples['summary'], max_length=150, padding="max_length", truncation=True)
    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

training_configuration = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=5,
    save_total_limit=2,
    predict_with_generate=True
)

model_trainer = Seq2SeqTrainer(
    model=model,
    args=training_configuration,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer
)

model_trainer.train()

model_save_path = './telugu_summary_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

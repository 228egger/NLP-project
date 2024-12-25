import subprocess
import sys
import wandb
import numpy as np

num_classes = 6

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("evaluate")
install("torchmetrics")
install("transformers")
install("peft")
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import evaluate 
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Функция для вычисления метрик
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'precision': precision['precision'],
        'recall': recall['recall'],
    }

from peft import LoraConfig, get_peft_model

def run_classifier(dataset, tokenizer, model, LLora = False, batch_size = 32, num_epochs=2, ):
    if LLora: 
        config = LoraConfig(
          r=16,
          lora_alpha=32,
          lora_dropout=0.1,
        )

        model = get_peft_model(model, config)
    train_data, test_data = torch.utils.data.random_split(dataset, lengths = [0.8, 0.2])
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        logging_steps = 10,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        save_steps=10_000,
        save_total_limit=2,
        report_to='wandb',
        evaluation_strategy="steps",
        fp16=False,
        prediction_loss_only= False,
        label_names=['labels'],
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics, 
    )
    trainer.can_return_loss = True

    trainer.train()
    return trainer


def run_LLM(dataset, tokenizer, model):
     scores = []

    for data in dataset:
        text, label = data
        # Создание промпта для модели

        # Генерация ответа моделью
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        
        # Токенизация ответа для получения числа
        score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            # Преобразуем ответ в целое число. Убеждаемся, что это действительно число от 0 до 5.
            relevancy_score = int(score_text.strip())
            relevancy_score = max(0, min(5, relevancy_score))  # Ограничиваем диапазон
            scores.append((relevancy_score, label))
        except ValueError:
            # В случае ошибки преобразования добавляем недопустимое значение
            scores.append(None)
    return scores


from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments, losses, trainer
def prepare_training(o_sentences, t_sentences, labels):
    """
    Returns a Hugging Face Dataset with columns 'sentence1', 'sentence2' and 'label' based on input sentences and labels
    
    :param o_sentences: List of original sentences
    :param t_sentences: List of target sentences
    :param labels: List of labels (0 or 1)
    :return: Hugging Face Dataset
    """
    return Dataset.from_list([{"sentence1": o_sentences[i],
                               "sentence2": t_sentences[i],
                               "label": int(labels[i])}
                              for i in range(len(o_sentences))])

def train_model(train_dataset, model, num_epochs, warmup_steps=100, eval_dataset=None, loss_fn=losses.CosineSimilarityLoss,
                output_dir=None, batch_size=128, log_steps=100):
    """
    Docstring for train_model
    
    :param train_dataset: Hugging Face Dataset with columns 'sentence1', 'sentence2' and 'label'
    :param model: model to train
    :param loss_fn: loss function to use
    :param warmup_steps: number of warmup steps
    :param num_epochs: number of epochs to train for
    """
    train_loss = loss_fn(model)

    training_args = SentenceTransformerTrainingArguments(num_train_epochs=num_epochs,
                                                         warmup_steps=warmup_steps,
                                                         per_device_train_batch_size = batch_size,
                                                         output_dir=output_dir,
                                                         logging_first_step=True,
                                                         logging_steps=log_steps)

    trainer.SentenceTransformerTrainer(model=model, args=training_args, train_dataset=train_dataset,
                                       eval_dataset=eval_dataset, loss=train_loss).train()

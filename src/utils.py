from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss

def evaluate_model(model, tokenizer, dl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    losses = []
    for batch in dl:
        batch = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=150)
        labels = torch.tensor([
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch['attention_mask'], batch['input_ids'])]
            ])
        batch['labels'] = labels
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.transpose(1,2), shift_labels)
            num_tokens = torch.sum(shift_labels != -100, dim=1)
            loss_sum = torch.sum(loss, dim=1)
            loss = loss_sum / num_tokens
            losses.append(loss)
    losses = torch.cat(losses)
    return losses

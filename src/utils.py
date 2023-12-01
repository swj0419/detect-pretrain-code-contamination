from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss

def evaluate_model(model, tokenizer, dl):
    # val_bar = tqdm(range(len(dl)))
    losses = []
    for batch in dl:
        batch = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=150)
        labels = torch.tensor([
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch['attention_mask'], batch['input_ids'])]
            ])
        batch['labels'] = labels
        batch = batch.to(model.device)
        
        with torch.no_grad():
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.transpose(1,2), shift_labels)
            num_tokens = torch.sum(shift_labels != -100, dim=1)

            # for i in range(loss.size(dim=0)):
            #     variances.append(torch.var(loss[i][:num_tokens[i]], dim=0, keepdim=True))
            
            loss_sum = torch.sum(loss, dim=1)
            loss = loss_sum / num_tokens
            # print(loss)
            losses.append(loss)
        # val_bar.update(1)
    # variances = torch.cat(variances)
    losses = torch.cat(losses)
    # with torch.no_grad():
    #     outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    #     losses.append(outputs.loss)
    return losses
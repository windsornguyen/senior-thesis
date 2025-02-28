import torch

def validate():
    pass

def train(
    # TODO: Should probably just pass a training state class?
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim, # correct type?
    device: torch.device,
) -> None: # something
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    preds = model(inputs)
    loss = loss_fn(preds.flatten(0, 1), targets.flatten(0, 1)) # TODO: This might depend on the task
    loss.backward()
    optimizer.step()
    # TODO: Also need something for a scheduler here
    
    total_loss += loss.item()
    training_step += 1
    
    metrics = {}
    callback_manager.on_step_end(training_step, loss.item(), metrics)
    
    
    if training_step % cfg.eval_every == 0:
        val_accuracy, avg_val_loss = validate(model, val_loader, loss_fn, cfg.num_categories, device)
        val_metrics = {
            "val_accuracy": val_accuracy,
            "val_loss": avg_val_loss
        }
        callback_manager.on_validation_end(training_step, val_metrics)
        model.train()  # switch back to train mode after validation

    if training_step >= config.total_steps:
        break

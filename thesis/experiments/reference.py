import typer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hydra import initialize, compose
from torch.utils.data import DataLoader, random_split
from synthetics import generate_copy, generate_induction_heads
from plot import prepare_data, plot_loss, plot_accuracy, plot_combined_metrics, exists

app = typer.Typer()


def get_optimizer(optim_name: str):
    try:
        return getattr(optim, optim_name)
    except AttributeError as error:
        raise typer.BadParameter(f"Optimizer '{optim_name}' not found in torch.optim") from error


def get_loss_fn(loss_name: str):
    try:
        return getattr(nn, loss_name)
    except AttributeError as error:
        raise typer.BadParameter(f"Loss function '{loss_name}' not found in torch.nn") from error

# Dataset creation and handling functions
def create_dataset(config):
    return generate_copy(
        num_examples=config.num_examples,
        num_categories=config.num_categories,
        copy_len=config.copy_len,
        blank_len=config.blank_len,
        selective=config.selective,
        seed=config.seed,
    )


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Training and validation functions
def step(model, batch_inputs, batch_targets, optimizer, loss_fn, num_categories):
    optimizer.zero_grad()
    outputs = model(batch_inputs)
    loss = loss_fn(outputs.view(-1, num_categories), batch_targets.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, val_loader, loss_fn, num_categories, device):
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 2)
            val_total += targets.numel()
            val_correct += (predicted == targets).sum().item()
            val_loss += loss_fn(outputs.view(-1, num_categories), targets.view(-1)).item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    return val_accuracy, avg_val_loss


def train(model, train_loader, val_loader, optimizer, loss_fn, config, device):
    model.train()
    training_step, total_loss = 0, 0
    progress_bar = tqdm(total=config.total_steps, desc="Training")
    plot_data = []

    while training_step < config.total_steps:
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            loss = step(model, batch_inputs, batch_targets, optimizer, loss_fn, config.num_categories)
            total_loss += loss
            training_step += 1
            progress_bar.update(1)

            if training_step % config.eval_every == 0:
                avg_loss = total_loss / config.eval_every
                print(f"\nStep {training_step}, Average Training Loss: {avg_loss:.4f}")

                val_accuracy, avg_val_loss = validate(model, val_loader, loss_fn, config.num_categories, device)
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Validation Loss: {avg_val_loss:.4f}")

                plot_data.append(
                    {
                        "step": training_step,
                        "train_loss": avg_loss,
                        "val_loss": avg_val_loss,
                        "val_accuracy": val_accuracy,
                    }
                )
                model.train()
                total_loss = 0

            if training_step >= config.total_steps:
                break

    progress_bar.close()
    return plot_data


# Entry point for the training script using Typer and Hydra
@app.command()
def benchmark(
    config: str = typer.Option(
        "config", help="Configuration file name for training"
    ),
    model: str = typer.Option(
        "model", help="Configuration file name for the model"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed output during training"
    )
):
    with initialize(config_path="conf"):
        config = compose(config_name=config)
        model_config = compose(config_name=model)

    # Create model, optimizer, and loss function using configurations
    model = ...  # Assuming you have model creation logic here
    optimizer = get_optimizer(config.optim)(model.parameters(), lr=config.lr)
    loss_fn = get_loss_fn(config.loss_fn)()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset and DataLoader setup
    dataset = create_dataset(config)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, model_config.bsz)

    typer.echo(
        f"Training with batch size: {model_config.bsz}, optimizer: {config.optim}, loss function: {config.loss_fn}"
    )
    plot_data = train(model, train_loader, val_loader, optimizer, loss_fn, config, device)

    # Prepare output directory and plot data
    output_dir = "logs/plots"
    exists(output_dir)
    df = prepare_data(plot_data)

    # Generate plots using the imported functions
    plot_loss(df, output_dir)
    plot_accuracy(df, output_dir)
    plot_combined_metrics(df, output_dir)


if __name__ == "__main__":
    app()

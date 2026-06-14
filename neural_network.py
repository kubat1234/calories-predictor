import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config_constants import (
    DATA_PATH,
    MAX_TRAIN_ROWS,
    MAX_TEST_ROWS,
    MODELS_DIR,
    PERCENTILE_FILTER,
    RANDOM_SEED,
    TARGET_NAMES,
)
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.tokenize import tokenize

import matplotlib.pyplot as plt


PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
DEFAULT_MODEL_NAME = "pytorch_nn.pt"

EPOCHS = 50
BATCH_SIZE = 128
EMBEDDING_DIM = 192
HIDDEN_DIM = 320
DROPOUT = 0.6
MAX_SEQ_LEN = 200
MAX_VOCAB = 50_000
MIN_TOKEN_FREQ = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-4
MODEL_OUTPUT_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
LOSS_PLOT_PATH = Path("img/neural_network.png")


class RecipeDataset(Dataset):
    def __init__(self, encoded_texts, servings, targets):
        self.encoded_texts = encoded_texts
        self.servings = torch.tensor(servings, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        token_ids = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        serving = self.servings[idx]
        target = self.targets[idx]
        return token_ids, serving, target


class MultiTargetRecipeRegressor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, token_ids, lengths, servings):
        embeddings = self.embedding(token_ids)
        mask = token_ids.ne(PAD_TOKEN_ID).unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        del lengths

        features = torch.cat([pooled, servings.unsqueeze(1)], dim=1)
        return self.regressor(features)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trenuje model PyTorch"
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Wymusza trening",
    )
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    rows = load_training_rows(DATA_PATH)[:MAX_TRAIN_ROWS + MAX_TEST_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    servings = np.array([float(row["servings"]) for row in rows], dtype=np.float32)
    y = np.array(
        [[float(row[target_name]) for target_name in TARGET_NAMES] for row in rows],
        dtype=np.float32,
    )

    return texts, servings, y


def build_vocab(texts, min_token_freq, max_vocab):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    vocab = {
        "<pad>": PAD_TOKEN_ID,
        "<unk>": UNK_TOKEN_ID,
    }

    for token, freq in sorted_tokens:
        if freq < min_token_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_seq_len):
    tokens = tokenize(text)
    if not tokens:
        return [UNK_TOKEN_ID]
    token_ids = [vocab.get(token, UNK_TOKEN_ID) for token in tokens[:max_seq_len]]
    return token_ids if token_ids else [UNK_TOKEN_ID]


def collate_batch(batch):
    token_ids_list, servings, targets = zip(*batch)
    lengths = torch.tensor([len(token_ids) for token_ids in token_ids_list], dtype=torch.long)
    padded = pad_sequence(token_ids_list, batch_first=True, padding_value=PAD_TOKEN_ID)
    servings_tensor = torch.stack(servings)
    targets_tensor = torch.stack(targets)
    return padded, lengths, servings_tensor, targets_tensor


def run_epoch(model, loader, optimizer, criterion, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    total_items = 0

    for token_ids, lengths, servings, targets in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        servings = servings.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            predictions = model(token_ids, lengths, servings)
            loss = criterion(predictions, targets)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = token_ids.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    if total_items == 0:
        return 0.0
    return total_loss / total_items


def is_improvement(current_loss, best_loss, min_delta):
	return current_loss < (best_loss - min_delta)


def evaluate_metrics(model, loader, device, y_mean, y_std):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for token_ids, lengths, servings, targets in loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            servings = servings.to(device)

            predictions = model(token_ids, lengths, servings).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    if not all_predictions:
        return {}, np.array([]), np.array([])

    y_pred_scaled = np.vstack(all_predictions)
    y_true_scaled = np.vstack(all_targets)

    y_pred = y_pred_scaled * y_std + y_mean
    y_true = y_true_scaled * y_std + y_mean

    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    rmse = root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
    return metrics, y_true, y_pred


def snapshot_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def plot_losses(train_losses, val_losses, output_path):
    if plt is None:
        print("Brak matplotlib - pomijam zapis wykresu strat.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, linestyle="--", linewidth=2, label="train loss")
    plt.plot(epochs, val_losses, linestyle="-", linewidth=2, label="val loss")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("Krzywe uczenia")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Zapisano wykres uczenia: {output_path.as_posix()}")


def print_markdown_table(model_name: str, results: list):
    print(f"\nWyniki dla modelu: **{model_name}**")
    print("| Target | MAE | MSE | RMSE | R2 |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Target']} | {r['MAE']:.4f} | {r['MSE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")
    print("", flush=True)


def save_results_to_png(all_results: list, output_filename: str = "results_summary_nn.png"):
    df = pd.DataFrame(all_results)
    if df.empty:
        return
        
    for col in ["MAE", "MSE", "RMSE", "R2"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
    
    fig, ax = plt.subplots(figsize=(12, 1 + len(df) * 0.3))
    ax.axis('tight')
    ax.axis('off')
    
    custom_col_widths = [0.35, 0.15, 0.12, 0.12, 0.12, 0.12]

    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns,
        colWidths=custom_col_widths,
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    plt.title("Zestawienie wyników modelu PyTorch", pad=20, fontsize=14, weight='bold')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nZapisano zestawienie wyników do pliku: {output_filename}")


def main():
    args = parse_args()

    out_path = MODEL_OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.train:
        print(f"Plik {out_path.as_posix()} juz istnieje. Pomijam trening.")
        print("Uzyj flagi -t, aby wymusic retrening.")
        return

    seed_everything(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}")

    print("Wczytuje dane ...")
    texts, servings, targets = load_data()
    
    split_idx = min(MAX_TRAIN_ROWS, len(texts))
    
    X_train_full = texts[:split_idx]
    serv_train_full = servings[:split_idx]
    y_train_full = targets[:split_idx]

    X_test_text = texts[split_idx:split_idx + MAX_TEST_ROWS]
    serv_test = servings[split_idx:split_idx + MAX_TEST_ROWS]
    y_test = targets[split_idx:split_idx + MAX_TEST_ROWS]

    print(f"Dane wczytane. Rozmiar zbioru wstepnie-treningowego: {len(X_train_full):,}, Rozmiar zbioru testowego: {len(X_test_text):,}")

    X_train_text, X_val_text, serv_train, serv_val, y_train, y_val = train_test_split(
        X_train_full,
        serv_train_full,
        y_train_full,
        test_size=0.1,
        random_state=RANDOM_SEED,
    )

    print("Buduje slownik tokenow...")
    vocab = build_vocab(
        X_train_text,
        min_token_freq=MIN_TOKEN_FREQ,
        max_vocab=MAX_VOCAB,
    )
    print(f"Rozmiar slownika: {len(vocab):,}")

    X_train_encoded = [encode_text(text, vocab, MAX_SEQ_LEN) for text in X_train_text]
    X_val_encoded = [encode_text(text, vocab, MAX_SEQ_LEN) for text in X_val_text]
    X_test_encoded = [encode_text(text, vocab, MAX_SEQ_LEN) for text in X_test_text]

    serv_mean = float(serv_train.mean())
    serv_std = float(serv_train.std())
    if serv_std < 1e-8:
        serv_std = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    serv_train_scaled = (serv_train - serv_mean) / serv_std
    serv_val_scaled = (serv_val - serv_mean) / serv_std
    serv_test_scaled = (serv_test - serv_mean) / serv_std
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    train_dataset = RecipeDataset(X_train_encoded, serv_train_scaled, y_train_scaled)
    val_dataset = RecipeDataset(X_val_encoded, serv_val_scaled, y_val_scaled)
    test_dataset = RecipeDataset(X_test_encoded, serv_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = MultiTargetRecipeRegressor(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        output_dim=len(TARGET_NAMES),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("Rozpoczynam trening...")
    start = perf_counter()
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, is_train=False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = snapshot_state_dict(model)

        print(f"Epoka {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    elapsed = perf_counter() - start
    print(f"Trening zakonczony po {elapsed:.1f}s")

    if best_state_dict is None:
        raise RuntimeError("Nie udalo sie wybrac najlepszego modelu po val_loss.")

    model.load_state_dict(best_state_dict)
    print(f"Wybrano model z epoki {best_epoch} (najmniejszy val_loss={best_val_loss:.4f})")

    metrics, _, _ = evaluate_metrics(model, test_loader, device, y_mean, y_std)

    model_name = DEFAULT_MODEL_NAME
    model_metrics = []
    for index, target_name in enumerate(TARGET_NAMES):
        metric_row = {
            "Model": model_name,
            "Target": target_name,
            "MAE": metrics["mae"][index],
            "MSE": metrics["mse"][index],
            "RMSE": metrics["rmse"][index],
            "R2": metrics["r2"][index]
        }
        model_metrics.append(metric_row)

    print_markdown_table(model_name, model_metrics)
    save_results_to_png(model_metrics, output_filename="results_summary_nn.png")

    plot_losses(train_losses, val_losses, LOSS_PLOT_PATH)
    torch.save(best_state_dict, out_path)
    print(f"Zapisano model: {out_path.as_posix()}")

if __name__ == "__main__":
    main()
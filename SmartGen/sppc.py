import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from models1 import TransformerAutoencoder, TimeSeriesDataset1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad(vocab_size, sequences):
    for sequence in sequences:
        if len(sequence) < 40:
            sequence.extend([vocab_size - 1] * (40 - len(sequence)))
    return sequences


def remove_pad(lst):
    for sublist in lst:
        while sublist and sublist[-1] == 0:
            sublist.pop()
    return lst


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def simi_pad(sequences):
    for sequence in sequences:
        if len(sequence) < 40:
            sequence.extend([0] * (40 - len(sequence)))
    return sequences


def make_data(vocab_size, data_file, batch_size=64):
    with open(data_file, 'rb') as file:
        sequence = pickle.load(file)
    data = pad(vocab_size, sequence)
    data = np.array(data)
    dataset = TimeSeriesDataset1(vocab_size, data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        src_emb = self.embedding(src)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory


class ProjectionHead(nn.Module):
    """Maps pooled encoder output to a lower-dim contrastive space."""
    def __init__(self, d_model=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


def augment_sequence(src: torch.Tensor, vocab_size: int,
                     mask_ratio: float = 0.15,
                     crop_ratio: float = 0.2) -> torch.Tensor:
    """
    Two light augmentations on a batch of token sequences.

    1. Random token masking  – replace ~mask_ratio of tokens with (vocab_size-1)
       (the same padding/unknown token used elsewhere in the codebase).
    2. Random crop + re-pad – drop the first crop_ratio * seq_len tokens and
       re-pad on the right so the length stays the same.

    src : (B, seq_len)  int tensor
    returns augmented tensor of the same shape
    """
    B, L = src.shape
    aug = src.clone()

    mask = torch.rand(B, L, device=src.device) < mask_ratio
    aug[mask] = vocab_size - 1

    crop_len = max(1, int(L * crop_ratio))
    start = random.randint(0, crop_len)
    aug = torch.cat([
        aug[:, start:],
        torch.full((B, start), vocab_size - 1, device=src.device)
    ], dim=1)

    return aug


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    """
    z1, z2 : (B, proj_dim) – two views of the same B sequences.
    Normalise → compute cosine similarity matrix → NT-Xent.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device)
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


def simclr_finetune(model: TransformerAutoencoder,
                    data_loader: DataLoader,
                    vocab_size: int,
                    d_model: int = 256,
                    proj_dim: int = 128,
                    num_epochs: int = 5,
                    temperature: float = 0.5,
                    lr: float = 1e-4):
    """
    Fine-tune the encoder with SimCLR on the day's sequences.
    Returns the fine-tuned model and the projection head (both on `device`).
    """
    proj_head = ProjectionHead(d_model=d_model, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()), lr=lr
    )

    model.train()
    proj_head.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in data_loader:
            src, padding_mask, _ = batch
            src = src.to(device)
            padding_mask = padding_mask.to(device)

            src_v1 = augment_sequence(src, vocab_size)
            src_v2 = augment_sequence(src, vocab_size)

            mem1 = model(src_v1, src_key_padding_mask=padding_mask).mean(dim=1)  # (B, d_model)
            mem2 = model(src_v2, src_key_padding_mask=padding_mask).mean(dim=1)

            z1 = proj_head(mem1)
            z2 = proj_head(mem2)

            loss = nt_xent_loss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  [SimCLR] epoch {epoch + 1}/{num_epochs}  loss={total_loss / n_batches:.4f}")

    model.eval()
    proj_head.eval()
    return model, proj_head


def extract_representations(model: TransformerAutoencoder,
                             data_loader: DataLoader) -> np.ndarray:
    """
    Run the encoder in eval mode, mean-pool the memory over the seq dimension,
    and return a 2-D numpy array of shape (N, d_model).
    """
    all_reps = []
    with torch.no_grad():
        for batch in data_loader:
            src, padding_mask, _ = batch
            src = src.to(device)
            padding_mask = padding_mask.to(device)
            memory = model(src, src_key_padding_mask=padding_mask)
            reps = memory.mean(dim=1).cpu().numpy()
            all_reps.append(reps)
    return np.vstack(all_reps)


def cluster_and_select(representations: np.ndarray,
                       text_collection: list,
                       n_clusters: int = None,
                       min_cluster_size: int = 1) -> list:
    """
    K-Means over the learned representations.
    For each cluster pick the sequence whose representation is
    closest (cosine distance) to the cluster centroid.

    n_clusters : if None, heuristic = max(1, N // 10)
    Returns a list of representative raw sequences ready for GSS.
    """
    N = len(text_collection)
    if N == 0:
        return []

    if n_clusters is None:
        n_clusters = max(1, N // 10)
    n_clusters = min(n_clusters, N)

    print(f"  [Clustering] {N} sequences → {n_clusters} clusters")

    kmeans = KMeans(n_clusters=n_clusters, random_state=2024, n_init='auto')
    labels = kmeans.fit_predict(representations)
    centroids = kmeans.cluster_centers_

    representatives = []
    for c in range(n_clusters):
        indices = np.where(labels == c)[0]
        if len(indices) < min_cluster_size:
            continue

        cluster_reps = representations[indices]
        centroid = centroids[c].reshape(1, -1)
        sims = cosine_similarity(cluster_reps, centroid).flatten()
        best_local = np.argmax(sims)
        best_global = indices[best_local]

        representatives.append(text_collection[best_global])

    print(f"  [Clustering] selected {len(representatives)} representative sequences")
    return representatives


def SPPC_select(dataset, ori_env, vocab_size, threshold,
                n_clusters=None,
                simclr_epochs=5,
                proj_dim=128,
                temperature=0.5):
    """
    Updated pipeline:
      1. Load pre-trained Transformer encoder.
      2. SimCLR fine-tuning on the day's sequences (Contrastive Learning).
      3. Extract mean-pooled representations.
      4. K-Means clustering + centroid-closest representative selection (CC).
      5. Save representative sequences for the GSS module.
    """
    setup_seed(2024)
    num_epochs = 15
    d_model = 256

    for day in range(7):
        print(f"\n=== Day {day} ===")

        model = TransformerAutoencoder(
            vocab_size, d_model=d_model, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2
        ).to(device)
        model_name = f"IoT_model/Transformer_{dataset}_{ori_env}_{num_epochs}epoch.pth"
        model.load_state_dict(torch.load(model_name, map_location=device))

        day_select_file = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl'
        with open(day_select_file, 'rb') as f:
            text_collection = pickle.load(f)

        print(f"  Loaded {len(text_collection)} sequences")

        batch_size = min(64, len(text_collection))
        train_loader = make_data(vocab_size, data_file=day_select_file,
                                 batch_size=batch_size)

        model, proj_head = simclr_finetune(
            model, train_loader, vocab_size,
            d_model=d_model, proj_dim=proj_dim,
            num_epochs=simclr_epochs, temperature=temperature
        )

        full_loader = make_data(vocab_size, data_file=day_select_file,
                                batch_size=len(text_collection))
        representations = extract_representations(model, full_loader)

        representative_sequences = cluster_and_select(
            representations, text_collection, n_clusters=n_clusters
        )

        print(f"  Final GSS input: {len(representative_sequences)} sequences")

        out_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_SPPC_th={threshold}.pkl'
        with open(out_path, 'wb') as f_out:
            pickle.dump(representative_sequences, f_out)

        print(f"  Saved → {out_path}")


def similarity_select(dataset, ori_env, threshold):
    for day in range(7):
        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl', 'rb') as f:
            text_collection = pickle.load(f)

        simi_pad(text_collection)
        similarity_matrix = cosine_similarity(text_collection)
        remove_pad(text_collection)

        to_remove = set()
        unique_indices = []
        for i in range(len(text_collection)):
            if i not in to_remove:
                unique_indices.append(i)
                for j in range(i + 1, len(text_collection)):
                    if similarity_matrix[i, j] > threshold:
                        to_remove.add(j)

        deduplicated_collection = [text_collection[i] for i in unique_indices]
        print(deduplicated_collection)
        print(len(deduplicated_collection))

        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_similarity_th={threshold}.pkl', 'wb') as f_out:
            pickle.dump(deduplicated_collection, f_out)

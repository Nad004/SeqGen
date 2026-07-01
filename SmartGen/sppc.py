import os
import pickle
import random
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader

from models1 import TransformerAutoencoder, TimeSeriesDataset1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_SEQ_LEN = 40


# ============================================================
# Compatibility utility functions
# ============================================================

def pad(vocab_size, sequences):
    """
    Compatibility with Code 1.
    Pads in-place using vocab_size - 1.
    """
    for sequence in sequences:
        if len(sequence) < MAX_SEQ_LEN:
            sequence.extend([vocab_size - 1] * (MAX_SEQ_LEN - len(sequence)))
    return sequences


def remove_pad(lst):
    """
    Compatibility with Code 1.
    Removes trailing zeros.
    """
    for sublist in lst:
        while sublist and sublist[-1] == 0:
            sublist.pop()
    return lst


def setup_seed(seed):
    """
    Reproducibility setup.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def simi_pad(sequences):
    """
    Compatibility with Code 1.
    Pads with 0 for raw similarity comparison.
    """
    for sequence in sequences:
        if len(sequence) < MAX_SEQ_LEN:
            sequence.extend([0] * (MAX_SEQ_LEN - len(sequence)))
    return sequences


def _copy_sequences(sequences):
    """
    Avoid modifying the original loaded pickle content accidentally.
    """
    return [list(seq) for seq in sequences]


def _pad_copy(sequences, pad_token, max_len=MAX_SEQ_LEN):
    """
    Safe padding without mutating original sequence objects.
    """
    copied = _copy_sequences(sequences)

    for seq in copied:
        if len(seq) < max_len:
            seq.extend([pad_token] * (max_len - len(seq)))
        elif len(seq) > max_len:
            del seq[max_len:]

    return copied


def make_data(vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=64):
    """
    Same interface as Code 1.

    Improvement:
    - Uses a safe copied version of sequences before padding.
    - Prevents accidental in-place corruption of loaded data.
    """
    with open(data_file, 'rb') as file:
        sequence = pickle.load(file)

    sequence = _copy_sequences(sequence)
    data = pad(vocab_size, sequence)
    data = np.array(data)

    dataset = TimeSeriesDataset1(vocab_size, data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


# ============================================================
# Transformer encoder
# ============================================================

class TransformerAutoencoder(nn.Module):
    """
    Architecture kept compatible with the saved model weights used by Code 1.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_decoder_layers
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        src_emb = self.embedding(src)
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )
        return memory


# ============================================================
# Contrastive representation learning
# ============================================================

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    It maps Transformer representations into a contrastive space.
    The projection head is used only during fine-tuning.
    """

    def __init__(self, d_model=256, proj_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


def _padding_mask_from_src(src, pad_token):
    return src.eq(pad_token)


def masked_mean_pool(memory, padding_mask):
    """
    Padding-aware mean pooling.

    Code 2 used simple mean pooling over all 40 positions.
    That allows padding tokens to influence the representation.

    This version ignores padded positions.
    """
    if padding_mask is None:
        return memory.mean(dim=1)

    valid_mask = (~padding_mask).float().unsqueeze(-1)
    lengths = valid_mask.sum(dim=1).clamp(min=1.0)

    pooled = (memory * valid_mask).sum(dim=1) / lengths
    return pooled


def masked_stat_pool(memory, padding_mask):
    """
    Richer sequence representation:
    - mean pooling
    - max pooling
    - standard deviation pooling
    - last valid hidden state

    This preserves more temporal and semantic information than mean pooling alone.
    """
    if padding_mask is None:
        mean_pool = memory.mean(dim=1)
        max_pool = memory.max(dim=1).values
        std_pool = memory.std(dim=1)
        last_pool = memory[:, -1, :]
        return torch.cat([mean_pool, max_pool, std_pool, last_pool], dim=1)

    valid_mask = (~padding_mask).float().unsqueeze(-1)
    lengths = valid_mask.sum(dim=1).clamp(min=1.0)

    mean_pool = (memory * valid_mask).sum(dim=1) / lengths

    masked_memory = memory.masked_fill(padding_mask.unsqueeze(-1), -1e9)
    max_pool = masked_memory.max(dim=1).values

    centered = (memory - mean_pool.unsqueeze(1)) * valid_mask
    var_pool = (centered ** 2).sum(dim=1) / lengths
    std_pool = torch.sqrt(var_pool + 1e-6)

    valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
    last_indices = valid_lengths - 1
    batch_indices = torch.arange(memory.size(0), device=memory.device)
    last_pool = memory[batch_indices, last_indices]

    return torch.cat([mean_pool, max_pool, std_pool, last_pool], dim=1)


def augment_sequence(src, vocab_size, mask_ratio=0.12, span_ratio=0.10, crop_ratio=0.10):
    """
    Improved sequence augmentation for contrastive learning.

    Compared to Code 2:
    - Does not corrupt padding tokens unnecessarily.
    - Applies token masking.
    - Applies small contiguous span masking.
    - Applies light temporal cropping while preserving sequence order.

    These augmentations encourage the encoder to learn stable behavioral meaning
    rather than memorizing exact token positions.
    """
    pad_token = vocab_size - 1
    B, L = src.shape

    aug = src.clone()

    valid = aug.ne(pad_token)

    # Random token masking.
    token_mask = (torch.rand(B, L, device=src.device) < mask_ratio) & valid
    aug[token_mask] = pad_token

    # Span masking.
    for b in range(B):
        valid_len = int(valid[b].sum().item())
        if valid_len <= 2:
            continue

        span_len = max(1, int(valid_len * span_ratio))
        span_len = min(span_len, valid_len)

        start = random.randint(0, max(0, valid_len - span_len))
        aug[b, start:start + span_len] = pad_token

    # Light crop/re-pad.
    cropped = torch.full_like(aug, pad_token)

    for b in range(B):
        tokens = aug[b][aug[b].ne(pad_token)]
        valid_len = tokens.numel()

        if valid_len == 0:
            continue

        drop = min(max(0, int(valid_len * crop_ratio)), valid_len - 1)

        if drop > 0:
            start = random.randint(0, drop)
            tokens = tokens[start:]

        keep_len = min(tokens.numel(), L)
        cropped[b, :keep_len] = tokens[:keep_len]

    return cropped


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent contrastive loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)

    if B == 0:
        return torch.tensor(0.0, device=z1.device)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device)
    ])

    return F.cross_entropy(sim, labels)


def simclr_finetune(model,
                    data_loader,
                    vocab_size,
                    d_model=256,
                    proj_dim=128,
                    num_epochs=5,
                    temperature=0.5,
                    lr=1e-4,
                    weight_decay=1e-5):
    """
    Fine-tunes the encoder using contrastive learning.

    Improvements over Code 2:
    - Uses padding-aware pooling.
    - Recomputes padding masks after augmentation.
    - Uses LayerNorm and Dropout in projection head.
    - Uses weight decay for robustness.
    - Handles tiny datasets safely.
    """
    if num_epochs <= 0:
        model.eval()
        return model, None

    proj_head = ProjectionHead(d_model=d_model, proj_dim=proj_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    model.train()
    proj_head.train()

    pad_token = vocab_size - 1

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in data_loader:
            src, padding_mask, _ = batch
            src = src.to(device)

            src_v1 = augment_sequence(src, vocab_size)
            src_v2 = augment_sequence(src, vocab_size)

            mask_v1 = _padding_mask_from_src(src_v1, pad_token)
            mask_v2 = _padding_mask_from_src(src_v2, pad_token)

            mem1 = model(src_v1, src_key_padding_mask=mask_v1)
            mem2 = model(src_v2, src_key_padding_mask=mask_v2)

            pooled1 = masked_mean_pool(mem1, mask_v1)
            pooled2 = masked_mean_pool(mem2, mask_v2)

            z1 = proj_head(pooled1)
            z2 = proj_head(pooled2)

            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proj_head.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"  [SimCLR] epoch {epoch + 1}/{num_epochs} loss={avg_loss:.4f}")

    model.eval()
    proj_head.eval()

    return model, proj_head


# ============================================================
# Behavioral feature extraction
# ============================================================

def _sequence_entropy(tokens):
    if len(tokens) == 0:
        return 0.0

    counter = Counter(tokens)
    total = len(tokens)

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)

    max_entropy = math.log(max(2, len(counter)))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _transition_entropy(tokens):
    if len(tokens) < 2:
        return 0.0

    transitions = list(zip(tokens[:-1], tokens[1:]))
    counter = Counter(transitions)
    total = len(transitions)

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)

    max_entropy = math.log(max(2, len(counter)))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def sequence_behavior_features(sequences,
                               vocab_size=None,
                               unigram_dim=64,
                               bigram_dim=128):
    """
    Extracts symbolic behavioral features directly from action sequences.

    Why this helps:
    Transformer embeddings capture learned semantics, but symbolic features
    preserve explicit information about:
    - token frequency
    - transitions
    - diversity
    - repetition
    - sequence length
    - behavioral complexity

    This is useful for LLM-oriented sequence compression because rare but
    meaningful device patterns should not be removed only because their
    neural embedding is close to another sequence.
    """
    features = []

    pad_token = vocab_size - 1 if vocab_size is not None else None

    for seq in sequences:
        seq = list(seq)

        # Remove trailing model pad token if present.
        if pad_token is not None:
            while seq and seq[-1] == pad_token:
                seq.pop()

        length = len(seq)

        if length == 0:
            base = np.zeros(6, dtype=np.float32)
            unigram = np.zeros(unigram_dim, dtype=np.float32)
            bigram = np.zeros(bigram_dim, dtype=np.float32)
            features.append(np.concatenate([base, unigram, bigram]))
            continue

        unique_count = len(set(seq))
        unique_ratio = unique_count / max(1, length)

        entropy = _sequence_entropy(seq)
        trans_entropy = _transition_entropy(seq)

        repeated_adjacent = 0
        for i in range(1, length):
            if seq[i] == seq[i - 1]:
                repeated_adjacent += 1

        repetition_ratio = repeated_adjacent / max(1, length - 1)

        length_norm = min(length, MAX_SEQ_LEN) / MAX_SEQ_LEN

        transition_count = max(0, length - 1)
        transition_density = transition_count / MAX_SEQ_LEN

        base = np.array([
            length_norm,
            unique_ratio,
            entropy,
            trans_entropy,
            repetition_ratio,
            transition_density
        ], dtype=np.float32)

        unigram = np.zeros(unigram_dim, dtype=np.float32)
        for token in seq:
            unigram[int(token) % unigram_dim] += 1.0
        unigram /= max(1.0, unigram.sum())

        bigram = np.zeros(bigram_dim, dtype=np.float32)
        for a, b in zip(seq[:-1], seq[1:]):
            h = (int(a) * 1000003 + int(b)) % bigram_dim
            bigram[h] += 1.0
        if bigram.sum() > 0:
            bigram /= bigram.sum()

        features.append(np.concatenate([base, unigram, bigram]))

    return np.vstack(features).astype(np.float32)


# ============================================================
# Representation extraction and fusion
# ============================================================

def extract_representations(model, data_loader):
    """
    Extracts padding-aware Transformer representations.

    Improvement over Code 2:
    - Uses mean, max, std, and last valid state.
    - Ignores padding positions.
    - Returns richer sequence-level representation.
    """
    model.eval()

    all_reps = []

    with torch.no_grad():
        for batch in data_loader:
            src, padding_mask, _ = batch

            src = src.to(device)
            padding_mask = padding_mask.to(device)

            memory = model(src, src_key_padding_mask=padding_mask)
            reps = masked_stat_pool(memory, padding_mask)

            all_reps.append(reps.cpu().numpy())

    if len(all_reps) == 0:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(all_reps).astype(np.float32)


def fuse_representations(transformer_reps,
                         behavior_reps,
                         transformer_weight=0.75,
                         behavior_weight=0.25):
    """
    Combines neural semantic representations with symbolic behavioral features.

    This improves:
    - semantic preservation
    - temporal pattern retention
    - rare behavior preservation
    - anomaly discrimination
    """
    N = behavior_reps.shape[0]

    if N == 0:
        return behavior_reps

    parts = []

    if transformer_reps is not None and transformer_reps.size > 0:
        if N > 1:
            transformer_reps = StandardScaler().fit_transform(transformer_reps)
        transformer_reps = normalize(transformer_reps)
        parts.append(transformer_weight * transformer_reps)

    if behavior_reps is not None and behavior_reps.size > 0:
        if N > 1:
            behavior_reps = StandardScaler().fit_transform(behavior_reps)
        behavior_reps = normalize(behavior_reps)
        parts.append(behavior_weight * behavior_reps)

    fused = np.concatenate(parts, axis=1)
    fused = normalize(fused)

    return fused.astype(np.float32)


# ============================================================
# High-quality prototype selection
# ============================================================

def _sequence_quality_scores(sequences, similarity_matrix=None, vocab_size=None):
    """
    Scores sequences by informativeness.

    The score favors:
    - representative sequences inside dense regions
    - reasonable length
    - token diversity
    - transition diversity

    Single outliers are not removed automatically; they are preserved as their
    own components during selection.
    """
    N = len(sequences)

    if N == 0:
        return np.array([], dtype=np.float32)

    lengths = np.array([len(seq) for seq in sequences], dtype=np.float32)
    median_len = np.median(lengths) if N > 0 else 1.0
    max_dev = max(1.0, np.max(np.abs(lengths - median_len)))

    length_score = 1.0 - (np.abs(lengths - median_len) / max_dev)
    length_score = np.clip(length_score, 0.0, 1.0)

    diversity_scores = []
    transition_scores = []

    pad_token = vocab_size - 1 if vocab_size is not None else None

    for seq in sequences:
        seq = list(seq)

        if pad_token is not None:
            while seq and seq[-1] == pad_token:
                seq.pop()

        if len(seq) == 0:
            diversity_scores.append(0.0)
            transition_scores.append(0.0)
            continue

        diversity_scores.append(len(set(seq)) / len(seq))

        if len(seq) < 2:
            transition_scores.append(0.0)
        else:
            transitions = list(zip(seq[:-1], seq[1:]))
            transition_scores.append(len(set(transitions)) / len(transitions))

    diversity_scores = np.array(diversity_scores, dtype=np.float32)
    transition_scores = np.array(transition_scores, dtype=np.float32)

    if similarity_matrix is not None and N > 1:
        density = (similarity_matrix.sum(axis=1) - 1.0) / max(1, N - 1)
    else:
        density = np.ones(N, dtype=np.float32)

    quality = (
        0.40 * density +
        0.25 * length_score +
        0.20 * diversity_scores +
        0.15 * transition_scores
    )

    return quality.astype(np.float32)


def semantic_deduplicate_select(representations,
                                text_collection,
                                threshold,
                                vocab_size=None):
    """
    Improved replacement for the original SPPC duplicate removal.

    Original Code 1:
    - Computes cosine similarity over flattened Transformer memory.
    - Greedily removes later sequences above threshold.

    Improved version:
    - Uses fused neural + symbolic representations.
    - Builds similarity components.
    - Selects the best prototype from each component.
    - Preserves rare/outlier components instead of discarding them.
    - Sorts selected indices to preserve original temporal/order structure.

    Output:
    - list of raw original sequences, same format as Code 1.
    """
    N = len(text_collection)

    if N == 0:
        return []

    if N == 1:
        return [text_collection[0]]

    threshold = float(threshold)
    threshold = max(-1.0, min(1.0, threshold))

    sim = cosine_similarity(representations)

    quality = _sequence_quality_scores(
        text_collection,
        similarity_matrix=sim,
        vocab_size=vocab_size
    )

    visited = np.zeros(N, dtype=bool)
    selected_indices = []

    for i in range(N):
        if visited[i]:
            continue

        # Connected component over similarity graph.
        stack = [i]
        component = []

        visited[i] = True

        while stack:
            u = stack.pop()
            component.append(u)

            neighbors = np.where((sim[u] > threshold) & (~visited))[0]

            for v in neighbors:
                visited[v] = True
                stack.append(v)

        component = np.array(component, dtype=int)

        if len(component) == 1:
            selected_indices.append(int(component[0]))
        else:
            # Prefer central and informative sequence.
            component_sim = sim[np.ix_(component, component)]
            centrality = component_sim.mean(axis=1)

            component_quality = quality[component]
            combined = 0.65 * centrality + 0.35 * component_quality

            best_local = int(np.argmax(combined))
            best_global = int(component[best_local])

            selected_indices.append(best_global)

    # Preserve original ordering for downstream temporal interpretation.
    selected_indices = sorted(selected_indices)

    return [text_collection[i] for i in selected_indices]


def cluster_and_select(representations,
                       text_collection,
                       n_clusters=None,
                       min_cluster_size=1):
    """
    Improved version of Code 2 clustering.

    Keeps the same function name and return type.

    Improvements:
    - Uses medoid-like representative selection.
    - Combines centroid similarity with sequence quality.
    - Avoids sklearn n_init='auto' incompatibility.
    - Keeps selected sequences in original order.
    """
    N = len(text_collection)

    if N == 0:
        return []

    if N == 1:
        return [text_collection[0]]

    if n_clusters is None:
        # More stable than N//10 for small and medium datasets.
        n_clusters = max(1, int(round(math.sqrt(N))))

    n_clusters = max(1, min(int(n_clusters), N))

    print(f"  [Clustering] {N} sequences -> {n_clusters} clusters")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=2024,
        n_init=10
    )

    labels = kmeans.fit_predict(representations)
    centroids = kmeans.cluster_centers_

    sim_all = cosine_similarity(representations)
    quality = _sequence_quality_scores(
        text_collection,
        similarity_matrix=sim_all
    )

    selected_indices = []

    for c in range(n_clusters):
        indices = np.where(labels == c)[0]

        if len(indices) < min_cluster_size:
            continue

        cluster_reps = representations[indices]
        centroid = centroids[c].reshape(1, -1)

        centroid_sim = cosine_similarity(cluster_reps, centroid).flatten()

        # Representative should be central, but also informative.
        combined = 0.70 * centroid_sim + 0.30 * quality[indices]

        best_local = int(np.argmax(combined))
        best_global = int(indices[best_local])

        selected_indices.append(best_global)

    selected_indices = sorted(set(selected_indices))

    representatives = [text_collection[i] for i in selected_indices]

    print(f"  [Clustering] selected {len(representatives)} representative sequences")

    return representatives


# ============================================================
# Main compatible SPPC selection function
# ============================================================

def SPPC_select(dataset,
                ori_env,
                vocab_size,
                threshold,
                n_clusters=None,
                simclr_epochs=5,
                proj_dim=128,
                temperature=0.5):
    """
    Drop-in replacement for Code 1 SPPC_select.

    Same required interface:
        SPPC_select(dataset, ori_env, vocab_size, threshold)

    Same output filename as Code 1:
        trn_day_{day}_SPPC_th={threshold}.pkl

    Internal improvements:
    1. Loads the same pretrained Transformer.
    2. Fine-tunes encoder using contrastive learning.
    3. Extracts padding-aware rich Transformer representations.
    4. Extracts symbolic behavioral features.
    5. Fuses neural and symbolic representations.
    6. Selects semantically representative prototypes.
    7. Preserves rare/outlier behaviors instead of blindly removing them.
    """
    setup_seed(2024)

    num_epochs = 15
    d_model = 256

    for day in range(7):
        print(f"\n=== Day {day} ===")

        model = TransformerAutoencoder(
            vocab_size,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        ).to(device)

        model_name = f"IoT_model/Transformer_{dataset}_{ori_env}_{num_epochs}epoch.pth"

        model.load_state_dict(
            torch.load(model_name, map_location=device)
        )

        day_select_file = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl'

        with open(day_select_file, 'rb') as file:
            text_collection = pickle.load(file)

        print(f"  Loaded {len(text_collection)} sequences")

        if len(text_collection) == 0:
            out_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_SPPC_th={threshold}.pkl'
            with open(out_path, 'wb') as f_out:
                pickle.dump([], f_out)
            print(f"  Saved empty output -> {out_path}")
            continue

        batch_size = min(64, max(1, len(text_collection)))

        train_loader = make_data(
            vocab_size,
            data_file=day_select_file,
            batch_size=batch_size
        )

        # Contrastive fine-tuning.
        if len(text_collection) > 1 and simclr_epochs > 0:
            model, _ = simclr_finetune(
                model=model,
                data_loader=train_loader,
                vocab_size=vocab_size,
                d_model=d_model,
                proj_dim=proj_dim,
                num_epochs=simclr_epochs,
                temperature=temperature
            )
        else:
            model.eval()

        full_loader = make_data(
            vocab_size,
            data_file=day_select_file,
            batch_size=len(text_collection)
        )

        transformer_reps = extract_representations(model, full_loader)

        behavior_reps = sequence_behavior_features(
            text_collection,
            vocab_size=vocab_size
        )

        fused_reps = fuse_representations(
            transformer_reps,
            behavior_reps,
            transformer_weight=0.75,
            behavior_weight=0.25
        )

        if n_clusters is None:
            selected_collection = semantic_deduplicate_select(
                fused_reps,
                text_collection,
                threshold=threshold,
                vocab_size=vocab_size
            )
        else:
            selected_collection = cluster_and_select(
                fused_reps,
                text_collection,
                n_clusters=n_clusters
            )

        print(f"  Selected {len(selected_collection)} high-quality sequences")

        # Compatibility-critical filename.
        out_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_SPPC_th={threshold}.pkl'

        with open(out_path, 'wb') as f_out:
            pickle.dump(selected_collection, f_out)

        print(f"  Saved -> {out_path}")


# ============================================================
# Improved similarity_select with same interface/output name
# ============================================================

def similarity_select(dataset, ori_env, threshold):
    """
    Drop-in replacement for Code 1 similarity_select.

    Same interface:
        similarity_select(dataset, ori_env, threshold)

    Same output filename:
        trn_day_{day}_similarity_th={threshold}.pkl

    Improvement:
    Instead of cosine similarity over raw token IDs, which is semantically weak,
    this version compares symbolic behavioral features:
    - unigram action distribution
    - bigram transition distribution
    - length/diversity/entropy features

    Output remains a list of original raw sequences.
    """
    setup_seed(2024)

    for day in range(7):
        day_file = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl'

        with open(day_file, 'rb') as file:
            text_collection = pickle.load(file)

        print(f"\n=== Similarity Day {day} ===")
        print(f"  Loaded {len(text_collection)} sequences")

        if len(text_collection) == 0:
            out_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_similarity_th={threshold}.pkl'
            with open(out_path, 'wb') as f_out:
                pickle.dump([], f_out)
            continue

        # Use behavior-aware features instead of raw padded token-ID vectors.
        behavior_reps = sequence_behavior_features(
            text_collection,
            vocab_size=None
        )

        if len(text_collection) > 1:
            behavior_reps = StandardScaler().fit_transform(behavior_reps)
            behavior_reps = normalize(behavior_reps)

        deduplicated_collection = semantic_deduplicate_select(
            behavior_reps,
            text_collection,
            threshold=threshold,
            vocab_size=None
        )

        print(f"  Selected {len(deduplicated_collection)} sequences")

        out_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_similarity_th={threshold}.pkl'

        with open(out_path, 'wb') as f_out:
            pickle.dump(deduplicated_collection, f_out)

        print(f"  Saved -> {out_path}")

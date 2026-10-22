from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_CANDIDATE_BOUND = 10000000
DATASET_QUERY_BOUND = 500000
MODALITY_TO_INDEX = {"text": 0, "image": 1, "multimodal": 2}


def hash_query_id(value):
    if isinstance(value, str) and ":" in value:
        dataset_id, local_id = value.split(":", 1)
        return int(dataset_id) * DATASET_QUERY_BOUND + int(local_id)
    return int(value)


def hash_item_id(value):
    if isinstance(value, str) and ":" in value:
        dataset_id, local_id = value.split(":", 1)
        return int(dataset_id) * DATASET_CANDIDATE_BOUND + int(local_id)
    return int(value)


def modality_index(value):
    if value is None or value == "":
        return -1
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in MODALITY_TO_INDEX:
            return MODALITY_TO_INDEX[lowered]
    return int(value)


def load_torch_file(path, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = "cpu"
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_tensor(path):
    path = Path(path)
    if path.suffix == ".pt" or path.suffix == ".pth":
        value = load_torch_file(path)
    elif path.suffix == ".npy":
        value = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        value = np.load(path, allow_pickle=True)["arr_0"]
    else:
        raise ValueError(f"Unsupported tensor file: {path}")
    if isinstance(value, dict):
        if {"img", "text", "img_mask", "text_mask"}.issubset(value.keys()):
            img = torch.as_tensor(value["img"], dtype=torch.float32)
            text = torch.as_tensor(value["text"], dtype=torch.float32)
            img_mask = torch.as_tensor(value["img_mask"], dtype=torch.float32).view(-1, 1)
            text_mask = torch.as_tensor(value["text_mask"], dtype=torch.float32).view(-1, 1)
            denom = (img_mask + text_mask).clamp_min(1.0)
            value = (img * img_mask + text * text_mask) / denom
            return value
        keys = sorted(value.keys())
        data = [value[k] for k in keys]
        value = torch.stack([torch.as_tensor(x) for x in data])
    return torch.as_tensor(value, dtype=torch.float32)


class FeatureStore:
    def __init__(self, value):
        self.multimodal = isinstance(value, dict) and {"img", "text", "img_mask", "text_mask"}.issubset(value.keys())
        if self.multimodal:
            self.img = torch.as_tensor(value["img"], dtype=torch.float32)
            self.text = torch.as_tensor(value["text"], dtype=torch.float32)
            self.img_mask = torch.as_tensor(value["img_mask"], dtype=torch.float32).view(-1)
            self.text_mask = torch.as_tensor(value["text_mask"], dtype=torch.float32).view(-1)
            id_to_index = value.get("id_to_index", None)
            self.id_to_index = {int(k): int(v) for k, v in id_to_index.items()} if isinstance(id_to_index, dict) else None
            if self.id_to_index is None:
                self.ids = list(range(int(self.img.shape[0])))
            else:
                self.ids = [None] * int(self.img.shape[0])
                for key, index in self.id_to_index.items():
                    if 0 <= int(index) < len(self.ids):
                        self.ids[int(index)] = int(key)
                self.ids = [int(x) if x is not None else int(i) for i, x in enumerate(self.ids)]
            self.tensor = None
        else:
            self.tensor = torch.as_tensor(value, dtype=torch.float32)
            self.id_to_index = None
            self.ids = list(range(int(self.tensor.shape[0])))

    def __len__(self):
        return len(self.ids)

    def _index(self, key):
        key = int(key)
        if self.id_to_index is not None:
            if key not in self.id_to_index:
                raise KeyError(f"Feature id {key} is not present")
            return int(self.id_to_index[key])
        return key

    def get(self, key):
        index = self._index(key)
        if self.multimodal:
            return {
                "img": self.img[index],
                "text": self.text[index],
                "img_mask": self.img_mask[index],
                "text_mask": self.text_mask[index],
            }
        return self.tensor[index]

    def all(self):
        if self.multimodal:
            return {
                "img": self.img,
                "text": self.text,
                "img_mask": self.img_mask,
                "text_mask": self.text_mask,
            }
        return self.tensor


def load_feature_store(path):
    path = Path(path)
    if path.suffix == ".pt" or path.suffix == ".pth":
        value = load_torch_file(path)
    elif path.suffix == ".npy":
        value = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        value = np.load(path, allow_pickle=True)["arr_0"]
    else:
        raise ValueError(f"Unsupported feature file: {path}")
    return FeatureStore(value)


def load_pairs(path, length=None):
    if path is None:
        if length is None:
            raise ValueError("length is required when pair file is omitted")
        return [(i, i, -1) for i in range(int(length))]
    path = Path(path)
    pairs = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "qid" in row and "pos_cand_list" in row:
                    positives = row.get("pos_cand_list", [])
                    if positives:
                        pairs.append((hash_query_id(row["qid"]), hash_item_id(positives[0]), -1))
                    continue
                modality = row.get("target_modality", row.get("item_modality", row.get("modality", -1)))
                query_id = row.get("query_id", row.get("qid"))
                item_id = row.get("item_id", row.get("did"))
                pairs.append((hash_query_id(query_id), hash_item_id(item_id), modality_index(modality)))
    else:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                modality = row.get("target_modality", row.get("item_modality", row.get("modality", -1)))
                query_id = row.get("query_id", row.get("qid"))
                item_id = row.get("item_id", row.get("did"))
                pairs.append((hash_query_id(query_id), hash_item_id(item_id), modality_index(modality)))
    return pairs


def load_modalities(path, length=None):
    path = Path(path)
    if path.suffix == ".pt" or path.suffix == ".pth":
        values = load_torch_file(path)
    elif path.suffix == ".npy":
        values = np.load(path, allow_pickle=True)
    elif path.suffix == ".npz":
        values = np.load(path, allow_pickle=True)["arr_0"]
    elif path.suffix == ".jsonl":
        rows = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                index = hash_item_id(row.get("item_id", row.get("did", row.get("id"))))
                rows[index] = modality_index(row.get("target_modality", row.get("item_modality", row.get("modality"))))
        keys = sorted(rows.keys())
        values = {int(k): int(rows[k]) for k in keys}
    elif path.suffix == ".csv":
        rows = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                index = hash_item_id(row.get("item_id", row.get("did", row.get("id"))))
                rows[index] = modality_index(row.get("target_modality", row.get("item_modality", row.get("modality"))))
        keys = sorted(rows.keys())
        values = {int(k): int(rows[k]) for k in keys}
    else:
        values = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().replace(",", " ").split()
                if len(parts) == 1:
                    values.append(int(parts[0]))
                elif len(parts) >= 2:
                    values.append(int(parts[-1]))
    if isinstance(values, dict):
        return values
    values = torch.as_tensor(values, dtype=torch.long)
    if length is not None and values.numel() < int(length):
        raise ValueError("modality file has fewer rows than item embeddings")
    return values


def load_qrels(path):
    path = Path(path)
    qrels = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 3:
                qid = hash_query_id(parts[0])
                did = hash_item_id(parts[2])
                qrels.setdefault(qid, set()).add(did)
            elif len(parts) == 2:
                qid = hash_query_id(parts[0])
                did = hash_item_id(parts[1])
                qrels.setdefault(qid, set()).add(did)
    return qrels


class PairEmbeddingDataset(Dataset):
    def __init__(self, query_embeddings, item_embeddings, pairs, item_modalities=None):
        self.query_embeddings = query_embeddings
        self.item_embeddings = item_embeddings
        self.pairs = pairs
        self.item_modalities = item_modalities

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        query_index, item_index, pair_modality = self.pairs[index]
        if self.item_modalities is None:
            modality = int(pair_modality)
        elif isinstance(self.item_modalities, dict):
            modality = int(self.item_modalities.get(int(item_index), int(pair_modality)))
        else:
            if hasattr(self.item_embeddings, "_index"):
                modality = int(self.item_modalities[int(self.item_embeddings._index(item_index))])
            else:
                modality = int(self.item_modalities[int(item_index)])
        if hasattr(self.query_embeddings, "get"):
            query = self.query_embeddings.get(query_index)
        else:
            query = self.query_embeddings[query_index]
        if hasattr(self.item_embeddings, "get"):
            item = self.item_embeddings.get(item_index)
        else:
            item = self.item_embeddings[item_index]
        return query, item, item_index, modality


class SequenceDataset(Dataset):
    def __init__(self, path, max_length):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                values = [int(x) for x in line.strip().replace(",", " ").split() if x]
                if len(values) > 1:
                    self.rows.append(values[-int(max_length):])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = self.rows[index]
        return torch.tensor(values[:-1], dtype=torch.long), torch.tensor(values[-1], dtype=torch.long)

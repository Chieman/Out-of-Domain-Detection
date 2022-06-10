from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from lib.datasets import datasets


def load_best_model(transformer_type, path):
    if transformer_type == "PhoBert":
        model = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    else:
        model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    new_state_dict = {}
    for (k, v) in torch.load(path).items():
        if "classifier" not in k:
            new_state_dict[k[12:]] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, tokenizer


def compute_ood_score(saved_path):
    train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
                                                           # tokenizer=tokenizer,
                                                           dataset_name="Ours")

if __name__ == "__main__":
    model, tokenizer = load_best_model("PhoBert", "/home/an/Documents/out-of-domain/Coding3/debug/back_up/best/best_model.pt")



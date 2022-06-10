from dataclasses import dataclass


@dataclass
class ExpConfig:
    experiment_name: str = "debug"
    data_name: str = "Ours"
    n_labels: int = None
    hidden_dropout_prob: float = 0.1
    bert_type: str = "roberta"
    lr: float = 1e-5
    batch_size: int = 1
    n_workers: int = 8
    n_epochs: int = 5
    overfit_pct: float = 0.0
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    temperature: float = 0.5
    accumulate_grad_batches: int = 1
    score_type: str = 'marginal-mahalanobis-pca'
    data_path: str = '/home/an/Documents/out-of-domain/Coding/data/train/My_Data.json'
    device: str = 'cpu'
    ood_type: str = None
    version: int = None
    balance_classes: bool = False
    gradient_clip_val: float = 0.
    use_checkpoint: bool = True


if __name__ == "__main__":
    a = ExpConfig
    print(getattr(a, "n_workers"))

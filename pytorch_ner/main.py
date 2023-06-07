import logging
from collections import Counter
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from pytorch_ner.dataset import NERCollator, NERDataset
from pytorch_ner.nn_modules.architecture import BiLSTM
from pytorch_ner.nn_modules.embedding import Embedding
from pytorch_ner.nn_modules.linear import LinearHead
from pytorch_ner.nn_modules.rnn import DynamicRNN
from pytorch_ner.prepare_data import (
    get_label2idx,
    get_token2idx,
    prepare_conll_data_format,
)
from pytorch_ner.save import save_model
from pytorch_ner.train import masking, train_loop
from pytorch_ner.utils import kv_flip, to_numpy, set_global_seed, str_to_class


def _get_model(
        _config: Dict[str, Any], 
        _token2idx: Dict[str, int], 
        _label2idx: Dict[str, int], 
        _device: torch.device
    ) -> "BiLSTM":
    # TODO: add more params to config.yaml
    # TODO: add pretrained embeddings
    # TODO: add dropout
    embedding_layer = Embedding(
        num_embeddings=len(_token2idx),
        embedding_dim=_config["model"]["embedding"]["embedding_dim"],
    )

    rnn_layer = DynamicRNN(
        rnn_unit=str_to_class(
            module_name="torch.nn",
            class_name=_config["model"]["rnn"]["rnn_unit"],
        ),
        input_size=_config["model"]["embedding"]["embedding_dim"],  # ref to emb_dim
        hidden_size=_config["model"]["rnn"]["hidden_size"],
        num_layers=_config["model"]["rnn"]["num_layers"],
        dropout=_config["model"]["rnn"]["dropout"],
        bidirectional=_config["model"]["rnn"]["bidirectional"],
    )

    # TODO: add attention if needed in config
    linear_head = LinearHead(
        linear_head=nn.Linear(
            in_features=(
                (2 if _config["model"]["rnn"]["bidirectional"] else 1)
                * _config["model"]["rnn"]["hidden_size"]
            ),
            out_features=len(_label2idx),
        ),
    )

    # TODO: add model architecture in config
    # TODO: add attention if needed
    model = BiLSTM(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        linear_head=linear_head,
    ).to(_device)

    return model


def _load_weights(_config: dict, _model: "BiLSTM") -> "BiLSTM":
    print(_config["save"])
    def get_latest_nonempty_folder(_model_path: Union[Path, str]):
        _model_path = Path(_model_path).parent # UNCLEAN HACK, BECAUSE APPARENTLY LOGGING CHANGES THE PATH
        print(_model_path)
        folders = os.listdir(_model_path)
        folders.sort(reverse=True)
        print(folders)
        for folder in folders:
            _path = _model_path / folder / "model.pth"
            if _path.exists():
                return _path
    
    model_path = _config["save"]["path_to_folder"]
    model_path = get_latest_nonempty_folder(model_path)
    print(model_path)
    _model.load_state_dict(torch.load(model_path))
    return _model


def _get_criterion_and_optimizer(_config, _model) -> Tuple[torch.nn.CrossEntropyLoss, torch.optim.Optimizer]:
    criterion = nn.CrossEntropyLoss(reduction="none")  # hardcoded

    optimizer_type = str_to_class(
        module_name="torch.optim",
        class_name=_config["optimizer"]["optimizer_type"],
    )
    optimizer = optimizer_type(
        params=_model.parameters(),
        **_config["optimizer"]["params"],
    )

    return criterion, optimizer


def _save_index_dicts(_config: dict, _token2idx: dict, _label2idx: dict) -> None:
    with open(_config["data"]["token2idx"]["path"], mode="w") as f:
        yaml.dump(_token2idx, f)
    with open(_config["data"]["label2idx"]["path"], mode="w") as f:
        yaml.dump(_label2idx, f)


def _open_index_dicts(_config: dict) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    with open(_config["data"]["token2idx"]["path"], mode="r") as f:
        _token2idx = yaml.safe_load(f)
    with open(_config["data"]["label2idx"]["path"], mode="r") as f:
        _label2idx = yaml.safe_load(f)
    _idx2token = kv_flip(_token2idx)
    _idx2label = kv_flip(_label2idx)
    return _token2idx, _label2idx, _idx2token, _idx2label


def _get_data(
    _config: Dict[str, Any], 
    _set: str) -> Tuple[List[List[str]], List[List[str]]]:
    _token_seq, _label_seq = prepare_conll_data_format(
        path=_config["data"][f"{_set}_data"]["path"],
        sep=_config["data"][f"{_set}_data"]["sep"],
        lower=_config["data"][f"{_set}_data"]["lower"],
        verbose=_config["data"][f"{_set}_data"]["verbose"],
    )
    return _token_seq, _label_seq

def _train(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Main function to train NER model."""

    # log config
    with open(config["save"]["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    device = torch.device(config["torch"]["device"])
    set_global_seed(config["torch"]["seed"])

    # LOAD DATA

    # tokens / labels sequences

    train_token_seq, train_label_seq = _get_data(config, "train")

    valid_token_seq, valid_label_seq = _get_data(config, "valid")

    if "test_data" in config["data"]:
        test_token_seq, test_label_seq = _get_data(config, "test")

    # token2idx / label2idx

    token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
    label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

    token2idx = get_token2idx(
        token2cnt=token2cnt,
        min_count=config["data"]["token2idx"]["min_count"],
        add_pad=config["data"]["token2idx"]["add_pad"],
        add_unk=config["data"]["token2idx"]["add_unk"],
    )

    label2idx = get_label2idx(label_set=label_set)
    
    _save_index_dicts(config, token2idx, label2idx)


    # datasets

    train_set = NERDataset(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    valid_set = NERDataset(
        token_seq=valid_token_seq,
        label_seq=valid_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    if "test_data" in config["data"]:
        test_set = NERDataset(
            token_seq=test_token_seq,
            label_seq=test_label_seq,
            token2idx=token2idx,
            label2idx=label2idx,
            preprocess=config["dataloader"]["preprocess"],
        )

    # collators

    train_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=config["dataloader"]["percentile"],
    )

    valid_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=100,  # hardcoded
    )

    if "test_data" in config["data"]:
        test_collator = NERCollator(
            token_padding_value=token2idx[config["dataloader"]["token_padding"]],
            label_padding_value=label2idx[config["dataloader"]["label_padding"]],
            percentile=100,  # hardcoded
        )

    # dataloaders

    # TODO: add more params to config.yaml
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,  # hardcoded
        collate_fn=train_collator,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=1,  # hardcoded
        shuffle=False,  # hardcoded
        collate_fn=valid_collator,
    )

    if "test_data" in config["data"]:
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,  # hardcoded
            shuffle=False,  # hardcoded
            collate_fn=test_collator,
        )

    # INIT MODEL

    model = _get_model(config, token2idx, label2idx, device)

    criterion, optimizer = _get_criterion_and_optimizer(config, model)

    # TRAIN MODEL

    train_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader if "test_data" in config["data"] else None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=config["optimizer"]["clip_grad_norm"],
        n_epoch=config["train"]["n_epoch"],
        verbose=config["train"]["verbose"],
        tensorboard=config["train"]["tensorboard"],
        logger=logger,
    )

    # SAVE MODEL

    save_model(
        path_to_folder=config["save"]["path_to_folder"],
        model=model,
        token2idx=token2idx,
        label2idx=label2idx,
        config=config,
        export_onnx=config["save"]["export_onnx"],
    )


def _predict(
    config: Dict[str, Any],
    logger: logging.Logger,
    pred_path: Path = Path("/tmp/preds.txt"),
    interactive=False,
) -> None:
    """Main function to run inference from NER model."""

    # log config
    with open(Path(config["save"]["path_to_config"]), mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    device = torch.device(config["torch"]["device"])
    set_global_seed(config["torch"]["seed"])
    
    # LOAD DATA
    
    train_token_seq, train_label_seq = _get_data(config, "train")

    if "test_data" in config["data"]:
        test_token_seq, test_label_seq = _get_data(config, "test")

    # token2idx / label2idx

    # token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
    # label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

    # token2idx = get_token2idx(
    #     token2cnt=token2cnt,
    #     min_count=config["data"]["token2idx"]["min_count"],
    #     add_pad=config["data"]["token2idx"]["add_pad"],
    #     add_unk=config["data"]["token2idx"]["add_unk"],
    # )

    # label2idx = get_label2idx(label_set=label_set)

    token2idx, label2idx, idx2token, idx2label = _open_index_dicts(config)

    # datasets

    train_set = NERDataset(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    if "test_data" in config["data"]:
        test_set = NERDataset(
            token_seq=test_token_seq,
            label_seq=test_label_seq,
            token2idx=token2idx,
            label2idx=label2idx,
            preprocess=config["dataloader"]["preprocess"],
        )

    # collators

    train_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=config["dataloader"]["percentile"],
    )

    if "test_data" in config["data"]:
        test_collator = NERCollator(
            token_padding_value=token2idx[config["dataloader"]["token_padding"]],
            label_padding_value=label2idx[config["dataloader"]["label_padding"]],
            percentile=100,  # hardcoded
        )

    # dataloaders
    if "test_data" in config["data"]:
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,  # hardcoded
            shuffle=False,  # hardcoded
            collate_fn=test_collator,
        )
    
    # prediction

    model = _get_model(config, token2idx, label2idx, device)
    model = _load_weights(config, model)

    def _make_predictions(
            _model: nn.Module, 
            _dataloader: DataLoader, 
            _device: torch.device, 
            _masking: Callable[[torch.Tensor], torch.Tensor], 
        ) -> List[List[str]]:
        
        _model.eval()

        preds = []
        
        for tokens, labels, lengths in _dataloader:
            tokens, labels, lengths = (
                tokens.to(_device),
                labels.to(_device),
                lengths.to(_device),
            )
        
            mask = _masking(lengths)

            # forward pass
            with torch.no_grad():
                logits = _model(tokens, lengths)
            #     loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
            #     loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

            # make predictions
            # y_true = to_numpy(labels[mask])
            

            y_pred = to_numpy(logits.argmax(dim=-1)[mask])
            preds.append(y_pred)

        # import ipython
        # ipython.embed()
        

        return preds

    # def _save_predictions(_config, _predictions):
    #     ...
    
    # idx2token = kv_flip(token2idx)
    # idx2label = kv_flip(label2idx)

    print(list(idx2token.items())[:40])
    print()
    print(list(idx2label.items())[:40])

    def preds2text(_preds, _token_seq: List[List[str]], _label_seq: List[List[str]]) -> str:
        lines = []
        for pred_sentence, token_sentence, label_sentence in zip(_preds, _token_seq, _label_seq):
            for pred_idx, token, true_label in zip(pred_sentence, token_sentence, label_sentence):
                #token = idx2token[token_idx]
                pred_label = idx2label[pred_idx]
                #true_label = idx2label[label_idx]
                lines.append(f"{token: <20}{pred_label: <20}{true_label: <20}")
        print(f"{pred_idx=}, {token=}, {true_label=}")
        print()
        return "\n".join(lines)
    
    preds = _make_predictions(model, test_loader, device, masking)
    for p in preds[:10]:
        print(p)
    preds = preds2text(preds, test_token_seq, test_label_seq)
    with open(pred_path, 'w') as f:
        f.write(preds)
    print("Predictions succesfully saved.")

    import pdb; pdb.set_trace()

    # _save_predictions(config, predictions)
    #return preds

"""
from pathlib import Path
path_to_config = Path.resolve(Path("../pytorch-ner-onnx/config.yaml"))
from pytorch_ner.config import get_config
from pytorch_ner.logger import get_logger
config  = get_config(path_to_config)
logger = get_logger("/tmp/tmplog")
pred_path = Path("/tmp/preds.txt")
preds = _predict(config, logger, pred_path)
"""

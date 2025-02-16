from copy import deepcopy
from functools import partial
import warnings

import optuna
import wandb
from comet.cli.train import initialize_model, initialize_trainer, read_arguments
from comet.cli.evaluate import evaluate
from pytorch_lightning import seed_everything


def optuna_objective(trial: optuna.trial.Trial, cfg):
    cfg = deepcopy(cfg)

    if cfg.regression_metric is not None:
        model_cfg = cfg.regression_metric.init_args
    elif cfg.referenceless_regression_metric is not None:
        model_cfg = cfg.referenceless_regression_metric.init_args
    elif cfg.ranking_metric is not None:
        model_cfg = cfg.ranking_metric.init_args
    elif cfg.unified_metric is not None:
        model_cfg = cfg.unified_metric.init_args
    else:
        raise Exception("Model configurations missing!")

    # Set the hyperparameters search space
    model_cfg.batch_size = trial.suggest_categorical("batch_size", [1, 8, 16, 32])
    model_cfg.encoder_learning_rate = trial.suggest_float("encoder_learning_rate", 1e-6, 5e-4, log=True)
    model_cfg.learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
    model_cfg.dropout = trial.suggest_float("dropout", 0, 0.5)
    # model_cfg.layerwise_decay = trial.suggest_float("layerwise_decay", 0.9, 1)
    # model_cfg.keep_embeddings_frozen = trial.suggest_categorical("keep_embeddings_frozen", [True, False])
    # model_cfg.nr_frozen_epochs = trial.suggest_float("nr_frozen_epochs", 0.15, 1)
    
    trainer = initialize_trainer(cfg)
    model = initialize_model(cfg)

    trainer.fit(model)
    
    # FIXME: Evaluate the model
    test_data = [
        "data/data-1736919579996/test_wo_bunny_with_label.csv",
        "data/data-1736919579996/test_bunny_with_label.csv"      
    ]
    
    metrics = [evaluate(model, data) for data in test_data]
    
    wandb.finish()

    return trainer.early_stopping_callback.best_score.item()


def hyperopt():
    parser = read_arguments()
    cfg = parser.parse_args()

    seed_everything(cfg.seed_everything)

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(optuna_objective, cfg=cfg), n_trials=20)
    print(study.best_params)


if __name__ == "__main__":
    hyperopt()

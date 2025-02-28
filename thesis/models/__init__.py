from thesis.models.mamba import mamba_configs, Mamba

models_config = {
    "mamba": mamba_configs,
}

model_name_to_cls = {"mamba": Mamba}

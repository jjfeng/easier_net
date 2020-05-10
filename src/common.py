def process_params(param_str, dtype):
    if param_str:
        return [dtype(r) for r in param_str.split(",")]
    else:
        return []


def get_dataset_display_name(dataset):
    if "gene" in dataset:
        return "gene cancer C" if "class" in dataset else "gene cancer R"
    else:
        return dataset.replace("_", " ")

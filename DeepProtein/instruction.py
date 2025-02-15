
def get_example(dataset_name):
    instruction_list = {
        "fluorescence": ["You should return a floating-point number.", "fluorescence intensity"],

        "beta": ["You should return a floating-point number.", "increased activity"],

        "stability": ["You should return a floating-point number.", "protein stability"],

        "solubility": ["You should return an integer (0 or 1) where 0 is not soluble and 1 is soluble", "protein solubility"]

    }
    return instruction_list[dataset_name]
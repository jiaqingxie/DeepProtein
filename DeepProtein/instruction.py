
def get_example(dataset_name):
    instruction_list = {
        "fluorescence": ["Return a floating number as the protein's fluorescence intensity.", "fluorescence intensity"],

        "beta": ["Return a floating number as the protein's increased activity.", "increased activity"],

        "stability": ["Return a floating number as the protein's stability.", "protein stability"],

        "solubility": ["Return 0 or 1 as the protein's solubility. 0 is not soluble and 1 is soluble", "protein solubility"]

    }
    return instruction_list[dataset_name]
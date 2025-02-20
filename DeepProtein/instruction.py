
def get_example(dataset_name):
    instruction_list = {
        "fluorescence": ["You should return a floating-point number.", "fluorescence intensity"],

        "beta": ["You should return a floating-point number.", "increased activity"],

        "stability": ["You should return a floating-point number.", "protein stability"],

        "solubility": ["You should return an integer (0 or 1) where 0 is not soluble and 1 is soluble", "protein solubility"],

        "subcellular": ["You should choose an integer within (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) to indicate the protein's location.", "location"],

        "subcellular_binary": ["You should return an integer (0 or 1) where 0 is membrane-bound and 1 is soluble.", "location"],

        "tap": ["You should return a floating-point number.", "developability"],

    }
    return instruction_list[dataset_name]
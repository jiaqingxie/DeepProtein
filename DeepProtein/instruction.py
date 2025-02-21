
def get_example(dataset_name):
    instruction_list = {
        "fluorescence": ["You should return a floating-point number.", "fluorescence intensity"],

        "beta": ["You should return a floating-point number.", "increased activity"],

        "stability": ["You should return a floating-point number.", "protein stability"],

        "solubility": ["You should return an integer (0 or 1) where 0 is not soluble and 1 is soluble", "protein solubility"],

        "subcellular": ["You should choose an integer within the range [0, 9] to indicate the protein's location.", "location"],

        "subcellular_binary": ["You should return an integer (0 or 1) where 0 is membrane-bound and 1 is soluble.", "location"],

        "tap": ["You should return a floating-point number.", "developability"],

        "SAbDab_Chen": ["You should return a floating-point number.", "developability"],

        "CRISPR": ["You should return a floating-point number.", "repair outcome"],

        "ppi_affinity": ["You should return a floating-point number.", "activity of protein-protein interaction"],

        "yeast_ppi": ["You should return an integer (0 or 1) where 0 is weak and 1 is strong", "activity of protein-protein interaction"],

        "human_ppi": ["You should return an integer (0 or 1) where 0 is weak and 1 is strong", "activity of protein-protein interaction"],

        "fold": ["You should return an integer within the range [0, 1194].", "global structural topology of a protein on the fold level"],

        "secondary": ["You should return an integer within the range [0, 2].",
                 "local structures of protein residues in their natural state"],

    }
    return instruction_list[dataset_name]
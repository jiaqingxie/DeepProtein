
def get_example(dataset_name):
    instruction_list = {
        "beta":  ["Protein fluorescence refers to the phenomenon where certain proteins can emit light of a specific "
                 "wavelength when excited by light of a shorter wavelength. It is a widely used technique to study protein "
                 "structure, dynamics, interactions, and function. The dataset consists of 54,025 protein sequences with real-valued "
                 "groundtruth. You should predict the label, which is the logarithm of fluorescence intensity.", "fluorescence intensity"]
    }
    return instruction_list[dataset_name]
import torch
from torchdrug import datasets
from torchdrug import transforms

truncate_transform = transforms.TruncateProtein(max_length=200, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

#dataset = datasets.BetaLactamase("~/protein-datasets/", atom_feature=None, bond_feature=None, residue_feature="default", transform=transform)
dataset = datasets.Fluorescence("~/protein-datasets/", atom_feature=None, bond_feature=None, residue_feature="default", transform=transform)


train_set, valid_set, test_set = dataset.split()

from torchdrug import core, models, tasks

model = models.ProteinCNN(input_dim=21,
                          hidden_dims=[1024, 1024],
                          kernel_size=5, padding=2, readout="max")


task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="mse", metric=("mae", "rmse", "spearmanr"),
                                normalization=False, num_mlp_layer=2)
                                    
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     batch_size=64, gpus=[0])
solver.train(num_epoch=100)

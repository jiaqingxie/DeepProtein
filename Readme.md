```python
from models.CNN import ProteinResNetForValuePrediction
train_data = FluorescenceDataset("../data", "train")
batch = train_data.collate_fn([train_data[0]])
loss, predictions = model(**batch)
```
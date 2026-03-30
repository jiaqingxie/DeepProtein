from cli_common import build_config, build_parser, initialize_run
from DeepProtein.load_dataset import load_single_dataset
import DeepProtein.ProteinPred as models

if __name__ == "__main__":
    args = build_parser("Protein Prediction with DeepProtein").parse_args()
    path = initialize_run(args, f"Fluorescence + {args.target_encoding}")
    train, val, test = load_single_dataset("Fluorescence", path, args.target_encoding)

    config = build_config(args, cls_hidden_dims=[1024, 1024], extra_config={'multi': False})
    model = models.model_initialize(**config)
    model.train(train, val, test, compute_pos_enc=args.compute_pos_enc)



from cli_common import build_config, build_parser, initialize_run
from DeepProtein.load_dataset import load_pair_dataset
import DeepProtein.PPI as models

if __name__ == '__main__':
    args = build_parser("PPI Prediction with `DeepProtein`").parse_args()
    path = initialize_run(args, f"Yeast_PPI + {args.target_encoding}")
    train, val, test = load_pair_dataset("Yeast_PPI", path, args.target_encoding)
    config = build_config(args, cls_hidden_dims=[512], extra_config={'multi': False, 'binary': True})
    model = models.model_initialize(**config)
    model.train(train, val, test, compute_pos_enc=args.compute_pos_enc)
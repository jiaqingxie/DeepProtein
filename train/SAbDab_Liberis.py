from cli_common import build_config, build_parser, initialize_run
from DeepProtein.load_dataset import load_residue_dataset
import DeepProtein.TokenPred as models

if __name__ == "__main__":
    args = build_parser(
        "Protein Prediction with DeepProtein",
        default_target_encoding='Token_Transformer',
        default_epochs=1,
    ).parse_args()
    path = initialize_run(args, f"SAbDab_Liberis + {args.target_encoding}")
    train, val, test = load_residue_dataset("SAbDab_Liberis", path, args.target_encoding)

    config = build_config(
        args,
        cls_hidden_dims=[1024, 1024],
        extra_config={'multi': False, 'binary': True, 'token': True, 'in_channels': 20},
    )
    model = models.model_initialize(**config)
    model.train(train, val, test, batch_size=args.batch_size)



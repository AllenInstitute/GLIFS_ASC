def add_data_args(parent_parser):
    parser = parent_parser.add_argument_group("data")
    parser.add_argument("--batch_size", type=float, default=0)

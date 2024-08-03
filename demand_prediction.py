import argparse

import toml

from pickup_demand_prediction import single_cell_fcnn, city_wide_cnn
from pickup_demand_prediction import analyse_models


def init_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default='')
    parser.add_argument('--bst', type=int, default=16)
    parser.add_argument('--bstv2', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


def main():
    args = init_args_parser()

    if args.m == 'ANALYSE':
        analyse_models.analyse_sc_fcnn()
        # analyse_models.analyse_cw_cnn()
    elif args.m == 'SINGLE-CELL':
        # Run single cell demand prediction model - use params: --m "SINGLE-CELL" --bst "1" --bstv2 "128" --lr "0.001"
        config = toml.load(r'./config/single_cell_demand_predictor.toml')
        config['batch_size_train'] = args.bst
        config['batch_size_train_v2'] = args.bstv2
        config['learning_rate'] = args.lr
        single_cell_fcnn.run(config)
    elif args.m == 'CITY-WIDE':
        city_wide_cnn.run()



if __name__ == '__main__':
    main()

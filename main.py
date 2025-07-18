import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data, get_il_data
from training import train_gnn
from inference import infer_gnn
from incremental_learning import train_il_gnn
import json

def main():
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    if args.incremental_learning:
        # Load old snapshot and new data for incremental learning
        logging.info("Loading data for incremental learning")
        data_old, data_new = get_il_data(data_config)
        logging.info(f"Running incremental learning")
        train_il_gnn(data_old, data_new, args, data_config)
        
    else:
        logging.info("Retrieving data")
        t1 = time.perf_counter()
        
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
        
        t2 = time.perf_counter()
        logging.info(f"Retrieved data in {t2-t1:.2f}s")

        if args.inference:
            #Inference
            logging.info(f"Running Inference")
            infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
        else:
            #Training
            logging.info(f"Running Training")
            train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

if __name__ == "__main__":
    main()

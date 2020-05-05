import train
import socket
import argparse

p = argparse.ArgumentParser(description="""generate commands""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("-m", "--method", default='none', help="comma separated compression methods to run, defaults to all supported methods")
args = p.parse_args()

if args.method == 'none':
    # Override some hyperparameters to train SGD
    train.config["optimizer_reducer"] = "ExactReducer"
    train.config["optimizer_memory"] = False
    train.config["optimizer_conv_learning_rate"] = 0.1
    train.config["optimizer_learning_rate"] = 0.1

elif args.method == 'powersgd':
    # Override some hyperparameters to train PowerSGD
    train.config["optimizer_reducer"] = "RankKReducer"
    train.config["optimizer_reducer_rank"] = 4
    train.config["optimizer_memory"] = True
    train.config["optimizer_reducer_reuse_query"] = True
    train.config["optimizer_reducer_n_power_iterations"] = 0


hostName = socket.gethostname()
# Configure the worker
train.config["n_workers"] = 8
train.config["rank"] = int(hostName[-2:]) - 33 # number of this worker in [0,4).
if train.config["rank"] == 0:
    print("=======Run Reducer: ", args.method)

# You can customize the outputs of the training script by overriding these members
# train.output_dir = "choose_a_directory"
# train.log_info = your_function_pointer
# train.log_metric = your_metric_function_pointer

# Start training
train.main()

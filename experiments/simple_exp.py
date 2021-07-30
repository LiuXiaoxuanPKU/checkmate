import time
import logging
import shutil
import pathlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from experiments.common.profile.cost_model import CostModel
from experiments.common.definitions import remat_data_dir

from remat.tensorflow2.extraction import dfgraph_from_keras
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi

from remat.core.schedule import OperatorEvaluation
from experiments.common.graph_plotting import render_dfgraph, tensor_plot, plot
from experiments.common.load_keras_model import get_keras_model

DEBUG_SPEED = True

def get_ckpt_layers(schedule_list):
    # if a node is calculated more than once,
    # then the node is rematerialized
    computed_nodes = []
    ckpt_nodes = []
    for s in schedule_list:
        # print(s)
        if isinstance(s, OperatorEvaluation):
            # print(s.id, s.is_backwards, )
            if not s.is_backwards and s.id in computed_nodes:
                ckpt_nodes.append(s.id)
            computed_nodes.append(s.id)
    return set(ckpt_nodes)
    

# load cifar10 dataset
batch_size = 48
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
# x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model_name = "MobileNet"
input_shape = (224, 224, 3)
# load TensorFlow model from Keras applications along with loss function and optimizer
model = get_keras_model(model_name, input_shape=input_shape)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)

platform = 'p32xlarge'
key = "_".join(map(str, [platform, model_name, batch_size, input_shape]))
log_base = remat_data_dir() / "budget_sweep" / key
shutil.rmtree(log_base, ignore_errors=True)
pathlib.Path(log_base).mkdir(parents=True, exist_ok=True)
cost_model = CostModel(model_name, platform, log_base, quantization=5)
cost_model.fit()

# cost_model = None
log_base = "graphs/"
print(len(model.layers))
g, idx_to_name = dfgraph_from_keras(model, batch_size=batch_size, cost_model=cost_model,
                           loss_cpu_cost=0, loss_ram_cost=(4 * batch_size))

# # render_dfgraph(g, log_base, name=model_name)
# result_chen = solve_chen_sqrtn(g, False)
# tensor_plot(g, result_chen.schedule, log_base)
# for s in result_chen.schedule:
#     if isinstance(s, OperatorEvaluation):
#         print(s)
GB = 1000000000
for budget in range(int(3 * GB), int(3.5 * GB), int(0.05 * GB)):
    result_ilp = solve_ilp_gurobi(g, budget)
    if result_ilp.schedule is None:
        continue
    # tensor_plot(g, result_ilp.schedule, log_base)
    print(result_ilp.schedule_aux_data.peak_ram)
    # plot(result_ilp, True, "./graphs/RSU")
    ckpt_layers = get_ckpt_layers(result_ilp.schedule)
    print(ckpt_layers)
    print(idx_to_name)
    budget_in_gb = budget / 1000000000.0
    with open('./results/ilp_' + str(budget_in_gb), 'w') as f:
        for layer_id in ckpt_layers:
            f.write(idx_to_name[layer_id] + '\n')
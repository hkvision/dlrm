import time
import sklearn.metrics
import numpy as np
import cloudpickle
import extend_distributed as ext_dist

def get_node_ip():
    """
    This function is ported from ray to get the ip of the current node. In the settings where
    Ray is not involved, calling ray.services.get_node_ip_address would introduce Ray overhead.
    """
    import socket
    import errno
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet connection.
        s.connect(("8.8.8.8", 80))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()
    return node_ip_address

print("Initializing distributed environment on ", get_node_ip())
ext_dist.init_distributed(backend="ccl")
print("Rank: ", ext_dist.my_rank)
print("Local rank: ", ext_dist.my_local_rank)
print("Size: ", ext_dist.my_size)
print("Local size: ", ext_dist.my_local_size)

with open("saved_estimator.pkl", "rb") as f:
    model_creator, optimizer_creator, loss_creator, scheduler_creator, config = cloudpickle.load(f)

with open("train_data.pkl", "rb") as f:
    train_data_creator, epochs, batch_size, validation_data_creator, validate_batch_size = cloudpickle.load(f)
config["batch_size"] = batch_size
config["validate_batch_size"] = validate_batch_size
config["rank"] = ext_dist.my_local_rank

# # Don't wrap DDP on sparse embedding layers.
dlrm = model_creator(config)
dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
dlrm.top_l = ext_dist.DDP(dlrm.top_l)
for i in range(len(dlrm.emb_dense)):
    dlrm.emb_dense[i] = ext_dist.DDP(dlrm.emb_dense[i])
print(dlrm)

optimizer = optimizer_creator(dlrm, config)
loss = loss_creator  # assume it is an instance
scheduler = scheduler_creator(optimizer, config)
train_ld = train_data_creator(config)
nbatches = len(train_ld)
train_batches = config["train_batches"]
test_batches = config["test_batches"]
print("Total train batches: ", nbatches)
print("Batches to train: ", train_batches)
print("Batches to test: ", test_batches)

# 1 epoch first
total_loss = 0
total_samp = 0
# Can use the same code as in TrainingOperator
for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
    ext_dist.barrier()
    t1 = time.time()

    # forward pass
    Z = dlrm(X, lS_o, lS_i)
    # loss
    E = loss(Z, T)
    L = E.detach().cpu().numpy()
    # scaled error gradient propagation
    # (where we do not accumulate gradients across mini-batches)
    optimizer.zero_grad()
    # backward pass
    E.backward()
    # optimizer
    optimizer.step()
    scheduler.step()

    t2 = time.time()
    T = T.detach().cpu().numpy()
    mbs = T.shape[0]
    total_loss += L * mbs
    total_samp += mbs

    should_print = ((j + 1) % config["print_freq"] == 0) or (j + 1 == nbatches)
    if should_print:
        gL = total_loss / total_samp
        print(
            "Finished training it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                j + 1, nbatches, 1, (t2 - t1) *1000)
            + "loss {:.6f}".format(gL)
        )
        total_loss = 0
        total_samp = 0

    should_test = ((j + 1) % config["test_freq"] == 0) or (j + 1 == train_batches)
    if should_test and validation_data_creator:
        valid_ld = validation_data_creator(config)
        print("Test size: ", len(valid_ld))
        scores = []
        targets = []
        for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(valid_ld):

            # forward pass
            Z_test = dlrm(X_test, lS_o_test, lS_i_test)
            if ext_dist.my_size > 1:
                Z_test = ext_dist.all_gather(Z_test, None)
                T_test = ext_dist.all_gather(T_test, None)
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            # print("S_test: ", S_test.shape)
            # print("T_test: ", T_test.shape)
            scores.append(S_test)
            targets.append(T_test)
            if i + 1 == test_batches:
                break
        scores = np.concatenate(scores, axis=0)
        targets = np.concatenate(targets, axis=0)
        # print("Scores: ", scores.shape)
        # print("Targets: ", targets.shape)
        validation_results = {}
        metrics = {
            'loss': sklearn.metrics.log_loss,
            'roc_auc': sklearn.metrics.roc_auc_score,
            'accuracy': lambda y_true, y_score:
            sklearn.metrics.accuracy_score(
                y_true=y_true,
                y_pred=np.round(y_score)
            ),
        }
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(
                targets,
                scores
            )
        print(
            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 1)
            + " loss {:.6f},".format(validation_results['loss'])
            + " auc {:.4f}".format(validation_results['roc_auc'])
            + " accuracy {:3.3f} %".format(validation_results['accuracy'] * 100)
        )

    if j + 1 == train_batches:  # Make sure all workers stop at the same time
        break

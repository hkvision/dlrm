import time
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

print("Initializing distributed environment")
ext_dist.init_distributed(backend="ccl")
print(get_node_ip())
print("Rank: ", ext_dist.my_rank)
print("Local rank: ", ext_dist.my_local_rank)
print("Size: ", ext_dist.my_size)
print("Local size: ", ext_dist.my_local_size)

with open("saved_estimator.pkl", "rb") as f:
    model_creator, optimizer_creator, loss_creator, scheduler_creator, config = cloudpickle.load(f)

with open("train_data.pkl", "rb") as f:
    train_data_creator, epochs, batch_size = cloudpickle.load(f)
config["batch_size"] = batch_size
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

# 1 epoch first

# Can use the same code as in TrainingOperator
for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
    ext_dist.barrier()
    t1 = time.time()

    # forward pass
    Z = dlrm(X, lS_o, lS_i)
    # loss
    E = loss(Z, T)
    # scaled error gradient propagation
    # (where we do not accumulate gradients across mini-batches)
    optimizer.zero_grad()
    # backward pass
    E.backward()
    # optimizer
    optimizer.step()
    scheduler.step()

    t2 = time.time()

    if j % config["print_freq"] == 0:
        print(
            "Finished training it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                j + 1, nbatches, 1, (t2 - t1) *1000)
        )

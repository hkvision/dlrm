import time
import cloudpickle
import extend_distributed as ext_dist

print("Initializing distributed environment")
ext_dist.init_distributed(backend="ccl")

with open("saved_estimator.pkl", "rb") as f:
    model_creator, optimizer_creator, loss_creator, scheduler_creator, config = cloudpickle.load(f)

with open("train_data.pkl", "rb") as f:
    train_data_creator, epochs, batch_size = cloudpickle.load(f)
config["batch_size"] = batch_size

# Don't wrap DDP on sparse embedding layers.
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

    print(
        "Finished training it {}/{} of epoch {}, {:.2f} ms/it, ".format(
            j + 1, nbatches, 1, (t2 - t1) *1000)
    )

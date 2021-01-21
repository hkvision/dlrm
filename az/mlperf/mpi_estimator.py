import os
import sys
import subprocess
import cloudpickle


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


class MPIEstimator:
    def __init__(self,
                 model_creator,
                 optimizer_creator,
                 loss_creator,
                 scheduler_creator,
                 config,
                 hosts=None,
                 workers_per_node=1,
                 env=None):
        self.hosts = hosts
        self.remote_hosts = []
        self.master = get_node_ip()
        print(self.master)
        self.env = env
        for host in hosts:
            if host != self.master:
                self.remote_hosts.append(host)
        print(self.remote_hosts)
        self.dir = os.getcwd()
        self.workers_per_node = workers_per_node
        with open("saved_estimator.pkl", "wb") as f:
            cloudpickle.dump(
                (model_creator, optimizer_creator, loss_creator, scheduler_creator, config), f)
        # Assumption: all hosts can ssh each other without password; all hosts have the same working directory.
        for host in self.remote_hosts:
            p = subprocess.Popen(["scp", "saved_estimator.pkl",
                                  "root@{}:{}/".format(host, self.dir)])
            os.waitpid(p.pid, 0)
            p = subprocess.Popen(["scp", "train.py",
                                  "root@{}:{}/".format(host, self.dir)])
            os.waitpid(p.pid, 0)

    def fit(self, data_creator, epochs=1, batch_size=32):
        with open("train_data.pkl", "wb") as f:
            cloudpickle.dump((data_creator, epochs, batch_size), f)
        for host in self.remote_hosts:
            p = subprocess.Popen(["scp", "train_data.pkl",
                                  "root@{}:{}/".format(host, self.dir)])
            os.waitpid(p.pid, 0)
        cmd = ['mpiexec.hydra']
        # TODO: make OMP_NUM_THREADS configurable
        mpi_config = "-l -np {} -ppn {} -genv OMP_NUM_THREADS=20 ".format(
            self.workers_per_node * len(self.hosts),
            self.workers_per_node, ",".join(self.hosts))
        if len(self.remote_hosts) > 0:
            mpi_config += "-hosts {}".format(",".join(self.hosts))
        cmd.extend(mpi_config.split())
        # cmd.append("ls")
        cmd.append(sys.executable)
        cmd.append("-u")  # This can print as the program runs
        cmd.append("train.py")
        # print(cmd)
        mpi_env = os.environ.copy()
        mpi_env.update(self.env)
        mpi_env["MASTER_ADDR"] = str(self.master)
        # print(mpi_env)
        process = subprocess.Popen(cmd, env=mpi_env)
        process.wait()

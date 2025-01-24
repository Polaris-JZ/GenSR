from torch.utils.data import Sampler
import math

class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, batch_size=100):
        self.dataset = dataset
        self.num_replicas = num_replicas  # 总 GPU 数量
        self.rank = rank  # 当前 GPU 的 rank
        self.batch_size = batch_size
        self.num_samples = len(dataset)

        assert self.num_samples % self.batch_size == 0, "Dataset size must be divisible by batch size."
        
        self.num_batches = self.num_samples // self.batch_size
        self.batches_per_gpu = math.ceil(self.num_batches / self.num_replicas)

    def __iter__(self):
        # 计算当前 GPU 应该处理的 batch 范围
        start_batch = self.rank * self.batches_per_gpu
        end_batch = min((self.rank + 1) * self.batches_per_gpu, self.num_batches)

        # 生成当前 GPU 负责的样本索引
        indices = []
        for batch_idx in range(start_batch, end_batch):
            batch_start = batch_idx * self.batch_size
            batch_end = batch_start + self.batch_size
            indices.extend(range(batch_start, batch_end))

        return iter(indices)

    def __len__(self):
        # 返回当前 GPU 上的样本数量
        return len(list(self.__iter__()))
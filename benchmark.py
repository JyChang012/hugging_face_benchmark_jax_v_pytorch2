import os
from collections import namedtuple
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import Process, Queue
from timeit import timeit, Timer
from typing import Callable, Optional, Literal

import numpy as np

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.97'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


JaxResult = namedtuple('JaxResult', ('bsize', 'first', 'duration'))
TorchResult = namedtuple('TorchResult', ('bsize', 'duration'))


@dataclass
class CompareConfig:
    mode: Literal['torch', 'jax'] = 'jax'
    bsize: int = 12
    maxlen: int = 256
    compile: bool = True
    donate: bool = True
    q: Optional[Queue] = None


def compare(config: CompareConfig):
    try:
        from transformers import DistilBertModel, FlaxDistilBertModel, DistilBertConfig
        # init config
        xnp = np.random.randint(0, 100, [config.bsize, config.maxlen], dtype=int)

        # config
        mdl_config = DistilBertConfig()
        lr = 1e-3
        iter = 10

        print('EXP start')
        print(config)

        # flax
        if config.mode == 'jax':
            import os
            # os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

            import jax, optax
            import jax.numpy as jnp
            from jax import tree_util

            mdlflax = FlaxDistilBertModel(mdl_config)
            print('params inited')
            xjnp = jax.device_put(xnp)

            def jax_step(params, x):
                def loss_fn(params):
                    y = mdlflax(x, params=params, train=True, dropout_rng=mdlflax.key).last_hidden_state
                    return jnp.sum(y)

                grad_fn = jax.grad(loss_fn)
                g = grad_fn(params)
                g = tree_util.tree_map(lambda a: -lr * a, g)
                params = optax.apply_updates(params, g)
                return params

            if config.compile:
                jax_step = jax.jit(jax_step, donate_argnums=0 if config.donate else tuple())

            def jax_train(iter=iter):
                for _ in range(iter):
                    mdlflax.params = jax_step(mdlflax.params, xjnp)
                jax.block_until_ready(mdlflax.params)

            timer = Timer(stmt='jax_train()', globals=dict(jax_train=partial(jax_train, 1)))
            first_time = timer.timeit(1)
            print(f'jax: first time {first_time}')

            timer = Timer(stmt='jax_train()', globals=dict(jax_train=partial(jax_train, iter)))
            duration = timer.timeit(1) / iter
            print(f'jax: {duration}')
            if config.q:
                config.q.put(JaxResult(config.bsize, first_time, duration))
            # del mdlflax, jax_step, jax_train
            return first_time, duration
        # torch
        if config.mode == 'torch':
            import torch
            import torch.utils.benchmark as benchmark

            num_threads = torch.get_num_threads()

            mdltorch = DistilBertModel(mdl_config)
            xtorch = torch.tensor(xnp, device='cuda:0')
            mdltorch = mdltorch.to(device='cuda:0')
            # PyTorch 2.0 feature
            if config.compile:
                mdltorch = torch.compile(mdltorch)

            mdltorch.train()

            def torch_train():
                for _ in range(iter):
                    x = torch.sum(mdltorch(xtorch).last_hidden_state)
                    x.backward()

                    with torch.no_grad():
                        for p in mdltorch.parameters():
                            p -= lr * p.grad
                            p.grad = None

            # pytorch benchmark timer would handle cuda sync, warmup, etc. automatically for torch
            timer = benchmark.Timer(stmt='torch_train()',
                                    globals=dict(torch_train=torch_train),
                                    num_threads=num_threads)

            measurement = timer.timeit(1)
            duration = measurement.mean / iter

            print(f'torch: {duration}')
            del xtorch, mdltorch
            if config.q:
                config.q.put(TorchResult(config.bsize, duration))
            return duration
    except RuntimeError as e:
        if 'memory' in str(e):
            print('OOM!')
            if config.q:
                if 't' in config.mode:
                    config.q.put(TorchResult(config.bsize, 0))
                else:
                    config.q.put(JaxResult(config.bsize, 0, 0))
        else:
            return (0, 0) if config.mode == 'jax' else 0


def main():
    bsizes = [1, 32, 64, 72, 84, 96, 128]
    lengths = [256]

    jax_first_time = []
    jax_time = []
    jax_unjit_time = []
    torch_time = []
    torch_compile_time = []


    q = Queue()

    for bsize in bsizes:
        for maxlen in lengths:
            config = CompareConfig(bsize=bsize, maxlen=maxlen, q=q)
            p = Process(target=compare, args=(config,))
            p.start()
            p.join()
            result: JaxResult = q.get()
            jax_first_time.append(result.first)
            jax_time.append(result.duration)

            p = Process(target=compare, args=[replace(config, compile=False, mode='torch')])
            p.start()
            p.join()
            result: TorchResult = q.get()
            torch_time.append(result.duration)

            p = Process(target=compare, args=[replace(config, mode='torch')])
            p.start()
            p.join()
            result: TorchResult = q.get()
            torch_compile_time.append(result.duration)

    q.close()
    q.join_thread()

    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 320
    plt.rcParams['savefig.dpi'] = 320

    all_time = jax_first_time, jax_time, jax_unjit_time, torch_time, torch_compile_time
    all_time = map(np.array, all_time)
    jax_first_time, jax_time, jax_unjit_time, torch_time, torch_compile_time = all_time

    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()
    fig: plt.Figure
    fig.set_size_inches(7, 7)
    ax: plt.Axes
    ax = axes[0]
    fig.suptitle('Compare HF Bert model: JAX vs PyTorch')
    ax.plot(bsizes, torch_time, 'b-x', label='torch')
    ax.plot(bsizes, jax_time, 'r-x', label='JAX')
    ax.plot(bsizes, torch_compile_time, 'g-x', label='torch2.0 compile')
    # ax.plot(bsizes, jax_unjit, 'r-+', label='JAX unjit')

    # ax.set_yscale('log')
    ax.set_ylabel('time (s)')
    ax.set_xlabel('batch size')
    ax.legend()

    ax = axes[1]
    ax.plot(bsizes, torch_time / jax_time, 'b-o', label='torch_time / JAX_time')
    ax.plot(bsizes, torch_compile_time / jax_time, 'g-o', label='torch_compile_time / JAX_time')
    ax.set_ylabel('ratio')
    ax.set_xlabel('batch size')
    ax.legend()

    ax = axes[2]
    ax.plot(bsizes, jax_first_time, 'r-o', label='JAX first run')
    ax.set_ylabel('time (s)')
    ax.set_xlabel('batch size')
    ax.legend()

    fig.tight_layout()
    fig.savefig('graph.jpg')


if __name__ == '__main__':
    main()

import multiprocessing
import multiprocessing.pool
import os
import signal
import threading
import typing

import psutil


def kill_child_processes(parent_pid: int, sig: int = signal.SIGTERM):
    """
    Kills all child processes (using ``SIGTERM``) of a process with a given PID.

    Parameters
    ----------
    parent_pid: int
        Process ID of the parent
    sig: int
        Signal to send (``SIGTERM`` by default)
    """
    children = collect_child_processes(parent_pid)
    kill_all_processes_in_list(children, sig)


def collect_child_processes(parent_pid: int) -> typing.List[psutil.Process]:
    """
    Recursively gathers all the child processes of a parent process (up to the default Python recursion depth)
    and casts them to a list.

    Parameters
    ----------
    parent_pid: int
        Process ID of the parent

    Returns
    -------
    typing.List[psutil.Process]
        List of all child processes of the parent process
    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    return parent.children(recursive=True)


def kill_all_processes_in_list(processes: typing.List[psutil.Process], sig: int = signal.SIGTERM):
    """
    Kills all ``psutil`` processes in the input list using the specified termination signal.

    Parameters
    ----------
    processes: typing.List[psutil.Process]
        List of processes to kill
    sig: int
        Termination signal to send to the processes. Default: ``signal.SIGTERM``.
    """
    for process in processes:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            continue


def kill_xfoil_mses_processes(sig: int = signal.SIGTERM):
    # In the rare case that an instance of XFOIL or MSES does not get shut down, force-close any processes with those
    # names
    xfoil_mses_processes = ["xfoil", "mset", "mses", "mplot", "mpolar"]
    xfoil_mses_processes.extend([name + ".exe" for name in xfoil_mses_processes])
    matching_processes = [proc for proc in psutil.process_iter(["name"]) if proc.info["name"] in xfoil_mses_processes]
    kill_all_processes_in_list(matching_processes, sig)


def pool_terminate_multi_tiered(pool: multiprocessing.Pool):
    """
    Multi-tiered multiprocessing pool termination function that tries several pool termination methods in sequence from
    least forceful to most forceful to guarantee that the pool is closed, especially when a significant portion
    of the CPU and RAM are being used. This function is intended for use with asynchronous parallel function calls like
    ``multiprocessing.Pool.apply_async()``, ``multiprocessing.Pool.map_async()``, ``multiprocessing.Pool.imap()``,
    ``multiprocessing.Pool.imap_unordered()``, and ``multiprocessing.Pool.starmap_async()``.

    First, the ``terminate`` and ``join`` methods of ``Pool`` are attempted within a daemon thread with a 5-second
    timeout. If the timeout is reached, each process in the pool is sent a ``CTRL_C_EVENT`` to shut it down somewhat
    forcefully. If the process is still alive, a ``CTRL_BREAK_EVENT`` is sent to shut it down more forcefully. After
    the signals are sent to the processes, the ``terminate`` method is called again.

    .. important::
        The ``multiprocessing.Pool`` context manager must still be used to ensure that the ``close``
        method is still called after the pool is terminated.
    """

    def pool_term_primary():
        """Primary pool termination. This function is attempted first."""
        pool.terminate()
        pool.join()

    def pool_term_last_resort():
        """
        Last-resort pool termination, adapted from https://stackoverflow.com/a/47580796.
        """
        pool._state = multiprocessing.pool.TERMINATE
        pool._worker_handler._state = multiprocessing.pool.TERMINATE
        for p in pool._pool:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            if p.is_alive():
                print(f"Process {p.pid} still alive after sending CTRL+C, sending CTRL+BREAK event...")
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        while any(p.is_alive() for p in pool._pool):
            pass
        pool.terminate()

    term_thread = threading.Thread(target=pool_term_primary)
    term_thread.daemon = True
    term_thread.start()
    term_thread.join(timeout=5)
    if term_thread.is_alive():
        pool_term_last_resort()

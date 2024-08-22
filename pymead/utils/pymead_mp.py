import signal
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
    print(f"In kill XFOIL/MSES process function, {len(matching_processes) = }, {matching_processes = }")
    kill_all_processes_in_list(matching_processes, sig)

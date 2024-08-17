import signal

import psutil


def kill_child_processes(parent_pid: int, sig: int = signal.SIGTERM):
    """
    Kills all child processes (using ``SIGTERM``) of a process with a given PID.
    Most code is from `this StackOverflow answer <https://stackoverflow.com/a/17112379>`_.

    Parameters
    ----------
    parent_pid: int
        Process ID of the parent

    sig: int
        Signal to send (``SIGTERM`` by default)

    Returns
    -------

    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
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
    for proc in matching_processes:
        try:
            proc.send_signal(sig)
        except psutil.NoSuchProcess:
            continue

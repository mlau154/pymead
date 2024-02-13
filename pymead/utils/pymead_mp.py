import signal

import psutil


def kill_child_processes(parent_pid: int, sig: int = signal.SIGTERM):
    """
    Kills all child processes (using ``SIGTERM``) of a process with a given PID.
    All code in the function body is directly pulled from
    `this StackOverflow answer <https://stackoverflow.com/a/17112379>`_

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

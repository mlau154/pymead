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

    # In the rare case that an instance of XFOIL or MSES does not get shut down, force-close any processes with those
    # names
    for proc in psutil.process_iter():
        # Try to get the name of the process. In rare cases, the process might get terminated in the middle of
        # the call to proc.name(). In these cases, simply continue
        try:
            proc_name = proc.name()
        except psutil.NoSuchProcess:
            continue
        if proc_name in ["xfoil.exe", "mses.exe", "xfoil", "mses"]:
            try:
                proc.send_signal(sig)
            except psutil.NoSuchProcess:
                continue

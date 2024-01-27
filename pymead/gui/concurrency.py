"""
Inspired by https://forum.qt.io/topic/110138/show-qprogressbar-with-computationally-heavy-background-process/2
"""

import multiprocessing as mp
from multiprocessing import active_children

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from pymead.utils.multiprocessing import kill_child_processes


class ProgressEmitter(QRunnable):
    """Listens to status of process"""

    class ProgressSignals(QObject):
        progress = pyqtSignal(str, object)
        finished = pyqtSignal(bool)

    def __init__(self, conn, process):
        super().__init__()
        self.conn = conn
        self.process = process
        self.signals = ProgressEmitter.ProgressSignals()
        self.running = True

    def run(self):
        while self.running:
            try:
                if self.conn.poll():
                    try:
                        status, data = self.conn.recv()
                        self.signals.progress.emit(status, data)
                        if status in ["finished"]:
                            self.signals.finished.emit(True)
                            self.running = False
                    except EOFError:
                        self.signals.finished.emit(True)
                        self.running = False
                elif not self.process.is_alive():
                    self.signals.progress.emit("terminated", None)
                    self.running = False
            except (BrokenPipeError, OSError, EOFError):
                print("Terminating!")
                self.signals.finished.emit(False)
                self.running = False


class CPUBoundProcess(QObject):

    def __init__(self, operation, args=(), parent=None):
        super().__init__(parent=parent)

        # Connection pipeline
        self.parent_conn, self.child_conn = mp.Pipe(duplex=True)

        # Process creation
        args = (self.child_conn, *args)
        self.process = mp.Process(target=operation, args=args)

        # Status emitter
        self.progress_emitter = ProgressEmitter(self.parent_conn, self.process)
        self.thread_pool = QThreadPool()

    def start(self):
        self.process.start()
        self.thread_pool.start(self.progress_emitter)

    def terminate(self):
        self.progress_emitter.running = False
        for child in active_children():
            kill_child_processes(child.pid)
        self.process.terminate()

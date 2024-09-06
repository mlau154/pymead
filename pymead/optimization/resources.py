import multiprocessing.connection
import psutil
import time


def display_resources(conn: multiprocessing.connection.Connection or None):

    def send_over_pipe(data: object):
        """
        Connection to the GUI that is only used if ``calculate_aero_data`` is being called directly from the GUI

        Parameters
        ----------
        data: object
            The intermediate information to pass to the GUI, normally a two-element tuple where the first argument
            is a string specifying the kind of data being sent, and the second argument being the actual data
            itself (note that the data must be picklable by the multiprocessing module)

        Returns
        -------

        """
        try:
            if conn is not None:
                conn.send(data)
        except BrokenPipeError:
            pass

    time_array = []
    cpu_percent_array = []
    mem_percent_array = []
    while True:
        time_array.append(0)
        cpu_percent_array.append(psutil.cpu_percent())
        mem_percent_array.append(psutil.virtual_memory().percent)
        for array in [time_array, cpu_percent_array, mem_percent_array]:
            if len(array) > 61:
                array.pop(0)

        send_over_pipe(("resources_update", (time_array, cpu_percent_array, mem_percent_array)))
        time.sleep(1)
        time_array = [t - 1 for t in time_array]

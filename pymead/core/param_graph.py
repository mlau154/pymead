import networkx


class ParamGraph(networkx.DiGraph):
    def __init__(self):
        self.param_list = []
        super().__init__()

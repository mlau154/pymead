from pymead.optimization.pop_chrom import Chromosome


class CustomGASampling:
    """
    To be removed in the future.
    """
    # TODO: remove this class
    def __init__(self, param_dict: dict, mea: dict, genes: list):
        self.param_set = param_dict
        self.mea = mea
        self.genes = genes

    def generate_first_parent(self) -> Chromosome:
        first_parent = Chromosome(self.param_set, population_idx=0, genes=self.genes, mea=self.mea)
        first_parent.generate()
        return first_parent

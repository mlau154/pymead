from pymead.optimization.pop_chrom import CustomGASettings, Chromosome
from pymead.core.mea import MEA


class CustomGASampling:
    def __init__(self, param_dict: dict, ga_settings: CustomGASettings or None, mea: MEA):
        self.param_set = param_dict
        self.ga_settings = ga_settings
        self.mea = mea

    def generate_first_parent(self) -> Chromosome:
        genes, _ = self.mea.extract_parameters()
        first_parent = Chromosome(self.param_set, self.ga_settings, category='parent',
                                  generation=0, population_idx=0, genes=genes, mea=self.mea)
        first_parent.generate()
        return first_parent

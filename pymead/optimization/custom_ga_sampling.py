from pymead.optimization.pop_chrom import CustomGASettings, Chromosome


class CustomGASampling:
    def __init__(self, param_dict: dict, ga_settings: CustomGASettings or None, mea: dict, genes: list):
        self.param_set = param_dict
        self.ga_settings = ga_settings
        self.mea = mea
        self.genes = genes

    def generate_first_parent(self) -> Chromosome:
        first_parent = Chromosome(self.param_set, self.ga_settings, category='parent',
                                  generation=0, population_idx=0, genes=self.genes, mea=self.mea)
        first_parent.generate()
        return first_parent

import torch


class Normalizer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.name = "{}_{}".format("normalizer", str(variables))
        self.variables = variables if type(variables) == list else [variables]

    def forward(self, variable_dict):
        with torch.no_grad():
            self.normalize(variable_dict)

    def normalize(self, variable_dict):
        raise NotImplementedError
        
        

class L2Normalizer(Normalizer):
    def __init__(self, variables):
        super().__init__(variables)

    def normalize(self, variable_dict):
        for v in self.variables:
            var = variable_dict[v + ".real"].data + 1j * variable_dict[v + ".imag"].data
            variable_dict[v + ".real"].data /= torch.linalg.norm(
                var, dim=1, keepdims=True
            )
            variable_dict[v + ".imag"].data /= torch.linalg.norm(
                var, dim=1, keepdims=True
            )
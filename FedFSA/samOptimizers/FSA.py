import torch
import torch.nn.functional as F 
# FedFSA
class FSA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer,cid,rho_default, rho_larger, adaptive=False, **kwargs):
        assert rho_default >= 0.0 and rho_larger >= 0.0, f"Invalid perturbation amplitude, should be non-negative: rho_default({rho_default}) rho_larger({rho_larger})"
        defaults = dict(rho_default=rho_default,rho_larger=rho_larger, adaptive=adaptive, **kwargs)
        super(FSA, self).__init__(params, defaults)
        self.cid = cid
        self.param_groups = base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho_default
            group["rho_larger"] = rho_larger
            group["adaptive"] = adaptive

    @torch.no_grad()
    def first_step(self,max2Layer=[]):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = 1 / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if idx not in max2Layer:#
                    e_w = p.grad * group["rho"] * scale.to(p)
                else:
                    e_w = p.grad * group["rho_larger"] * scale.to(p)
                # original SAM , scale = group["rho"] / (grad_norm + 1e-7)
                # e_w = p.grad * scale.to(p)
                # ASAM 
                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w
    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]), p=2)
        return norm

    def nextDisruptLayer(self, global_params,local_named_params,TopC):
        # Calculate the perturbation sensitivity of each layer, only considering the weights
        perturbation_sensitivity = [0 if any(keyword in name.lower() for keyword in ["bn","downsample","shortcut", "norm", "bias"])\
                else self.compute_perturbation_sensitivity(paramAfter, paramBefore) \
                for (name, paramAfter), paramBefore in zip(local_named_params, global_params)]
        # Normalization and ranking
        normalized_perturbation_sensitivity = self.normalize(torch.tensor(perturbation_sensitivity))
        ranked_perturbation_sensitivity = sorted(enumerate(normalized_perturbation_sensitivity), key=lambda x: x[1] \
                                            if not torch.isnan(x[1]) else float('-inf'), reverse=True)

        # Print perturbation_sensitivity information
        # if(self.cid == 10):
        #     for idx, sensi in enumerate(normalized_perturbation_sensitivity):
        #         print(f"LayerIndex: {idx}, Sensitivity: {sensi}")
        #     print(f"Ranked Sensitivity: {ranked_perturbation_sensitivity}")

        # ############### uncomment code below to switch to FSA_Reverse, more details please see ablation study in our paper###
        # nonzero_elements = []
        # for idx, (position, tensor_value) in enumerate(ranked_perturbation_sensitivity):
        #     if tensor_value.item() == 0.:
        #         break
        #     nonzero_elements.append(position)
        # # Extract the first TopC layers whose values are not zero
        # max2Layer = nonzero_elements[-TopC:]
        # ############### uncomment code above to switch to FSA_Reverse #########################

        ############### uncomment code below to switch to FSA #########################
        max2Layer = [original_idx for idx, (original_idx, _) in enumerate(ranked_perturbation_sensitivity) if idx < TopC]
        ############### uncomment code above to switch to FSA #########################

        # Print selected layers
        # if(self.cid == 10):
        #     print(f"client:{self.cid}\tNextDisruptLayer: {max2Layer}")
        return max2Layer

    # compute perturbation sensitivity
    def compute_perturbation_sensitivity(self,paramAfter,paramBefore):
        return torch.mean((self.normalize(paramAfter) - self.normalize(paramBefore))**2)
    # min-max normalization
    def normalize(self, data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        return (data - min_val) / (max_val - min_val)

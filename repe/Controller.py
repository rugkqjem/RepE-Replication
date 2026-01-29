import torch

class HonestyController:
    def __init__(self,model,reader):
        self.model=model
        self.reader=reader
        self.directions=reader.directions
        self.hooks=[]

    def _get_hook(self,layer,coeff):
        direction=self.directions[layer]
        
        if not isinstance(direction,torch.Tensor):
            direction=torch.tensor(direction,dtype=self.model.dtype)
        direction=direction.to(self.model.device)

        def hook_fn(module,input,output):
            if isinstance(output,tuple):
                hiddens=output[0]
                hiddens+=coeff*direction
                return (hiddens,)+output[1:]
            else:
                output+=coeff*direction
                return output

        return hook_fn

    def set_control(self,layers,coeff=1.0):
        self.reset()
        print(f"Applying honesty control on layers {layers} with coeff {coeff}")
        for layer in layers:
            if layer not in self.directions:
                continue
            target_module=self.model.model.layers[layer]
            hook=target_module.register_forward_hook(self._get_hook(layer,coeff))
            self.hooks.append(hook)

    def reset(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks=[]
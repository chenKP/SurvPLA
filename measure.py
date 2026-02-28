import torch


def get_params(model, ignore_auxiliary_head=True):
    if not ignore_auxiliary_head:
        params = sum([m.numel() for m in model.parameters()])
    else:
        params = sum([m.numel() for k, m in model.named_parameters() if 'auxiliary_head' not in k])
    return params



def get_flops(model, input_shape):
    if hasattr(model, 'flops'):
        return model.flops(input_shape)
    else:
        return get_flops_hook(model, input_shape)


def get_flops_hook(model, input_shapes):

    is_training = model.training
    list_conv, list_linear = [], []

    # hook: Conv2d
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)

    # hook: Linear
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement()
        flops = batch_size * weight_ops
        list_linear.append(flops)

    # 递归注册hook
    def register_hooks(net, hook_handles):
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            elif isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            register_hooks(c, hook_handles)

    hook_handles = []
    register_hooks(model, hook_handles)


    if not isinstance(input_shapes, (tuple, list)):
        raise TypeError("input_shapes: tuple or list ")

    def make_tensor_from_shape(shape):
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"input: tuple or list, not {type(shape)}")
        return torch.rand(1, *shape)

    device = next(model.parameters()).device
    model_inputs = tuple(make_tensor_from_shape(s).to(device) for s in input_shapes)

    model.eval()
    with torch.no_grad():
        _ = model(*model_inputs)

    for h in hook_handles:
        h.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(is_training)
    return total_flops

import flax.nnx as nnx
from .models import register_model

@register_model('resnet')
class ResNet(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(32, kernel_size=(3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, use_running_average=True, rngs=rngs)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        return x

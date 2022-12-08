import torch.nn as nn


class PointNet2Base(nn.Module):
    """
    PointNet++ architecture for segmentation.
    Using library by erikwijmans:
    https://github.com/erikwijmans/Pointnet2_PyTorch
    """

    def __init__(self, config):
        super().__init__()

        self.ncomponents = config.ncomponents
        self.classes = config.classes
        self.model_type = config.model_type

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, x):

        xyz, features = self._break_up_pc(x)
        l_xyz, l_features = [xyz], [features]

        # Self abstraction modules with skip link concatenation
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Feature propagation layers using skip link concatenations
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        output = self.head(l_features[0])
        return output

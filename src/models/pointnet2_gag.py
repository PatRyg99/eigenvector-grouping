import torch.nn as nn

from src.models.pointnet2_base import PointNet2Base
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import (
    PointnetFPModuleGAG,
    PointnetSAModuleGAG,
    PointnetSAModuleGAGKNN,
)


class PointNet2GAG(PointNet2Base):
    """
    PointNet++ with geometry-aware grouping (GAG)
    """

    def __init__(self, config):
        super().__init__(config)

        build_model = {"radius": self.build_radius, "knn": self.build_knn}
        build_model[self.model_type](
            config.classes, config.features, config.components
        )

    def build_radius(self, classes: int, features: int, ncomponents: int):
        
        # Set abstraction modules
        self.SA_modules = nn.ModuleList(
            [
                PointnetSAModuleGAG(
                    npoint=2048,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    radii=[0.05, 0.1],
                    nsamples=[64, 128],
                    mlps=[[features, 16, 16, 32], [features, 32, 32, 64]],
                ),
                PointnetSAModuleGAG(
                    npoint=1024,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    radii=[0.1, 0.2],
                    nsamples=[64, 128],
                    mlps=[[32 + 64, 64, 64, 96], [32 + 64, 64, 96, 96]],
                ),
                PointnetSAModuleGAG(
                    npoint=256,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    radii=[0.2, 0.4],
                    nsamples=[64, 128],
                    mlps=[[96 + 96, 64, 64, 128], [96 + 96, 64, 96, 128]],
                ),
                PointnetSAModuleGAG(
                    npoint=64,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    radii=[0.4, 0.6],
                    nsamples=[64, 128],
                    mlps=[[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                ),
                PointnetSAModuleGAG(
                    npoint=16,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    radii=[0.4, 0.8],
                    nsamples=[64, 128],
                    mlps=[[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
                ),
            ]
        )
        
        # Feature propagation module
        self.FP_modules = nn.ModuleList(
            [
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[256 + features, 128, 128],
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25, ncomponents=ncomponents, mlp=[256 + 32 + 64, 256, 256]
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25, ncomponents=ncomponents, mlp=[512 + 96 + 96, 256, 256]
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[512 + 128 + 128, 512, 512],
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[512 + 512 + 256 + 256, 512, 512],
                ),
            ]
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, classes, kernel_size=1),
        )

    def build_knn(self, classes: int, features: int, ncomponents: int):
        
        # Set abstraction modules
        self.SA_modules = nn.ModuleList(
            [
                PointnetSAModuleGAGKNN(
                    npoint=2048,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    nsamples=[256, 512],
                    mlps=[[features, 16, 16, 32], [features, 32, 32, 64]],
                ),
                PointnetSAModuleGAGKNN(
                    npoint=1024,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    nsamples=[64, 128],
                    mlps=[[32 + 64, 64, 64, 96], [32 + 64, 64, 96, 96]],
                ),
                PointnetSAModuleGAGKNN(
                    npoint=256,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    nsamples=[32, 64],
                    mlps=[[96 + 96, 64, 64, 128], [96 + 96, 64, 96, 128]],
                ),
                PointnetSAModuleGAGKNN(
                    npoint=64,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    nsamples=[16, 32],
                    mlps=[[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                ),
                PointnetSAModuleGAGKNN(
                    npoint=16,
                    ncomponents=ncomponents,
                    _lambda=0.25,
                    nsamples=[8, 16],
                    mlps=[[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
                ),
            ]
        )

        # Feature propagation module
        self.FP_modules = nn.ModuleList(
            [
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[256 + features, 128, 128],
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25, ncomponents=ncomponents, mlp=[256 + 32 + 64, 256, 256]
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25, ncomponents=ncomponents, mlp=[512 + 96 + 96, 256, 256]
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[512 + 128 + 128, 512, 512],
                ),
                PointnetFPModuleGAG(
                    _lambda=0.25,
                    ncomponents=ncomponents,
                    mlp=[512 + 512 + 256 + 256, 512, 512],
                ),
            ]
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, classes, kernel_size=1),
        )

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

        output = self.head(l_features[0][:, self.ncomponents :, :])
        return output

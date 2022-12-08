import torch.nn as nn

from src.models.pointnet2_base import PointNet2Base
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import (
    PointnetFPModule,
    PointnetSAModuleMSG,
    PointnetSAModuleMSGKNN,
)


class PointNet2MSG(PointNet2Base):
    """
    PointNet++ with multi-scale grouping (MSG)
    """

    def __init__(self, config):
        super().__init__(config)

        build_model = {"radius": self.build_radius, "knn": self.build_knn}
        build_model[self.model_type](
            config.classes, config.features
        )

    def build_radius(self, classes: int, features: int):

        # Set abstraction modules
        self.SA_modules = nn.ModuleList(
            [
                PointnetSAModuleMSG(
                    npoint=2048,
                    radii=[0.05, 0.1],
                    nsamples=[64, 128],
                    mlps=[[features, 16, 16, 32], [features, 32, 32, 64]],
                ),
                PointnetSAModuleMSG(
                    npoint=1024,
                    radii=[0.1, 0.2],
                    nsamples=[64, 128],
                    mlps=[[32 + 64, 64, 64, 96], [32 + 64, 64, 96, 96]],
                ),
                PointnetSAModuleMSG(
                    npoint=256,
                    radii=[0.2, 0.4],
                    nsamples=[64, 128],
                    mlps=[[96 + 96, 64, 64, 128], [96 + 96, 64, 96, 128]],
                ),
                PointnetSAModuleMSG(
                    npoint=64,
                    radii=[0.4, 0.6],
                    nsamples=[64, 128],
                    mlps=[[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                ),
                PointnetSAModuleMSG(
                    npoint=16,
                    radii=[0.4, 0.8],
                    nsamples=[64, 128],
                    mlps=[[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
                ),
            ]
        )

        # Feature propagator modules
        self.FP_modules = nn.ModuleList(
            [
                PointnetFPModule(mlp=[256 + features, 128, 128]),
                PointnetFPModule(mlp=[256 + 32 + 64, 256, 256]),
                PointnetFPModule(mlp=[512 + 96 + 96, 256, 256]),
                PointnetFPModule(mlp=[512 + 128 + 128, 512, 512]),
                PointnetFPModule(mlp=[512 + 512 + 256 + 256, 512, 512]),
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

    def build_knn(self, classes: int, features: int):
        
        # Set abstraction modules
        self.SA_modules = nn.ModuleList(
            [
                PointnetSAModuleMSGKNN(
                    npoint=2048,
                    nsamples=[256, 512],
                    mlps=[[features, 16, 16, 32], [features, 32, 32, 64]],
                ),
                PointnetSAModuleMSGKNN(
                    npoint=1024,
                    nsamples=[64, 128],
                    mlps=[[32 + 64, 64, 64, 96], [32 + 64, 64, 96, 96]],
                ),
                PointnetSAModuleMSGKNN(
                    npoint=256,
                    nsamples=[32, 64],
                    mlps=[[96 + 96, 64, 64, 128], [96 + 96, 64, 96, 128]],
                ),
                PointnetSAModuleMSGKNN(
                    npoint=64,
                    nsamples=[16, 32],
                    mlps=[[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                ),
                PointnetSAModuleMSGKNN(
                    npoint=16,
                    nsamples=[8, 16],
                    mlps=[[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
                ),
            ]
        )

        # Feature propagator modules
        self.FP_modules = nn.ModuleList(
            [
                PointnetFPModule(mlp=[256 + features, 128, 128]),
                PointnetFPModule(mlp=[256 + 32 + 64, 256, 256]),
                PointnetFPModule(mlp=[512 + 96 + 96, 256, 256]),
                PointnetFPModule(mlp=[512 + 128 + 128, 512, 512]),
                PointnetFPModule(mlp=[512 + 512 + 256 + 256, 512, 512]),
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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


# MSG
class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet SA MSG layer with radius grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModuleMSGKNN(PointnetSAModuleMSG):
    r"""Pointnet SA MSG layer with knn grouping

    Parameters
    ----------
    npoint : int
        Number of features
    nsamples : list of int32
        Number of samples in each knn query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, npoint, nsamples, mlps, bn=True, use_xyz=True,
    ):
        nn.Module.__init__(self)

        assert len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for nsample, mlp in zip(nsamples, mlps):
            self.groupers.append(
                pointnet2_utils.KNNQueryAndGroup(nsample, use_xyz=use_xyz,)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )

            if use_xyz:
                mlp[0] += 3

            self.mlps.append(build_shared_mlp(mlp, bn))


# GAG
class PointnetSAModuleGAG(nn.Module):
    r"""Pointnet SA GAG with radius grouping

    Parameters
    ----------
    npoint : int
        Number of features
    ncomponents : int
        Number of component ids per point used in grouping
    _lambda : float
        Geometry aware grouping parameter
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, npoint, ncomponents, _lambda, radii, nsamples, mlps, bn=True, use_xyz=True
    ):
        super(PointnetSAModuleGAG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.ncomponents = ncomponents
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.GAGQueryAndGroup(
                    radius, nsample, ncomponents, _lambda=_lambda, use_xyz=use_xyz
                )
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        # Consider first ncomponents features as components ids
        components, features = (
            features[:, : self.ncomponents, :],
            features[:, self.ncomponents :, :],
        )

        # Do not sample when npoint not given
        new_xyz = None
        new_components = None

        if self.npoint is not None:
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            xyz_fps = pointnet2_utils.furthest_point_sample(xyz, self.npoint)

            new_xyz = (
                pointnet2_utils.gather_operation(xyz_flipped, xyz_fps)
                .transpose(1, 2)
                .contiguous()
            )

            new_components = (
                pointnet2_utils.gather_operation(components.contiguous(), xyz_fps)
                .transpose(1, 2)
                .contiguous()
            )

        new_features_list = [new_components.transpose(1, 2)]

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz,
                new_xyz,
                components.transpose(1, 2).contiguous(),
                new_components.contiguous(),
                features.contiguous(),
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleGAGKNN(PointnetSAModuleGAG):
    r"""Pointnet SA GAG with radius grouping

    Parameters
    ----------
    npoint : int
        Number of features
    ncomponents : int
        Number of component ids per point used in grouping
    _lambda : float
        Geometry aware grouping parameter
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, npoint, ncomponents, _lambda, nsamples, mlps, bn=True, use_xyz=True
    ):
        nn.Module.__init__(self)

        assert len(nsamples) == len(mlps)

        self.npoint = npoint
        self.ncomponents = ncomponents
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(nsamples)):
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.GAGKNNQueryAndGroup(
                    nsample, _lambda=_lambda, use_xyz=use_xyz
                )
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


# EVG
class PointnetSAModuleEVG(PointnetSAModuleMSG):
    r"""Pointnet SA EVG with radius grouping

    Parameters
    ----------
    npoint : int
        Number of features
    knn_nsamples : list of int32
        Number of samples in each initial knn query
    vec_radii : list of float32
        list of radii to ball group with
    vec_lengths : list of float32
        list of radii to vector group with
    vec_nsamples : list of int32
        Number of samples in each vector query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
        self,
        npoint,
        knn_nsamples,
        vec_radii,
        vec_lengths,
        vec_nsamples,
        mlps,
        bn=True,
        use_xyz=True,
    ):
        nn.Module.__init__(self)

        assert (
            len(vec_radii)
            == len(vec_lengths)
            == len(knn_nsamples)
            == len(vec_nsamples)
            == len(mlps)
        )

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for knn_nsample, vec_radius, vec_length, vec_nsample, mlp in zip(
            knn_nsamples, vec_radii, vec_lengths, vec_nsamples, mlps
        ):
            self.groupers.append(
                pointnet2_utils.EVGQueryAndGroup(
                    knn_nsample, vec_radius, vec_length, vec_nsample, use_xyz=use_xyz,
                )
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )

            if use_xyz:
                mlp[0] += 3

            self.mlps.append(build_shared_mlp(mlp, bn))


class PointnetSAModuleEVGKNN(PointnetSAModuleMSG):
    r"""Pointnet SA EVG with radius grouping

    Parameters
    ----------
    npoint : int
        Number of features
    knn_nsamples : list of int32
        Number of samples in each initial knn query
    vec_radii : list of float32
        list of radii to ball group with
    vec_lengths : list of float32
        list of radii to vector group with
    vec_nsamples : list of int32
        Number of samples in each vector query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
        self,
        npoint,
        knn_nsamples,
        vec_lengths,
        vec_nsamples,
        mlps,
        bn=True,
        use_xyz=True,
    ):
        nn.Module.__init__(self)

        assert len(vec_lengths) == len(knn_nsamples) == len(vec_nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for knn_nsample, vec_length, vec_nsample, mlp in zip(
            knn_nsamples, vec_lengths, vec_nsamples, mlps
        ):
            self.groupers.append(
                pointnet2_utils.EVGKNNQueryAndGroup(
                    knn_nsample, vec_length, vec_nsample, use_xyz=use_xyz,
                )
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )

            if use_xyz:
                mlp[0] += 3

            self.mlps.append(build_shared_mlp(mlp, bn))


# SSG
class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(list(known_feats.size()[0:2]) + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointnetFPModuleGAG(nn.Module):
    r"""Propigates the features of one set to another with geometry aware grouping

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, _lambda, ncomponents, bn=True):
        super(PointnetFPModuleGAG, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)
        self._lambda = _lambda
        self.ncomponents = ncomponents

    def forward(self, unknown, known, unknown_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknown_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            known_components, known_feats = (
                known_feats[:, : self.ncomponents, :],
                known_feats[:, self.ncomponents :, :],
            )
            unknown_components, unknown_feats = (
                unknown_feats[:, : self.ncomponents, :],
                unknown_feats[:, self.ncomponents :, :],
            )

            dist, idx = pointnet2_utils.gag_three_nn(
                unknown,
                known,
                unknown_components.transpose(1, 2).contiguous(),
                known_components.transpose(1, 2).contiguous(),
                self._lambda,
                self.ncomponents,
            )

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats.contiguous(), idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(list(known_feats.size()[0:2]) + [unknown.size(1)])
            )

        if unknown_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknown_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        new_features = new_features.squeeze(-1)

        new_features = torch.cat([unknown_components, new_features], dim=1)

        return new_features

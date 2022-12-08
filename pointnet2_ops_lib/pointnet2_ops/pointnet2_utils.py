import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


def knn_gpu(new_xyz, xyz, k):
    def square_distance(src, dst):
        return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

    dists = square_distance(new_xyz, xyz)
    neighbors = dists.argsort()[:, :, :k]
    torch.cuda.empty_cache()
    return neighbors.contiguous().int()


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class GAGThreeNN(Function):
    @staticmethod
    def forward(
        ctx, unknown, known, unknown_components, known_components, _lambda, ncomponents
    ):
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features
        unknown_components : torch.Tensor
            (B, n, ncomponents) tensor of known components idx
        known_components : torch.Tensor
            (B, m, ncomponents) tensor of unknown components idx
        _lambda : float
            geometry aware grouping parameter

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) gag l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        unknown = torch.cat((unknown, unknown_components), 2)
        known = torch.cat((known, known_components), 2)

        dist2, idx = _ext.gag_three_nn(unknown, known, _lambda, ncomponents)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


gag_three_nn = GAGThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply

# QUERYING
# 1) MSG
class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class KNNQuery(Function):
    @staticmethod
    def forward(
        ctx, nsample, xyz, new_xyz,
    ):
        """
        Parameters
        ----------
        nsample : int
            maximum number of features in the knn grouping
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the knn query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the knn vector query
        """
        output = knn_gpu(new_xyz, xyz, k=nsample)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


knn_query = KNNQuery.apply

# 2) GAG
class GAGBallQuery(Function):
    @staticmethod
    def forward(
        ctx,
        radius,
        nsample,
        _lambda,
        ncomponents,
        xyz,
        new_xyz,
        components,
        new_components,
    ):
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        _lambda : float
            geometry aware grouping parameter
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query
        components : torch.Tensor
            (B, N, ncomponents) component idx of xyz
        new_components : torch.Tensor
            (B, npoint, ncomponents) component idx of new_xyz
        
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        xyz = torch.cat((xyz, components), 2)
        new_xyz = torch.cat((new_xyz, new_components), 2)

        output = _ext.gag_ball_query(
            new_xyz, xyz, radius, nsample, _lambda, ncomponents, 0
        )

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


gag_ball_query = GAGBallQuery.apply


class GAGKNNQuery(Function):
    @staticmethod
    def forward(
        ctx, knn_nsample, _lambda, xyz, new_xyz, components, new_components,
    ):
        r"""

        Parameters
        ----------
        knn_nsample : int
            number of features
        _lambda : float
            geometry aware grouping parameter
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the queries
        components : torch.Tensor
            (B, N, ncomponents) component idx of xyz
        new_components : torch.Tensor
            (B, npoint, ncomponents) component idx of new_xyz
        
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """

        # Extract first component only
        components = components[..., 0]
        new_components = new_components[..., 0]

        output = GAGKNNQuery.gag_knn_gpu(
            new_xyz, xyz, new_components, components, k=knn_nsample, _lambda=_lambda
        )
        return output

    @staticmethod
    def gag_knn_gpu(new_xyz, xyz, new_components, components, k, _lambda):
        def square_distance(src, dst):
            return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

        dists = square_distance(new_xyz, xyz)
        component_mask = components[:, None] == new_components[:, :, None]

        dists[component_mask] *= _lambda
        dists[~component_mask] *= 1 - _lambda

        neighbors = dists.argsort()[:, :, :k]
        torch.cuda.empty_cache()
        return neighbors.contiguous().int()

    @staticmethod
    def backward(ctx, grad_out):
        return ()


gag_knn_query = GAGKNNQuery.apply

# 3) EVG
class EVGQueryBase(Function):
    @staticmethod
    def forward(
        ctx, xyz, new_xyz, vec_length, knn_nsample, use_root_as_mean=False,
    ):

        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the knn query
        vec_length : float
            length of vector to perform grouping with
        knn_nsample : int
            maximum number of features in the knn initial grouping
        use_root_as_mean : bool
            whether to use root as a blob mean for vector anchor

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the evg query
        """

        # Prior KNN query
        queried = knn_gpu(new_xyz, xyz, k=knn_nsample)

        # Group points with indices from ball queries
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, queried)  # (B, 3, npoint, nsample)
        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1)  # (B, npoint, nsample, 3)

        B, N, S = queried.size()

        dir_vectors_batched = torch.zeros(B, N, 6).to(xyz)

        # Iterate per batch
        for b in range(B):

            # Extract grouped coordinates
            coords = grouped_xyz[b]

            # Compute mean and covariance
            if use_root_as_mean:
                mean = new_xyz[b]
            else:
                mean = coords.mean(axis=1)

            cov = EVGQueryBase._cov(
                (coords - mean.unsqueeze(1).repeat(1, S, 1)).transpose(1, 2)
            )

            # Extract top eigenvector
            dir_vectors = EVGQueryBase._top_eigenvector(cov).squeeze()
            dir_vectors *= vec_length

            # Collect eigenvectors for query
            dir_vectors_batched[b] = torch.cat(
                (mean - dir_vectors, mean + dir_vectors), axis=1
            )

        return dir_vectors_batched

    @staticmethod
    def _top_eigenvector(K, n_power_iterations=20, dim=1):
        """Iterative method to approximate eigenvector with top eigenvalue"""
        v = torch.ones(K.shape[0], K.shape[1], 1).to(K)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n
        return v

    @staticmethod
    def _cov(tensor, rowvar=True, bias=False):
        """
        Estimate a covariance matrix (np.cov)
        https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
        """
        tensor = tensor if rowvar else tensor.transpose(-1, -2)
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
        return factor * tensor @ tensor.transpose(-1, -2).conj()

    @staticmethod
    def backward(ctx, grad_out):
        return ()


class EVGBallQuery(Function):
    @staticmethod
    def forward(
        ctx, knn_nsample, vec_radius, vec_length, vec_nsample, xyz, new_xyz,
    ):
        """
        Parameters
        ----------
        knn_nsample : int
            maximum number of features in the knn initial grouping
        vec_radius : float
            radius around the vector to group from
        vec_length : float
            length of vector to perform grouping with
        vec_nsample : int
            maximum number of features in the vector spheroid
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the knn query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the knn vector query
        """

        # Perform vector estimation
        dir_vectors = EVGQueryBase.forward(
            ctx, xyz, new_xyz, vec_length, knn_nsample, use_root_as_mean=True
        )

        # EVG ball query
        output = _ext.vector_query(dir_vectors, xyz, vec_radius, vec_nsample)
        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


evg_ball_query = EVGBallQuery.apply


class EVGKNNQuery(Function):
    @staticmethod
    def forward(
        ctx, knn_nsample, vec_length, vec_nsample, xyz, new_xyz,
    ):
        """
        Parameters
        ----------
        knn_nsample : int
            maximum number of features in the knn initial grouping
        vec_length : float
            length of vector to perform grouping with
        vec_nsample : int
            maximum number of features in the vector grouping
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the knn query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the knn vector query
        """

        b, n, _ = xyz.size()
        _, m, _ = new_xyz.size()

        # Perform vector estimation
        dir_vectors = EVGQueryBase.forward(
            ctx, xyz, new_xyz, vec_length, knn_nsample, use_root_as_mean=True
        )

        # Vector difference
        sub = dir_vectors[..., :3] - dir_vectors[..., 3:]

        # Translating points
        p1_xyz = xyz[:, None] - dir_vectors[:, :, None][..., :3]

        # Cross product
        cross = sub.repeat(1, n, 1, 1).reshape(b, m, n, -1)
        cross = torch.cross(cross, p1_xyz)

        # Computing distances
        dists = torch.linalg.norm(cross, dim=-1) / torch.linalg.norm(sub)
        del cross

        # Calculate p1 mask
        p1_mask = (p1_xyz * sub[:, :, None]).sum(-1) <= 0
        dists[p1_mask] = torch.linalg.norm(p1_xyz[p1_mask], dim=-1)
        del p1_xyz, p1_mask

        # Calculate p2 mask
        p2_xyz = xyz[:, None] - dir_vectors[:, :, None][..., 3:]
        p2_mask = (p2_xyz * sub[:, :, None]).sum(-1) >= 0
        dists[p2_mask] = torch.linalg.norm(p2_xyz[p2_mask], dim=-1)
        del p2_xyz, p2_mask

        # Calculate knn
        neighbors = dists.argsort()[:, :, :vec_nsample]
        torch.cuda.empty_cache()

        return neighbors.contiguous().int()

    @staticmethod
    def backward(ctx, grad_out):
        return ()


evg_knn_query = EVGKNNQuery.apply


# GROUPING
# 1) MSG & SSG
class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class KNNQueryAndGroup(nn.Module):
    r"""
    Groups with a knn query

    Parameters
    ---------
    nsample : int32
        Maximum number of features to gather with knn
    """

    def __init__(
        self, nsample, use_xyz=True,
    ):
        super(KNNQueryAndGroup, self).__init__()

        self.nsample = nsample
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = knn_query(self.nsample, xyz, new_xyz,)

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


# 2) GAG
class GAGQueryAndGroup(nn.Module):
    r"""
    Groups with a gag ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    lambda : float
        Geometry aware grouping parameter
    """

    def __init__(self, radius, nsample, ncomponents, _lambda=0.5, use_xyz=True):
        super(GAGQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.ncomponents, self._lambda, self.use_xyz = (
            radius,
            nsample,
            ncomponents,
            _lambda,
            use_xyz,
        )

    def forward(self, xyz, new_xyz, components, new_components, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        components : torch.Tensor
            xyz components idx (B, N, 1)
        new_components : torch.Tensor
            new_xyz components idx (B, npoint, 1)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = gag_ball_query(
            self.radius,
            self.nsample,
            self._lambda,
            self.ncomponents,
            xyz,
            new_xyz,
            components,
            new_components,
        )
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GAGKNNQueryAndGroup(nn.Module):
    r"""
    Groups with a gag knn query

    Parameters
    ---------
    nsample : int32
        Maximum number of features to gather with knn
    """

    def __init__(
        self, nsample, _lambda=0.5, use_xyz=True,
    ):
        super(GAGKNNQueryAndGroup, self).__init__()

        self.nsample = nsample
        self._lambda = _lambda
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, components, new_components, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        components : torch.Tensor
            xyz components idx (B, N, 1)
        new_components : torch.Tensor
            new_xyz components idx (B, npoint, 1)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = gag_knn_query(
            self.nsample, self._lambda, xyz, new_xyz, components, new_components,
        )
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


# 3) EVG
class EVGQueryAndGroup(nn.Module):
    r"""
    Groups with a vector query with initial knn query

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self, knn_nsample, vec_radius, vec_length, vec_nsample, use_xyz=True,
    ):
        super(EVGQueryAndGroup, self).__init__()

        self.knn_nsample = knn_nsample
        self.vec_radius = vec_radius
        self.vec_length = vec_length
        self.vec_nsample = vec_nsample
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = evg_ball_query(
            self.knn_nsample,
            self.vec_radius,
            self.vec_length,
            self.vec_nsample,
            xyz,
            new_xyz,
        )

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class EVGKNNQueryAndGroup(nn.Module):
    r"""
    Groups with a vector knn query with initial knn query

    Parameters
    ---------
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self, knn_nsample, vec_length, vec_nsample, use_xyz=True,
    ):
        super(EVGKNNQueryAndGroup, self).__init__()

        self.knn_nsample = knn_nsample
        self.vec_length = vec_length
        self.vec_nsample = vec_nsample
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = evg_knn_query(
            self.knn_nsample, self.vec_length, self.vec_nsample, xyz, new_xyz,
        )

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features

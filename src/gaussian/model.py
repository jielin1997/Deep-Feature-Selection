import argparse
from collections import OrderedDict
import copy

from albumentations.pytorch.transforms import ToTensorV2

import numpy as np
import scipy.io as io
import pytorch_lightning as pl
import os
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM

from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import make_grid

from ..common.augmentation import (
    dataset_aug,
    Compose,
    unnormalize,
    MaskToSize,
)
from albumentations import Resize, Normalize

from ..common.dataset import (
    FoldSplit,
    StratifiedFoldSplit,
    AnomalyDetectionDataset,
)
from ..common.evaluation import (
    log_roc_figure,
    latent_tsne_figure,
    latent_pca_figure,
    MinMaxSaver,
    sample_covariance,
)
from ..common.utils import TensorList, batched_index_select, flatten
from .transparent import build_transparent, MODEL_NAMES
from tqdm import tqdm
from typing import NoReturn, Union, Iterable, Optional, Type

class Mahalanobis(pl.LightningModule):  # 继承pl.LightningModule类，要求重写相应的方法，类似于抽象类

    def __init__(
        self,
        hparams: argparse.Namespace,
        dataset_cls: Type[AnomalyDetectionDataset],
    ) -> NoReturn:
        # 马氏距离，
        super(Mahalanobis, self).__init__()  # 调用父类的初始化方法

        self.hparams = hparams

        self.normalization_epochs = 100 if self.hparams.augment else 1
        # 这里的moel是transparent中定义的TransparentEfficientNet类
        self.model, self.input_size = build_transparent(
            hparams.arch,
            pretrained=True,
            extract_blocks=hparams.extract_blocks,
            freeze=True,  # freeze参数用于在训练时固定某一层的参数，使其不再更新
        )
        # hasattr() 函数用于判断对象是否包含对应的属性
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()

        # We do not want to augment in val/test
        final_transform = [Normalize(), ToTensorV2()]
        # 由于python存储数据的特殊性，使用deepcoay复制，这个复制就是复制一遍并将其作为独立的个体存在
        final_eval_transform = copy.deepcopy(final_transform)

        if hparams.augment:
            uncached = Compose(
                [dataset_aug(hparams, dataset_cls), *final_transform]
            )
        else:
            uncached = Compose(final_transform)

        # 调用数据集类的初始化方法
        dataset = dataset_cls(
            hparams=hparams,
            train=True,
            transform=Resize(self.input_size, self.input_size),  # 380*380，这是在efficientNet中定义的最佳分辨率
            uncached_transform=uncached,
        )
        # 作者重写了pytorch自带的k-fold交叉验证，并在此处调用
        self.datasplit = FoldSplit(
            dataset,
            self.hparams.fold,
            uncached_eval_transform=Compose(final_eval_transform),
        )

        self.testset = dataset_cls(
            hparams=hparams,
            train=False,
            load_masks=True,
            transform=Compose(
                [
                    Resize(self.input_size, self.input_size),
                    # Reduce mask to size for max perf.
                    MaskToSize(),
                ]
            ),
            uncached_transform=Compose(final_eval_transform),
        )

        self.test_saver = MinMaxSaver(unnormalize_fn=unnormalize)

        if self.hparams.variance_threshold:
            assert (
                self.hparams.pca != self.hparams.npca
            ), "need to set either only pca or npca flag"
        else:
            assert (
                not self.hparams.pca and not self.hparams.npca
            ), "cannot specify pca/npca and not specify variance_threshold"

        self.statistics_computed = False

        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.hparams.gpus
            else "cpu"
        )

    def forward(self, x: torch.Tensor) -> TensorList:
        """Output features of self.model."""
        with torch.no_grad():
            z = self.model(x)
        return z

    @staticmethod
    def tensorlist_or_tensor(items: list) -> Union[torch.Tensor, TensorList]:
        if len(items) == 1:
            return items[0].unsqueeze(0)
        return TensorList(items)

    def compute_train_sed(self, features: TensorList) -> NoReturn:
        """Compute sed normalization mean & stddev.

        This is the per feature independent gaussian assumption (only mean
        and stddev).
        """

        mean = features.mean(dim=1)  # mean is level x features.
        stddev = features.std(dim=1)
        self.sed_mean = mean  # equal to the mean of mvg, but not equal to pca_mean (as sed/maha may be performed on reduced dimensionalities)
        self.sed_stddev = stddev

    @staticmethod
    def compute_mahalanobis_threshold(
        k: int, p: float = 0.9973
    ) -> torch.Tensor:
        """Compute a threshold on the mahalanobis distance.

        So that the probability of mahalanobis with k dimensions being less
        than the returned threshold is p.
        """
        # Mahalanobis² is Chi² distributed with k degrees of freedom.
        # So t is square root of the inverse cdf at p.
        return torch.Tensor([chi2.ppf(p, k)]).sqrt()  # 在给定RV的情况下，在q点的百分点函数(cdf的倒数)

    def compute_train_gaussian(self, features: TensorList) -> NoReturn:
        """
        features: TensorList
        """

        def fit_inv_covariance(samples):
            return torch.Tensor(LedoitWolf().fit(samples.cpu()).precision_).to(
                samples.device
            )

        print("Performing Covariance Estimation")
        inv_covariance = TensorList(
            [fit_inv_covariance(level) for level in features]
        )
        mean = features.mean(dim=1)  # mean features.

        self.mvg_mean = mean
        self.mvg_inv_covariance = inv_covariance
        # Also cache the number of features each level outputs for later.
        feature_count = torch.cat(
            [torch.Tensor([level.shape[-1]]) for level in mean]
        )
        self.feature_count = feature_count
        # 保存训练集的马氏距离
        # maha = TensorList(
        #     [
        #         self.mahalanobis_distance(
        #             level, val_mean, val_icov
        #         )
        #         for level, val_mean, val_icov in zip(
        #         features, self.mvg_mean, self.mvg_inv_covariance
        #     )
        #     ]
        # )
        # space = self.hparams.select_space
        # save_path = '/home/lj/Work/gaussian/maha/'+self.hparams.category+'/train/'
        # if not os.path.exists(save_path):  # 判断文件夹是否存在
        #     os.makedirs(save_path)  # 不存在则新建文件夹
        # level_num = 1
        # for level in maha:
        #     level = level.cpu()
        #     io.savemat(save_path + 'train_maha_{space}_level{level}.mat'.format(space=space, level=level_num),
        #                {'train_maha': np.array(level)})
        #     # io.savemat(save_path + 'valid_maha_{space}_level{level}.mat'.format(space=space, level=level_num),
        #     #            {'valid_maha': np.array(level)})
        #     level_num = level_num + 1

    def compute_train_ocsvm(self, features: TensorList) -> NoReturn:
        def fit_ocsvm(samples):
            ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
            ocsvm.fit(samples)
            return ocsvm

        print("Fitting OCSVM")
        self.ocsvms = [fit_ocsvm(level) for level in features.cpu()]

    def ocsvm_predict(self, features: TensorList) -> TensorList:
        return TensorList(
            [
                torch.Tensor(
                    -ocsvm.decision_function(level.mean(dim=(-2, -1)).cpu())
                )
                for level, ocsvm in zip(features, self.ocsvms)
            ]
        )

    def compute_pca(
        self, features: TensorList, variance_threshold: float = 0.95
    ) -> NoReturn:
        """Compute pca normalization of teacher features retaining variance.

        Contrary to normal pca, this throws away the features with large
        variance and only keeps the ones with small amounts of variance.
        It is expected that those features will activate on the anomalies.
        """
        '''关于sklearn的pca方法：https://blog.csdn.net/qq_20135597/article/details/95247381'''
        mean = features.mean(dim=1)  # mean is level x features.

        def fit_level(features: torch.Tensor) -> torch.Tensor:
            pca = PCA(n_components=None).fit(features)
            # fit方法：表示用feature数据来训练PCA模型，返回的是PCA这个方法本身，因为PCA是一种无监督学习算法，因此需要训练
            # n-components参数表示PCA算法中需要保留的主成分个数，Node表示特征个数不改变

            # Select features above variance_threshold.
            variances = pca.explained_variance_ratio_.cumsum()
            last_component = (variances > variance_threshold).argmax()
            # last_component is the index of the last value needed to reach at
            # least the required explained variance.
            # As the true variance lies somewhere in between [last_component - 1,
            # last_component], we include the whole interval for both pca as
            # well as NPCA based dimensionality reduction
            if self.hparams.pca:
                return torch.Tensor(pca.components_[: last_component + 1])
            elif self.hparams.npca:
                variance_pca = pca.explained_variance_  # feature矩阵的特征值
                [Ns, Nf] = features.shape
                rw = max(np.nonzero(variance_pca)[0])  # rw:feature矩阵实际的的秩 - 1，即最小非零值的下标
                mr_of_feature = Ns if Ns <= Nf else Nf  # mr_of_feature:feature有可能存在的最大秩
                lmd_med = np.median(variance_pca[:rw + 1])  # 这里rw+1是因为python是左闭右开的，而rw就是下标，所以要+1
                # print(variance_pca[rw])
                miu = 1
                above_zero = np.maximum(variance_pca - (lmd_med + miu * (lmd_med - variance_pca[rw])), 0)
                m1 = max(np.nonzero(above_zero)[0])  # m1直接就是下标
                # 求本征比，虽然python的range是左闭右开的集合，这里用rw是因为本征比的个数会比特征值少1，因为最后一个值没法算本征比
                rk = np.zeros(rw)
                for i in range(rw):
                    rk[i] = variance_pca[i + 1] / variance_pca[i]
                window_size = 10
                # 一直到if方法结束的注释部分用于保存特征值等信息
                # save_path = '/home/lj/Work/gaussian/tezhengzhi/'
                # io.savemat(save_path+'chushibenzhengbi.mat',{'chushibenzhengbi':rk})
                rk = np.convolve(rk, np.ones(window_size) / window_size, mode="same")  # 滑动平均
                # io.savemat(save_path+'pinghuabenzhengbi.mat',{'pinghuabenzhengbi':rk})
                m2 = int(np.where(rk == max(rk[m1 + 1:]))[0][0])  # m2直接就是下标
                if self.hparams.select_space == "2_3":
                    return torch.Tensor(pca.components_[m1:])  # 原文的只改横向
                elif self.hparams.select_space == "3":
                    return torch.Tensor(pca.components_[m2:])
                elif self.hparams.select_space == "2":
                    return torch.Tensor(pca.components_[m1:m2])
                elif self.hparams.select_space == "1":
                    return torch.Tensor(pca.components_[:m1])
                elif self.hparams.select_space == "all":
                    return torch.Tensor(pca.components_[:])
                elif self.hparams.select_space == None:
                    return torch.Tensor(pca.components_[last_component - 1:])
                # return torch.Tensor(pca.components_[last_component - 1:])  # 原文的NPCA1%
            else:
                raise ValueError(
                    "either hparams.pca or hparams.npca need to be specified"
                )

        components = self.tensorlist_or_tensor(
            [fit_level(level) for level in features.cpu()]
        )

        self.pca_mean = mean
        self.pca_components = components

    # TensorList if len(extract_blocks) > 1 else torch.Tensor
    def l2_distance(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        return (
            (features - self.mvg_mean.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            .mean(dim=(-2, -1))
            .pow(2)
            .mean(dim=2)
            .sqrt()
        )

    def sed_distance(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        """Return normalized features (using the computed normalization)."""
        # Unsqueeze batch, height & width.
        return (
            (
                (
                    features
                    - self.sed_mean.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )
                / self.sed_stddev.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )
            .mean(dim=(-2, -1))
            .pow(2)
            .mean(dim=2)
            .sqrt()
        )

    def pca_reduction(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        """Return pca-reduced features (using the computed PCA)."""
        # Features is level x training_samples x features x height x width.
        # Unsqueeze batch, height & width.
        demeaned = features - self.pca_mean.unsqueeze(1).unsqueeze(
            -1
        ).unsqueeze(-1)

        def batched_mul_components(
            level: torch.Tensor, components: torch.Tensor
        ) -> torch.Tensor:
            # Cannot use einsum because of unsupported broadcasting.
            # So do a permute to (samples x height x width x features).
            reduced = torch.matmul(  # Batched vector matrix multiply.
                level.permute(0, 2, 3, 1).unsqueeze(-2),
                components.t().unsqueeze(0).unsqueeze(0).unsqueeze(0),
            ).squeeze(
                -2
            )  # Squeeze so this is vector matrix multiply.
            return reduced.permute(0, 3, 1, 2)  # Back to BCHW.

        return self.tensorlist_or_tensor(
            # This is (X - mean).dot(components.t()).
            [
                batched_mul_components(level, components)
                for level, components in zip(
                    demeaned, self.pca_components.to(self._device)
                )
            ]
        )

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.

        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert inv_covariance.dim() == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return OrderedDict({"loss/val": torch.tensor(0)})

    def validation_end(self, outputs: list) -> dict:
        return {"progress_bar": {}, "log": {}}

    def compute_train_statistics(self) -> NoReturn:
        # Use same dataloader as training.
        trainset = self.datasplit.train()
        print(
            "Computing training statistics with {} images".format(
                len(trainset)
            )
        )
        # 保存用于训练的图片index
        # save_path = '/home/lj/Work/gaussian/maha/'+self.hparams.category+'/train/'
        # if not os.path.exists(save_path):  # 判断文件夹是否存在
        #     os.makedirs(save_path)  # 不存在则新建文件夹
        # io.savemat(save_path + 'fold_0_train_index.mat', {'train_index': np.array(trainset.indices)})


        dataloader = DataLoader(
            trainset, batch_size=self.hparams.batch_size, num_workers=2
        )

        with torch.no_grad():
            outputs = []
            for epoch in tqdm(range(self.normalization_epochs)):
                for batch in dataloader:
                    z = self.model(batch["image"].to(self._device))
                    # Mean across locations (at this point to save GPU RAM).
                    z = z.mean(dim=(3, 4), keepdim=True)
                    outputs.append(z)
            # Features is level x training_samples x features.
            features = TensorList.cat(outputs, dim=1)

        # TODO: Beautify this so pca reduction does not require singleton dimensions for hxw
        if self.hparams.npca or self.hparams.pca:
            self.compute_pca(
                features.mean(dim=(3, 4)),
                variance_threshold=self.hparams.variance_threshold,
            )

            outputs_reduced = []
            for batch in outputs:
                reduced = self.pca_reduction(batch)
                outputs_reduced.append(reduced.mean(dim=(3, 4)))
            features = TensorList.cat(outputs_reduced, dim=1)
        else:
            features = features.mean(dim=(3, 4))

        self.compute_train_gaussian(features)
        if self.hparams.ocsvm:
            self.compute_train_ocsvm(features)
        if self.hparams.sed:
            self.compute_train_sed(features)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        # 新数据集地，去掉mask
        # x, target, mask = batch["image"], batch["target"], batch["mask"]
        x, target = batch["image"], batch["target"]
        z = self(x)
        # 按照batch保存测试集降维前features矩阵和标签
        # features_test_path = "/home/lj/Work/gaussian/feature/"+self.hparams.category+"/test/before_reduce/"
        # if not os.path.exists(features_test_path):  # 判断文件夹是否存在
        #     os.makedirs(features_test_path)  # 不存在则新建文件夹
        # level_num = 1
        # for level in z:
        #     level = level.cpu().mean(dim=(-2, -1))
        #     target_tmp = target.cpu()
        #     io.savemat(
        #         features_test_path + 'original_features_test_batch{batch}_level{level}.mat'.format(batch=batch_idx,
        #                                                                                            level=level_num),
        #         {'level{}'.format(level_num): np.array(level)})
        #     io.savemat(
        #         features_test_path + 'target_test_batch{batch}.mat'.format(batch=batch_idx),
        #         {'target_batch{}'.format(batch_idx): np.array(target_tmp)})
        #     level_num = level_num + 1

        # 保存样本属于哪个具体的故障类型
        # save_path = "/home/lj/Work/gaussian/feature/"+self.hparams.category+"/test/"
        # if not os.path.exists(save_path):  # 判断文件夹是否存在
        #     os.makedirs(save_path)  # 不存在则新建文件夹
        # sample_belong = []
        # for sample in self.testset.samples:
        #     sample_belong.append(sample[1].strip())
        # io.savemat(save_path + 'sample_belong.mat', {'sample_belong': sample_belong})

        if not self.statistics_computed:
            self.compute_train_statistics()
            self.statistics_computed = True

        # reduce features also here
        if self.hparams.npca or self.hparams.pca:
            z = self.pca_reduction(z)

        # 按照batch保存测试集降维后的features矩阵
        # space = self.hparams.select_space
        # features_test_path = "/home/lj/Work/gaussian/feature/"+self.hparams.category+"/test/after_reduce/ori_eig/{space}/".format(space=space)
        # # features_test_path = "/home/lj/Work/gaussian/feature/'+self.hparams.category+'/test/after_reduce/cor_eig/{space}/".format(space=space)
        # if not os.path.exists(features_test_path):  # 判断文件夹是否存在
        #     os.makedirs(features_test_path)  # 不存在则新建文件夹
        # level_num = 1
        # for level in z:
        #     level = level.cpu().mean(dim=(-2, -1))
        #     io.savemat(
        #         features_test_path + 'original_features_reduce_test_batch{batch}_level{level}.mat'.format(
        #             batch=batch_idx,
        #             level=level_num),
        #         {'level{}'.format(level_num): np.array(level)})
        #         # features_test_path + 'weight_features_reduce_test_batch{batch}_level{level}.mat'.format(
        #         #     batch=batch_idx,
        #         #     level=level_num),
        #         # {'level{}'.format(level_num): np.array(level)})
        #     level_num = level_num + 1



        maha = TensorList(
            [
                self.mahalanobis_distance(
                    level.mean(dim=(-2, -1)), val_mean, val_icov
                )
                for level, val_mean, val_icov in zip(
                    z, self.mvg_mean, self.mvg_inv_covariance
                )
            ]
        )

        if self.hparams.ocsvm:
            ocsvm = self.ocsvm_predict(z)

        if self.hparams.sed:
            sed = self.sed_distance(z)

        if self.hparams.l2:
            l2 = self.l2_distance(z)

        self.test_saver.update(x, target, maha.mean(dim=0))
        # 新数据集修改地，不需要mask
        # result = OrderedDict({"target": target, "x": x.cpu(), "mask": mask})
        result = OrderedDict({"target": target, "x": x.cpu()})
        for name in ("maha", "sed", "l2", "ocsvm"):
            if name in locals():
                result[name] = locals()[name].cpu()

        return result

    def evaluate_latent(
        self,
        outputs: Iterable[dict],
        target: torch.Tensor,
        key: str,
        images: Optional[torch.Tensor] = None,
    ) -> dict:
        """Plot & evaluate the latent space from test output.

        Return the new entries to the log dict as a dict.
        """
        # Latent space is given as [batch x dimension] tensors for each level.
        z = TensorList.cat([output[key] for output in outputs], dim=1)
        # Iterate all levels of the latent space.
        result = {}
        for i, (z_i, feature_c) in enumerate(zip(z, self.feature_count)):
            # Map index to correct attach_block level.
            i = self.hparams.extract_blocks[i]
            pred = z_i

            auc = log_roc_figure(
                self.logger,
                target,
                pred,
                self.current_epoch,
                kind="latent_{}/level_{}".format(key, i),
            )
            result["latent_{}/level_{}/auc/test".format(key, i)] = auc

            if key == "maha":
                # Mahalanobis TPR/FPR at n sigma.
                sigmas = [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ]  # values > 8 are not computable.
                # 一直到函数结束之前的注释部分用于保存单层置信度阈值、测试集马氏距离等信息
                # space = self.hparams.select_space
                # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test/'
                # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test_swap/'
                # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test_all_swap/'
                # if not os.path.exists(save_path):  # 判断文件夹是否存在
                #     os.makedirs(save_path)  # 不存在则新建文件夹
                # sample_belong = []
                # for sample in self.testset.samples:
                #     sample_belong.append(sample[1].strip())
                # io.savemat(save_path + 'sample_belong.mat', {'sample_belong': sample_belong})
                # # print(np.array(pred))
                # io.savemat(save_path + 'test_maha_{space}_level{level}.mat'.format(space=space, level=i + 1), {'test_maha': np.array(pred)})
                # io.savemat(save_path + 'target.mat', {'target': np.array(target)})
                # thr = []
                for sigma in sigmas:
                    # Probability of a Gaussian at n sigma.
                    p = torch.erf(sigma / torch.DoubleTensor([2]).sqrt())
                    # Threshold on mahalanobis distance at p (n sigma).
                    threshold = self.compute_mahalanobis_threshold(
                        feature_c, p=p.item()
                    )
                    # thr.append(threshold)
                    anomaly_pred = pred > threshold
                    tpr = (
                            anomaly_pred[target != 0].sum()
                            / (target != 0).sum().float()
                    )
                    result[
                        "latent_{}/level_{}/sigma_{}/tpr/test".format(
                            key, i, sigma
                        )
                    ] = tpr
                    fpr = (
                            anomaly_pred[target == 0].sum()
                            / (target == 0).sum().float()
                    )
                    result[
                        "latent_{}/level_{}/sigma_{}/fpr/test".format(
                            key, i, sigma
                        )
                    ] = fpr
                # io.savemat(save_path + 'thr_{space}_level{level}.mat'.format(space=space, level=i + 1), {'thr': np.array(thr)})

        return result

    def test_end(self, outputs: Iterable[dict]) -> dict:
        self.logger.experiment.add_image(
            "min_max_good_pred_images/test",
            self.test_saver.good_grid(),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            "min_max_anomaly_pred_images/test",
            self.test_saver.anomaly_grid(),
            self.current_epoch,
        )

        # compute & draw roc / connected component analysis
        target = torch.cat([output["target"].cpu() for output in outputs], 0)
        x = torch.cat([output["x"] for output in outputs], 0)

        tqdm_dict = {}
        # Classification evaluation.

        for distance in ("maha", "sed", "l2", "ocsvm"):
            # check if distance is in returned dict
            if distance in outputs[0]:
                pred = torch.cat(
                    [
                        output.get(distance).mean(dim=0).cpu()
                        for output in outputs
                    ],
                    0,
                )
                # 保存all stage时的马氏距离
                # space = self.hparams.select_space
                # # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test/'
                # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test_swap/'
                # # save_path = '/home/lj/Work/gaussian/maha/' + self.hparams.category + '/test_all_swap/'
                # if not os.path.exists(save_path):  # 判断文件夹是否存在
                #     os.makedirs(save_path)  # 不存在则新建文件夹
                # io.savemat(save_path + 'test_maha_{space}_all_level.mat'.format(space=space),
                #            {'test_maha': np.array(pred)})

                tqdm_dict[
                    "{}/full_auc/test".format(distance)
                ] = log_roc_figure(
                    self.logger,
                    target,
                    pred,
                    self.current_epoch,
                    kind="full/{}".format(distance),
                )
                print(
                    "{} test auc: {}".format(
                        distance,
                        tqdm_dict.get("{}/full_auc/test".format(distance)),
                    )
                )

                # Latent space evaluation / visualisation.
                tqdm_dict.update(
                    self.evaluate_latent(outputs, target, distance, images=x)
                )

        if self.hparams.npca:
            # Log how many features where removed.
            for i, level in enumerate(self.pca_components):
                # Map index to correct attach_block level.
                i = self.hparams.extract_blocks[i]
                percentage = level.shape[0] / level.shape[1]
                print(
                    "npca features level {}: {} ({})".format(
                        i, level.shape[0], percentage
                    )
                )
                tqdm_dict["npca_feat/level_{}/test".format(i)] = level.shape[0]
                tqdm_dict["npca_perc/level_{}/test".format(i)] = percentage
        tqdm_dict = flatten(tqdm_dict)
        return {"progress_bar": tqdm_dict, "log": tqdm_dict}

    # required by pytorch lightning
    def configure_optimizers(self) -> list:
        return []

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        # already shuffled in datasplit
        trainset = self.datasplit.train()
        sampler = RandomSampler(trainset)
        print("Training with {} images".format(len(trainset)))
        return DataLoader(
            trainset,
            batch_size=self.hparams.batch_size,
            num_workers=2,
            sampler=sampler,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        valset = self.datasplit.val()
        print("Validating with {} images".format(len(valset)))
        return DataLoader(
            valset, batch_size=self.hparams.batch_size, num_workers=2
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        print("Testing with {} images".format(len(self.testset)))
        return DataLoader(self.testset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Specify the hyperparams for this LightningModule."""
        # MODEL specific
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",  # 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
            default="resnet18",
            choices=MODEL_NAMES,  # 参数可允许的值的一个容器
            help="model architecture: "
            + " | ".join(MODEL_NAMES)
            + " (default: resnet18)",
        )
        parser.add_argument(
            "--extract_blocks",
            type=int,
            nargs="+",
            default=[5],
            help="Blocks to extract the features from "
            "Compared to the paper, we have an index"
            " offset of 1 (we star 0 based here but"
            " with base 1 in the paper)",
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument(
            "--variance_threshold",
            type=float,
            default=None,
            help="variance threshold to apply",
        )
        parser.add_argument("--pca", action="store_true", help="enable pca")
        parser.add_argument("--npca", action="store_true", help="Enable npca")
        parser.add_argument(
            "--select_space",
            type=str,
            default=None,
            help="select space",
        )
        parser.add_argument(
            "--augment", action="store_true", help="Enable data augmentation"
        )
        parser.add_argument(
            "--l2", action="store_true", help="Evaluate l2 distance"
        )
        parser.add_argument("--sed", action="store_true", help="SED distance")
        parser.add_argument("--ocsvm", action="store_true", help="train ocsvm")
        return parser

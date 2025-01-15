"""Model class for SIMVI for single cell expression data."""

import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import Delaunay
import pytorch_lightning as pl
import torch.optim as optim
from scvi import settings
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from scipy.sparse import coo_matrix, csr_matrix

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp
from scvi.train import TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split


from simvi.module.simvigraphonly import SimVIGOModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class SimVIGraphOnly(BaseModelClass):
    """
    Model class for SIMVI graph only.

    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_intrinsic: int = 20,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0,
        use_observed_lib_size: bool = True,
        lam_mi: float = 1,
        reg_to_use: str = 'mmd',
        dis_to_use: str = 'zinb',
        permutation_rate: float = 0.25,
        var_eps: float = 1e-4,
        kl_weight: float = 1,
        kl_gatweight: float = 1,
    ) -> None:
        super(SimVIGraphOnly, self).__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = SimVIGOModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_output=n_intrinsic,
            n_spatial=n_spatial,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            lam_mi = lam_mi,
            reg_to_use = reg_to_use,
            dis_to_use = dis_to_use,
            permutation_rate = permutation_rate,
            var_eps = var_eps,
            kl_weight = kl_weight,
            kl_gatweight = kl_gatweight,
        )
        self._model_summary_string = "SimVIGraphOnly"
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Set up AnnData instance for SIMVI graph only model.

        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def extract_edge_index(
        adata: AnnData,
        batch_key: Optional[str] = None,
        spatial_key: Optional[str] = 'spatial',
        method: str = 'knn',
        n_neighbors: int = 30,
        ):
        """
        Define edge_index for SIMVI graph only model training.

        """
        if batch_key is not None:
            j = 0
            for i in adata.obs[batch_key].unique():
                adata_tmp = adata[adata.obs[batch_key]==i].copy()
                if method == 'knn':
                    A = kneighbors_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                    edge_index_tmp, edge_weight = from_scipy_sparse_matrix(A)
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)

                else:
                    tri = Delaunay(adata_tmp.obsm[spatial_key])
                    triangles = tri.simplices
                    edges = set()
                    for triangle in triangles:
                        for i in range(3):
                            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                            edges.add(edge)
                    edge_index_tmp = torch.tensor(list(edges)).t().contiguous()
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)
        else:
            if method == 'knn':
                A = kneighbors_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
                edge_index, edge_weight = from_scipy_sparse_matrix(A)
            else:
                tri = Delaunay(adata.obsm[spatial_key])
                triangles = tri.simplices
                edges = set()
                for triangle in triangles:
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        edges.add(edge)
                edge_index = torch.tensor(list(edges)).t().contiguous().type(torch.LongTensor)

        return edge_index


    @torch.no_grad()
    def get_latent_representation(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.

        """
        available_representation_kinds = ["intrinsic", "interaction","output","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
        outputs = self.module.inference(data,edge_index,eval_mode=True)
        latent = []
        if representation_kind == "intrinsic":
            latent_m = outputs["q_m"]
            latent_sample = outputs["z"]
        elif representation_kind == "interaction":
            latent_m = outputs["qgat_m"]
            latent_sample = outputs["z_gat"]
        elif representation_kind == "output":
            latent_m = self.module.gat_mean.lin_r(outputs["q_m"][:,-self.module.n_spatial:])
            latent_sample = latent_m
        elif representation_kind == "all":
            latent_m = outputs["qall_m"]
            latent_sample = outputs["z_all"]

        if give_mean:
            latent_sample = latent_m

        latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()
    
    
    @torch.no_grad()
    def get_decoded_expression(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        Return the decoded expression for each cell.

        """
        available_representation_kinds = ["intrinsic", "interaction","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
        outputs = self.module.inference(data,edge_index,eval_mode=True)
        
        decoded = []
        if representation_kind == "intrinsic":
            decoded_m = self.module._generic_generative(outputs["q_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z"],outputs["library"],outputs["batch_index"])["px_scale"]
            
        elif representation_kind == "interaction":
            decoded_m = self.module._generic_generative(outputs["qgat_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z_gat"],outputs["library"],outputs["batch_index"])["px_scale"]

        elif representation_kind == "all":
            decoded_m = self.module._generic_generative(outputs["qall_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z_all"],outputs["library"],outputs["batch_index"])["px_scale"]
        
        if give_mean:
            decoded_sample = decoded_m
            
        decoded += [decoded_sample.detach().cpu()]
        return torch.cat(decoded).numpy()


    def train(
        self,
        edge_index: torch.Tensor,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        batch_size: Optional[int] = None,
        anneal_epoches: int = 50,
        validation_size: Optional[float] = None,
        lr = 1e-3,
        weight_decay = 1e-4,
    ) -> None:
        """
        Train the SIMVI graph only model.
        """
        if max_epochs is None:
            n_cells = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
            

        if validation_size is None:
            validation_size = 1 - train_size

        n_train, n_val = validate_data_split(self.adata_manager.adata.n_obs, train_size, validation_size)
        random_state = np.random.RandomState(seed=settings.seed)
        permutation = random_state.permutation(self.adata_manager.adata.n_obs)
        train_mask = permutation[:n_train]
        val_mask = permutation[n_train : (n_val + n_train)]
        test_mask = permutation[(n_val + n_train) :]
        if use_gpu & torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = 'cpu'
        self.module = self.module.to(device)
        edge_index = edge_index.to(device)

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(device)

        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        pbar = tqdm(range(1, max_epochs + 1))
            
        if batch_size is not None:
            batch_indices = [train_mask[i:i + batch_size] for i in range(0, train_mask.shape[0], batch_size)]
            train_loader = {}
            for i, batch_index in enumerate(batch_indices):
                data_masked = {}
                for key, value in data.items():
                    if value is None:
                        data_masked[key] = None
                    else:
                        data_masked[key] = value[batch_index]

                train_loader[i] = data_masked
        else:
            data_masked = {}
            for key, value in data.items():
                if value is None:
                    data_masked[key] = None
                else:
                    data_masked[key] = value[train_mask]
            train_loader = data_masked
            
        val_loader = {}
        for key, value in data.items():
            if value is None:
                val_loader[key] = None
            else:
                val_loader[key] = value[val_mask]

        for epoch in pbar:
            weight = min(1.0, epoch / anneal_epoches)
            train_loss.append(_train(self.module, data, edge_index, train_mask, train_loader, optimizer, batch_size, weight))
            val_loss.append(_eval(self.module, data, edge_index, val_mask,val_loader, weight))
            pbar.set_description('Epoch '+str(epoch)+'/'+str(max_epochs))
            pbar.set_postfix(train_loss=train_loss[epoch-1], val_loss=val_loss[epoch-1].numpy())

        return train_loss, val_loss

def _train(model, data, edge_index, mask, train_loader, optimizer, batch_size, weight):
    train_loss = []
    model.train()
    #print(latent_dict)
    if batch_size is None:
        optimizer.zero_grad()
        latent_dict = model.inference(data,edge_index)
        latent_dict_masked = {}
        for key, value in latent_dict.items():
            if value is None:
                latent_dict_masked[key] = None
            else:
                latent_dict_masked[key] = value[mask]

        decoder_dict = model.generative(latent_dict_masked)
        lossrecorder = model.loss(train_loader, latent_dict_masked, decoder_dict, weight)
        loss = lossrecorder.loss
        loss.backward()
        #clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
        train_loss.append(loss.detach())
    else:
        batch_indices = [mask[i:i + batch_size] for i in range(0, mask.shape[0], batch_size)]
        
        for i, batch_index in enumerate(batch_indices):
            optimizer.zero_grad()
            latent_dict = model.inference(data,edge_index)
            latent_dict_masked = {}
            data_masked = {}
            for key, value in latent_dict.items():
                if value is None:
                    latent_dict_masked[key] = None
                else:
                    latent_dict_masked[key] = value[batch_index]

            decoder_dict = model.generative(latent_dict_masked)
            lossrecorder = model.loss(train_loader[i], latent_dict_masked, decoder_dict, weight)
            loss = lossrecorder.loss
            loss.backward()
            #clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()
            train_loss.append(loss.detach().cpu())
    return np.array(train_loss).mean()

def _eval(model, data, edge_index, mask,val_loader, weight):
    model.eval()
    latent_dict = model.inference(data,edge_index)
    #print(latent_dict)
    latent_dict_masked = {}
    for key, value in latent_dict.items():
        if value is None:
            latent_dict_masked[key] = None
        else:
            latent_dict_masked[key] = value[mask]
            
    decoder_dict = model.generative(latent_dict_masked)
    lossrecorder = model.loss(val_loader, latent_dict_masked, decoder_dict, weight)
    return lossrecorder.loss.detach().cpu()
    
def _prob(loc,scale,value):
    ### diagonal covariance, therefore the density can be decomposed
    var = (scale * scale)
    log_scale = torch.log(scale)
    log_prob = -((value[None,:] - loc) * (value[None,:] - loc)) / (2 * var) - log_scale - torch.log(torch.tensor(math.sqrt(2 * math.pi)))
    return torch.exp(log_prob.sum(1))


        
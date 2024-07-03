from xtuner.model.modules.projector import *
from transformers import AutoConfig, AutoModel
from .configuration_projector import ProjectorConfig_OMG_LLaVA
from .modeling_projector import ProjectorModel_OMG_LLaVA

AutoConfig.register('projector', ProjectorConfig_OMG_LLaVA)
AutoModel.register(ProjectorConfig_OMG_LLaVA, ProjectorModel_OMG_LLaVA)

__all__ = ['ProjectorConfig_OMG_LLaVA', 'ProjectorModel_OMG_LLaVA']

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import wandb

class BaseLogger(ABC):

    @abstractmethod
    def init_run(self):
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict):
        pass

    @abstractmethod
    def log_metadata(self, metadata: dict):
        pass
    


class WandbLogger(BaseLogger):

    def __init__(self, 
    project: str,
    entity: str,
    group: str):
        
        self.project = project
        self.entity = entity
        self.group = group

        self._is_init = False

    def init_run(self, name: str, config: OmegaConf):
        wandb.init(
            project=self.project,
            group=self.group, # create a group in wandb for this run
            name=name,
            config=config,
        )
        print(f"Initialized wandb: {self.project}/{self.group}")
        self._is_init = True

    @property
    def is_init(self) -> bool:
        return self._is_init

    def log_metrics(self, metrics: dict):
        wandb.log(metrics)

    def log_metadata(self, metadata: dict):
        wandb.log(metadata)
    
    def finish_run(self):
        wandb.finish()

    def __del__(self):
        wandb.finish()

import abc
from dataclasses import dataclass
from pathlib import Path
import draccus

@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry,abc.ABC):
    id :str | None =None
    calibration_dir : Path | None = None

    #To get the subclasses'name of this registry
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    #check the subclasses'cameras is correct
    def __post_init__(self):
        if hasattr(self,"cameras") and self.cameras :
            for _,config in self.cameras.item():
                for attr in ["width","height","fps"]:
                    if getattr(config,attr) is None:
                        raise ValueError(
                            f"width,height,fps needed for cameras"
                        )

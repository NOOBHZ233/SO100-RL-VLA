import abc
from dataclasses import dataclass
from pathlib import Path
import draccus

@dataclass(kw_only=True)
class TeleoperatorConfig(draccus.ChoiceRegistry,abc.ABC):
    id :str | None =None
    calibration_dir : Path | None = None

    @property
    def type(self)->str :
        return self.get_choice_name(self.__class__)
    
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


foam_inlaying_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_left", "head_right", "wrist_left", "wrist_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_state", "object_state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joint_action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(foam_inlaying_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

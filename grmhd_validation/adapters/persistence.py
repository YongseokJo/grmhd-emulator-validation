from ..adapter import ModelAdapter


class Persistence(ModelAdapter):
    name = "persistence"
    version = "v1"
    history_length = 1
    max_batch = 1
    requires_checkpoint = False

    def __init__(self, checkpoint, geometry, device, options):
        if checkpoint is not None:
            raise ValueError("Persistence requests must use checkpoint: null")
        if options:
            raise ValueError("Persistence has no model options")

    def step(self, history):
        return history[:, -1].clone()

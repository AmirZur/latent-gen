from intervention_trainer import make_dataloader
from pyreft import ReftTrainer
from torch.utils.data import DataLoader

class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)
    
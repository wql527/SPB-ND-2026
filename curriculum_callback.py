class CurriculumCallback:
    def __call__(self, trainer):
        dataset = trainer.train_loader.dataset

        if not hasattr(dataset, "update_augmentations"):
            return

        dataset.epoch = trainer.epoch
        dataset.total_epochs = trainer.epochs
        dataset.update_augmentations(trainer.args)

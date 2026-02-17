import hydra

from recipes_2.finetuning import CheckpointManager, ExperimentState
from recipes_2.finetuning_setup import FineTuningConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):

  args = FineTuningConfig(**cfg)

  experiment_state = ExperimentState(args=args, saved_train_states={})
  checkpoint_manager = CheckpointManager(args.checkpointing_args, experiment_state=experiment_state)

  model = checkpoint_manager.setup_model_from_checkpoint(
        args.resume_args.resume_experiment_root,
        args.resume_args.resume_checkpoint,
        parallel_args=args.parallel_args
      )

if __name__ == "__main__":
  main()
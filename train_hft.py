import snap2midi as s2m

trainer = s2m.trainer.Trainer()

# base_path is the path to the extracted data
trainer.train_hft(base_path="data/hft_maps", logger_name="wandb", num_workers=4)
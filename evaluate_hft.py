import snap2midi as s2m

evaluator = s2m.evaluator.Evaluator()
check = "./save_dir/hft-epoch=06-valid_total_loss=0.1725.ckpt"
print(evaluator.evaluate_hft("./data/hft_maps/feature/test",check))
from refine.train.ui import TrainUI


def test_ui_renders_with_per_task_metrics():
    ui = TrainUI(run_name="t", total_steps=100, headless=True, task_names=["colorize", "denoise"])
    ui.tick(step=1, losses={"l1_rgb": 0.3, "perceptual_vgg16bn": 0.1}, lr=1e-4,
            throughput_imgs=100.0, per_task_psnr={"colorize": 20.5, "denoise": 28.1})
    frame = ui.render()
    assert frame is not None

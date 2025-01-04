## Observations from training

Tried Gaussian Fourier features for proprio/action input but did not help.

adaLN(-Zero) for time conditioning seemed to speed up training a bit initially, but did not make a significant difference after a while.

Overall, Beta sampling in flow matching timesteps achieved better validation loss, but I usually saw Uniform sampling matching or even outperforming in validation accuracy with low threshold (e.g., predicted actions within 0.05 from the normalized ground-truth ones in all dimensions). With high threshold like 0.3 or 0.5, Beta seemed to consistently outperform Uniform.

I was able to train with learning rate as high as 3e-4 with batch size 1024, thanks to the stability of the flow matching / diffusion objective?

I tried training with batch size from 256 to 2048, and the training curves (wall-clock time vs. training loss) were similar.

I switched to using [-1, 1] normalization from unit Gaussian used in Octo because I find the bridge dataset has some weird, very large action values (e.g., 80). Without clipping after being normalized with unit std, it causes a lot of spikes in training loss.

Not using pre-trained PaliGemma weights trained much worse. Training the action expert only (freezing PaliGemma) did not work at all.

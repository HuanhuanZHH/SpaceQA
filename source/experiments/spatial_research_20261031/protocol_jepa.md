# I-JEPA comparison protocol (before fits)

Investigate the user's JEPA alternative with an actual public I-JEPA image
encoder, facebook/ijepa_vith14_1k (630,762,240 backbone parameters; CC-BY-NC-4.0).
Download its immutable public weights only on icarus0, recording hashes.
V-JEPA 2.1 is also surveyed, including its 80M distilled video backbone, but
not benchmarked in this round: the current samples are single images, and
repeating one frame does not demonstrate temporal reasoning.

After the isolated 448 experiment completes, run I-JEPA on the same 224px raw
images and frozen groups as DINOv2. Frozen backbone batch8, same 64-channel
spatial decoder, 60 epochs, same seed/objective/calibration/controls. The input
dimension of the decoder follows the actual encoder width. Report its larger
memory and disk requirements. Preserve the earlier detector_results.json as
detector_results_before_jepa.json. This is an exploratory regression benchmark,
not fresh confirmation. Selection remains validation only. No formal release.

Maximum JEPA stage 30 minutes plus detector evaluation 15 minutes. A download
failure is a recorded failure, not evidence of model inapplicability. Downloads
and raw data must remain remote.

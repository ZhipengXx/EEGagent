"""What a candidate can and cannot use. Shared by the coder, the planner and the reviewer."""

CANDIDATE_INTERFACE = {
    "input": "[batch, 17, 250] float32 EEG; channels and time window are fixed by the protocol",
    "output": "[batch, 1024] finite embedding; the frozen RN50 image feature is the target",
    "build_encoder": (
        "build_encoder(self, input_spec, model_config=None) -> any torch.nn.Module defined in "
        "extension/eeg_candidate.py; it may replace EEGProjectLayer entirely with convolution, "
        "temporal pooling, attention or a new projection head"
    ),
    "fit_statistics": (
        "optional fit_statistics(self, encoder, stats); called once before training with per-channel "
        "mean and std (lists of 17 floats) computed over the training files only; store them as "
        "registered buffers, not parameters; validation reuses them"
    ),
    "not_available": [
        "subject id or any per-subject metadata",
        "per-subject statistics",
        "validation or test data before evaluation",
        "changes to the loss target, evaluator, gallery, split or image features",
    ],
    "runtime_guarantees": [
        "the validation gallery (1654 images) and query ids are fixed by the contract fingerprint",
        "image features come from the frozen DirectT RN50 cache",
        "train, validation and held-out files are assigned by the frozen split manifest",
        "the trainer imports only extension/eeg_candidate.py and records its hash",
        "training length and fidelity (pilot 3 epochs, full per goal) are set by the runtime, not by the candidate",
    ],
}

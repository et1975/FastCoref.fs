namespace FastCoref

open TorchSharp
open TorchSharp.Modules
open type TorchSharp.torch.nn
open FastCoref.Utils

/// Shared coreference-head primitives reused by both FCoref (RoBERTa) and
/// LingMess (Longformer) inference models. State_dict-compatible with the
/// Python `fastcoref` package: the F# field names below MUST equal the
/// Python attribute names verbatim so `PyBridge` can match weight keys.
module CorefHead =

    let inline private gelu (x: torch.Tensor) : torch.Tensor = torch.nn.functional.gelu (x)

    /// `Linear → GELU → LayerNorm → Dropout`. Mirrors
    /// `fastcoref.coref_models.modeling_fcoref.FullyConnectedLayer`.
    ///
    /// State_dict keys produced when owned at attribute path `<owner>.X`:
    ///   `<owner>.X.dense.weight`        (outputDim, inputDim)
    ///   `<owner>.X.dense.bias`          (outputDim,)
    ///   `<owner>.X.layer_norm.weight`   (outputDim,)
    ///   `<owner>.X.layer_norm.bias`     (outputDim,)
    type FullyConnectedLayer(inputDim: int64, outputDim: int64, dropoutProb: float32, layerNormEps: float) as this =
        inherit Module<torch.Tensor, torch.Tensor>("FullyConnectedLayer")

        let dense = Linear(inputDim, outputDim)
        let layer_norm = LayerNorm([| outputDim |], eps = layerNormEps)
        let dropout = Dropout(double dropoutProb)

        do this.RegisterComponents()

        override _.forward(x: torch.Tensor) : torch.Tensor =
            x |> dense.forward |> gelu |> layer_norm.forward |> dropout.forward

    /// Combines per-token start/end scalars with a bilinear start↔end score
    /// into a full `[B, L, L]` mention-logits matrix `m[b, s, e]`. Used
    /// identically by both FCoref and LingMess prior to span pruning.
    ///
    ///   joint        = (start_reps @ W_s2e.T + b_s2e) @ end_reps.transpose(-2,-1)
    ///   start_logits = mentionStartClf(start_reps).squeeze(-1)     # [B,L]
    ///   end_logits   = mentionEndClf(end_reps).squeeze(-1)         # [B,L]
    ///   result       = joint + start_logits[:,:,None] + end_logits[:,None,:]
    let computeMentionLogits
        (startReps: torch.Tensor) // [B, L, F]
        (endReps: torch.Tensor) // [B, L, F]
        (mentionS2E: Linear) // Linear(F, F)
        (mentionStartClf: Linear) // Linear(F, 1)
        (mentionEndClf: Linear) // Linear(F, 1)
        : torch.Tensor =
        let temp = mentionS2E.forward (startReps) // [B, L, F]
        let joint = temp.bmm (endReps.transpose (-2L, -1L)) // [B, L, L]
        let sLogit = mentionStartClf.forward(startReps).squeeze (-1L) // [B, L]
        let eLogit = mentionEndClf.forward(endReps).squeeze (-1L) // [B, L]
        joint + sLogit.unsqueeze (-1L) + eLogit.unsqueeze (-2L)

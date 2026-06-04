namespace FastCoref

open System.Collections.Generic
open TorchSharp
open TorchSharp.Modules
open TorchSharp.PyBridge
open type TorchSharp.torch.nn
open FastCoref.Utils
open FastCoref.Config

/// LingMess (Longformer-large) coreference model — composes the
/// shared `CorefHead` primitives with eight raw `nn.Parameter` tensors
/// implementing the 7-expert (6 linguistic categories + ALL) antecedent
/// scorers. State_dict keys mirror `fastcoref.coref_models.modeling_lingmess`
/// verbatim so `TorchSharp.PyBridge` can load `pytorch_model.bin` without
/// any manual key remapping.
module LingMessModel =

    /// Number of antecedent expert categories: 6 linguistic + 1 ALL.
    [<Literal>]
    let NumCats = 7L

    /// Top-level LingMess module. Owned at the root, so the antecedent
    /// `Parameter` fields produce state-dict keys without prefix
    /// (e.g. `antecedent_s2s_all_weights`), while the encoder is nested
    /// under `longformer.*` and the head FCLs / linears under their own
    /// field names. Auto-registration by `RegisterComponents()` matches
    /// each `let`-bound `Parameter` / `Module` to its F# field name.
    type LingMessModel(config: ModelConfig) as this =
        inherit Module("LingMessModel")

        let H = i64 (Hidden.value config.Encoder.HiddenSize) // 1024
        let F = i64 (Ffnn.value config.CorefHead.FfnnSize) // 2048
        let HeadOut = NumCats * F // 14336
        let dropoutP = config.CorefHead.DropoutProb
        let lnEps = float config.Encoder.LayerNormEps

        // 7-expert antecedent bilinears. Raw Parameters so the path is
        // exactly `antecedent_*_all_{weights,biases}` under this module.
        let antecedent_s2s_all_weights = Parameter(torch.empty ([| NumCats; F; F |]))
        let antecedent_e2e_all_weights = Parameter(torch.empty ([| NumCats; F; F |]))
        let antecedent_s2e_all_weights = Parameter(torch.empty ([| NumCats; F; F |]))
        let antecedent_e2s_all_weights = Parameter(torch.empty ([| NumCats; F; F |]))
        let antecedent_s2s_all_biases = Parameter(torch.empty ([| NumCats; F |]))
        let antecedent_e2e_all_biases = Parameter(torch.empty ([| NumCats; F |]))
        let antecedent_s2e_all_biases = Parameter(torch.empty ([| NumCats; F |]))
        let antecedent_e2s_all_biases = Parameter(torch.empty ([| NumCats; F |]))

        // Longformer-large encoder. Nested as `longformer.*`.
        let longformer = new Longformer.LongformerModel(config.Encoder)

        // Mention-side FCLs project hidden → ffnn for the start/end
        // halves of every candidate span.
        let start_mention_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)
        let end_mention_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)

        // Mention scorers used by `CorefHead.computeMentionLogits`.
        let mention_start_classifier = Linear(F, 1L)
        let mention_end_classifier = Linear(F, 1L)
        let mention_s2e_classifier = Linear(F, F)

        // Coref-side "all" FCLs produce 7 × ffnn-sized projections in one
        // shot; consumers reshape to [B, L, NumCats, F] for per-category
        // antecedent scoring.
        let coref_start_all_mlps =
            new CorefHead.FullyConnectedLayer(H, HeadOut, dropoutP, lnEps)

        let coref_end_all_mlps =
            new CorefHead.FullyConnectedLayer(H, HeadOut, dropoutP, lnEps)

        do this.RegisterComponents()

        /// Parsed model config (encoder + coref-head hyperparameters).
        member _.Config = config
        /// Underlying Longformer-large encoder.
        member _.Longformer = longformer

        /// Mention-side start-half FCL (Linear → GELU → LN → Dropout).
        member _.StartMentionMlp = start_mention_mlp
        /// Mention-side end-half FCL.
        member _.EndMentionMlp = end_mention_mlp

        /// `Linear(F, 1)` start-token mention scorer.
        member _.MentionStartClf = mention_start_classifier
        /// `Linear(F, 1)` end-token mention scorer.
        member _.MentionEndClf = mention_end_classifier
        /// `Linear(F, F)` bilinear projection feeding the start↔end joint score.
        member _.MentionS2EClf = mention_s2e_classifier

        /// Coref-side "all" FCL projecting hidden → `NumCats * F` start reps.
        member _.CorefStartAllMlps = coref_start_all_mlps
        /// Coref-side "all" FCL projecting hidden → `NumCats * F` end reps.
        member _.CorefEndAllMlps = coref_end_all_mlps

        /// Per-category start↔start antecedent weight tensor `[NumCats, F, F]`.
        member _.AntecedentS2SWeights = antecedent_s2s_all_weights
        /// Per-category end↔end antecedent weight tensor `[NumCats, F, F]`.
        member _.AntecedentE2EWeights = antecedent_e2e_all_weights
        /// Per-category start↔end antecedent weight tensor `[NumCats, F, F]`.
        member _.AntecedentS2EWeights = antecedent_s2e_all_weights
        /// Per-category end↔start antecedent weight tensor `[NumCats, F, F]`.
        member _.AntecedentE2SWeights = antecedent_e2s_all_weights
        /// Per-category start↔start antecedent bias `[NumCats, F]`.
        member _.AntecedentS2SBiases = antecedent_s2s_all_biases
        /// Per-category end↔end antecedent bias `[NumCats, F]`.
        member _.AntecedentE2EBiases = antecedent_e2e_all_biases
        /// Per-category start↔end antecedent bias `[NumCats, F]`.
        member _.AntecedentS2EBiases = antecedent_s2e_all_biases
        /// Per-category end↔start antecedent bias `[NumCats, F]`.
        member _.AntecedentE2SBiases = antecedent_e2s_all_biases

        /// Encodes `inputIds` with the Longformer backbone and returns
        /// `(startMentionReps, endMentionReps, startCorefAll, endCorefAll)`
        /// where mention reps are `[B, L, F]` and coref reps are reshaped
        /// to `[B, L, NumCats, F]` for downstream per-category antecedent
        /// scoring in `LingMessInference`.
        member _.ForwardReps
            (inputIds: torch.Tensor, attentionMask: torch.Tensor, globalAttentionMask: torch.Tensor)
            : torch.Tensor * torch.Tensor * torch.Tensor * torch.Tensor =
            let hidden = longformer.forward (inputIds, attentionMask, globalAttentionMask)
            let sMent = start_mention_mlp.forward (hidden)
            let eMent = end_mention_mlp.forward (hidden)
            let sCorefFlat = coref_start_all_mlps.forward (hidden)
            let eCorefFlat = coref_end_all_mlps.forward (hidden)
            let B = sCorefFlat.shape.[0]
            let L = sCorefFlat.shape.[1]
            let sCoref = sCorefFlat.view (B, L, NumCats, F)
            let eCoref = eCorefFlat.view (B, L, NumCats, F)
            sMent, eMent, sCoref, eCoref

    /// Loads a LingMess checkpoint from a HuggingFace snapshot directory.
    /// The model is constructed under the current torch default dtype;
    /// call `torch.set_default_dtype(torch.bfloat16)` beforehand to match
    /// the Python reference's bf16 inference path. Returns the model and
    /// a `LoadReport` partitioning parameter paths into loaded vs missing
    /// so callers can audit which state-dict keys were matched.
    let load (modelDir: string) (device: torch.Device) : LingMessModel * LoadReport =
        let cfg = Config.load modelDir
        let model = new LingMessModel(cfg)
        let weightPath = Utils.modelFile modelDir Utils.HfFiles.Weights
        let loaded = Dictionary<string, bool>()

        model.load_py (location = weightPath, strict = false, loadedParameters = loaded)
        |> ignore

        model.``to`` (device) |> ignore
        model, LoadReport.ofPyBridgeDict (loaded :> IReadOnlyDictionary<_, _>)

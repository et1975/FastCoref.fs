namespace FastCoref

open System.Collections.Generic
open TorchSharp
open TorchSharp.Modules
open TorchSharp.PyBridge
open type TorchSharp.torch.nn
open FastCoref.Utils
open FastCoref.Config

/// Top-level FCoref model: owns the RoBERTa encoder plus the eight mention /
/// coref projection MLPs and seven Linear classifiers that make up the
/// coreference head. Field names mirror the Python `fastcoref.coref_models.
/// modeling_fcoref.FCorefModel` attribute names so `TorchSharp.PyBridge`
/// can bind each tensor in `pytorch_model.bin` to the matching parameter.
module FCorefModel =

    /// FCoref backbone + head, weight-compatible with the HuggingFace
    /// `biu-nlp/f-coref` checkpoint. Construction is pure module wiring;
    /// weight loading is done by the companion `load` function.
    type FCorefModel(config: ModelConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor>("FCorefModel")

        let H = i64 (Hidden.value config.Encoder.HiddenSize)
        let F = i64 (Ffnn.value config.CorefHead.FfnnSize)
        let dropoutP = config.CorefHead.DropoutProb
        let lnEps = float config.Encoder.LayerNormEps

        let roberta = new Roberta.RobertaModel(config.Encoder)

        let start_mention_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)
        let end_mention_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)
        let start_coref_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)
        let end_coref_mlp = new CorefHead.FullyConnectedLayer(H, F, dropoutP, lnEps)

        let mention_start_classifier = Linear(F, 1L)
        let mention_end_classifier = Linear(F, 1L)
        let mention_s2e_classifier = Linear(F, F)

        let antecedent_s2s_classifier = Linear(F, F)
        let antecedent_e2e_classifier = Linear(F, F)
        let antecedent_s2e_classifier = Linear(F, F)
        let antecedent_e2s_classifier = Linear(F, F)

        do this.RegisterComponents()

        /// Parsed config used to construct this model.
        member _.Config = config

        /// Underlying RoBERTa encoder owning the full `roberta.*` subtree.
        member _.Roberta = roberta

        /// Projects encoder hidden states into start-of-mention reps `[B,L,F]`.
        member _.StartMentionMlp = start_mention_mlp
        /// Projects encoder hidden states into end-of-mention reps `[B,L,F]`.
        member _.EndMentionMlp = end_mention_mlp
        /// Projects encoder hidden states into start-of-coref reps `[B,L,F]`.
        member _.StartCorefMlp = start_coref_mlp
        /// Projects encoder hidden states into end-of-coref reps `[B,L,F]`.
        member _.EndCorefMlp = end_coref_mlp

        /// Per-token mention-start scalar classifier `Linear(F, 1)`.
        member _.MentionStartClf = mention_start_classifier
        /// Per-token mention-end scalar classifier `Linear(F, 1)`.
        member _.MentionEndClf = mention_end_classifier
        /// Bilinear start↔end mention classifier `Linear(F, F)`.
        member _.MentionS2EClf = mention_s2e_classifier

        /// Antecedent start-to-start bilinear `Linear(F, F)`.
        member _.AntecedentS2S = antecedent_s2s_classifier
        /// Antecedent end-to-end bilinear `Linear(F, F)`.
        member _.AntecedentE2E = antecedent_e2e_classifier
        /// Antecedent start-to-end bilinear `Linear(F, F)`.
        member _.AntecedentS2E = antecedent_s2e_classifier
        /// Antecedent end-to-start bilinear `Linear(F, F)`.
        member _.AntecedentE2S = antecedent_e2s_classifier

        /// Runs the RoBERTa encoder and returns the raw
        /// `sequence_output [B, L, H]`. Provided to satisfy the typed
        /// `Module<Tensor, Tensor, Tensor>` base — inference code should
        /// call `ForwardReps` instead.
        override _.forward(inputIds: torch.Tensor, attentionMask: torch.Tensor) : torch.Tensor =
            roberta.forward (inputIds, attentionMask)

        /// Runs the encoder and the four mention/coref projection MLPs.
        /// Returns
        /// `(startMentionReps [B,L,F], endMentionReps [B,L,F],
        ///   startCorefReps   [B,L,F], endCorefReps   [B,L,F])`.
        member _.ForwardReps
            (inputIds: torch.Tensor, attentionMask: torch.Tensor)
            : torch.Tensor * torch.Tensor * torch.Tensor * torch.Tensor =
            let hidden = roberta.forward (inputIds, attentionMask)
            let sMent = start_mention_mlp.forward (hidden)
            let eMent = end_mention_mlp.forward (hidden)
            let sCoref = start_coref_mlp.forward (hidden)
            let eCoref = end_coref_mlp.forward (hidden)
            sMent, eMent, sCoref, eCoref

        override _.ToString() =
            sprintf "FCorefModel(H=%d, F=%d)" (int H) (int F)

    /// Constructs an `FCorefModel` from a HuggingFace snapshot directory
    /// (`config.json` + `pytorch_model.bin`) and loads its weights via
    /// `TorchSharp.PyBridge`. The model is built under the current torch
    /// default dtype — call `torch.set_default_dtype(torch.bfloat16)` before
    /// invoking `load` to mirror Python `CorefModel.__init__`'s bf16 default.
    ///
    /// Returns the loaded model and a `LoadReport` partitioning parameter
    /// paths into loaded vs missing. Non-empty `Missing` indicates a
    /// state-dict key mismatch and should be surfaced to the caller —
    /// silent mismatches cause garbage inference.
    let load (modelDir: string) (device: torch.Device) : FCorefModel * LoadReport =
        let cfg = Config.load modelDir
        let model = new FCorefModel(cfg)
        let weightPath = Utils.modelFile modelDir Utils.HfFiles.Weights
        let loaded = Dictionary<string, bool>()

        model.load_py (location = weightPath, strict = false, loadedParameters = loaded)
        |> ignore

        model.``to`` (device) |> ignore
        model, LoadReport.ofPyBridgeDict (loaded :> IReadOnlyDictionary<_, _>)

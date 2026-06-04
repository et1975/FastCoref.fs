namespace FastCoref

open TorchSharp
open TorchSharp.Modules
open type TorchSharp.torch.nn
open FastCoref.Utils
open FastCoref.Config

/// Longformer-large encoder used as the backbone of the LingMess
/// coreference model (`biu-nlp/lingmess-coref`).
///
/// Field names below are part of the PUBLIC contract: they must equal
/// the Python `state_dict` attribute paths verbatim so
/// `TorchSharp.PyBridge.load_py(strict = false)` can match weight keys
/// without manual remapping.
///
/// SIMPLIFIED v1: the self-attention layer uses dense O(L²) attention
/// with a per-token blend between local and global Q/K/V projections.
/// The true HuggingFace Longformer uses sliding-window chunked
/// attention via `_sliding_chunks_query_key_matmul` / `as_strided`;
/// those operations are not surfaced ergonomically in TorchSharp
/// 0.101.5, so we punt on the optimization. Correctness over speed.
/// TODO(v2): replace dense attention with proper sliding-window
/// chunked attention.
module Longformer =

    let private gelu (x: torch.Tensor) : torch.Tensor = torch.nn.functional.gelu (x)

    // ----- Embeddings -----------------------------------------------

    /// `longformer.embeddings.*` — same layout as RoBERTa embeddings
    /// but with `MaxPositionEmbeddings = 4098` and `HiddenSize = 1024`.
    /// Position ids are built from a cumsum over the non-pad mask,
    /// offset by `PadTokenId`, matching RoBERTa's convention.
    type LongformerEmbeddings(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor>("LongformerEmbeddings")

        let word_embeddings = Embedding(i64 (Vocab.value config.VocabSize), i64 (Hidden.value config.HiddenSize))

        let position_embeddings =
            Embedding(i64 (PosEmb.value config.MaxPositionEmbeddings), i64 (Hidden.value config.HiddenSize))

        let token_type_embeddings =
            Embedding(i64 (Vocab.value config.TypeVocabSize), i64 (Hidden.value config.HiddenSize))

        let LayerNorm =
            LayerNorm([| i64 (Hidden.value config.HiddenSize) |], eps = float config.LayerNormEps)

        let dropout = Dropout(float config.HiddenDropoutProb)

        let padTokenId = i64 (TokenId.value config.PadTokenId)

        do
            // Buffer matches the Python `position_ids` registered buffer of
            // shape `[1, MaxPositionEmbeddings]`. Required so PyBridge can
            // round-trip the state_dict cleanly.
            let posIds =
                torch.arange(i64 (PosEmb.value config.MaxPositionEmbeddings), dtype = torch.ScalarType.Int64).unsqueeze (0L)

            this.register_buffer ("position_ids", posIds) |> ignore
            this.RegisterComponents()

        override _.forward(inputIds) =
            // RoBERTa-style position id construction:
            //   mask = inputIds != pad
            //   positions = cumsum(mask) * mask + pad_id
            let mask =
                inputIds.ne(Scalar.op_Implicit padTokenId).to_type (torch.ScalarType.Int64)

            let positions =
                (torch.cumsum (mask, dim = 1L) * mask) + Scalar.op_Implicit padTokenId

            let typeIds = torch.zeros_like (inputIds)

            let we = word_embeddings.forward (inputIds)
            let pe = position_embeddings.forward (positions)
            let te = token_type_embeddings.forward (typeIds)

            (we + pe + te) |> LayerNorm.forward |> dropout.forward

    // ----- Self-attention -------------------------------------------

    /// `longformer.encoder.layer.{i}.attention.self.*` — six Linear
    /// projections (local Q/K/V plus Longformer-specific
    /// global Q/K/V) plus an attention dropout.
    type LongformerSelfAttention(config: EncoderConfig, layerIdx: Layer) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>("LongformerSelfAttention")

        let H = i64 (Hidden.value config.HiddenSize)

        let query = Linear(H, H)
        let key = Linear(H, H)
        let value = Linear(H, H)
        let query_global = Linear(H, H)
        let key_global = Linear(H, H)
        let value_global = Linear(H, H)
        let dropout = Dropout(float config.AttentionProbsDropoutProb)

        let numHeads = i64 (Head.value config.NumAttentionHeads)
        let headDim = H / numHeads
        let scale = sqrtf (f32 (int headDim))

        // One-sided sliding window — unused by the v1 dense implementation
        // but kept on the type so v2 can wire it in without an API change.
        let _oneSidedWindow =
            let li = Layer.value layerIdx
            match config.AttentionWindow with
            | Some ws when ws.Length > li -> ws.[li] / 2
            | _ -> 256

        do this.RegisterComponents()

        /// SIMPLIFIED v1: dense O(L²) attention with per-row global blending.
        /// Positions with `globalAttnMask = 1` use the `*_global` projections
        /// for both their Q rows and the K/V rows the rest of the sequence
        /// attends to. Equivalent to HuggingFace's combined output for the
        /// case where every token can see every other token.
        override _.forward(x, attentionMask, globalAttnMask) =
            let B = x.shape.[0]
            let L = x.shape.[1]

            let q_local = query.forward (x)
            let k_local = key.forward (x)
            let v_local = value.forward (x)
            let q_global = query_global.forward (x)
            let k_global = key_global.forward (x)
            let v_global = value_global.forward (x)

            // [B, L, 1] — broadcasts over the hidden dimension.
            let gMaskF = globalAttnMask.to_type(x.dtype).unsqueeze (-1L)
            let invG = torch.ones_like (gMaskF) - gMaskF

            let qEff = gMaskF * q_global + invG * q_local
            let kEff = gMaskF * k_global + invG * k_local
            let vEff = gMaskF * v_global + invG * v_local

            let reshape (t: torch.Tensor) =
                t.view(B, L, numHeads, headDim).transpose (1L, 2L)

            let q = reshape qEff
            let k = reshape kEff
            let v = reshape vEff

            // [B, H, L, L]
            let scores = q.matmul (k.transpose (-2L, -1L)) / Scalar.op_Implicit scale

            // Additive mask: 0 where keep, -1e4 where padded.
            let keepF = attentionMask.to_type (scores.dtype)

            let extMask =
                (torch.ones_like (keepF) - keepF).unsqueeze(1L).unsqueeze (2L)
                * Scalar.op_Implicit -1e4f

            let scoresMasked = scores + extMask
            let attn = safeSoftmaxLastDim scoresMasked |> dropout.forward
            let ctx = attn.matmul(v).transpose(1L, 2L).contiguous().view (B, L, H)
            ctx

    /// `longformer.encoder.layer.{i}.attention.output.*` — projection,
    /// residual + LayerNorm, dropout.
    type LongformerSelfOutput(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor>("LongformerSelfOutput")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(H, H)

        let LayerNorm =
            LayerNorm([| H |], eps = float config.LayerNormEps)

        let dropout = Dropout(float config.HiddenDropoutProb)

        do this.RegisterComponents()

        override _.forward(hidden, input) =
            let h = hidden |> dense.forward |> dropout.forward
            LayerNorm.forward (h + input)

    /// `longformer.encoder.layer.{i}.attention.*` — composes the two above.
    type LongformerAttention(config: EncoderConfig, layerIdx: Layer) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>("LongformerAttention")

        let self = new LongformerSelfAttention(config, layerIdx)
        let output = new LongformerSelfOutput(config)

        do this.RegisterComponents()

        override _.forward(x, attentionMask, globalAttnMask) =
            let attended = self.forward (x, attentionMask, globalAttnMask)
            output.forward (attended, x)

    /// `longformer.encoder.layer.{i}.intermediate.*`
    type LongformerIntermediate(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor>("LongformerIntermediate")

        let dense = Linear(i64 (Hidden.value config.HiddenSize), i64 (Ffnn.value config.IntermediateSize))

        do this.RegisterComponents()

        override _.forward(x) = x |> dense.forward |> gelu

    /// `longformer.encoder.layer.{i}.output.*`
    type LongformerOutput(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor>("LongformerOutput")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(i64 (Ffnn.value config.IntermediateSize), H)

        let LayerNorm =
            LayerNorm([| H |], eps = float config.LayerNormEps)

        let dropout = Dropout(float config.HiddenDropoutProb)

        do this.RegisterComponents()

        override _.forward(hidden, input) =
            let h = hidden |> dense.forward |> dropout.forward
            LayerNorm.forward (h + input)

    /// `longformer.encoder.layer.{i}.*` — full transformer block.
    type LongformerLayer(config: EncoderConfig, layerIdx: Layer) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>("LongformerLayer")

        let attention = new LongformerAttention(config, layerIdx)
        let intermediate = new LongformerIntermediate(config)
        let output = new LongformerOutput(config)

        do this.RegisterComponents()

        override _.forward(x, attentionMask, globalAttnMask) =
            let attended = attention.forward (x, attentionMask, globalAttnMask)
            let mid = intermediate.forward (attended)
            output.forward (mid, attended)

    /// `longformer.encoder.*` — field name `layer` (singular) so the
    /// state_dict keys become `encoder.layer.{i}.*`.
    type LongformerEncoder(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>("LongformerEncoder")

        let layer =
            ModuleList<LongformerLayer>(
                [| for i in 0 .. Layer.value config.NumHiddenLayers - 1 -> new LongformerLayer(config, Layer.ofInt i) |]
            )

        let numLayers = Layer.value config.NumHiddenLayers

        do this.RegisterComponents()

        override _.forward(x, attentionMask, globalAttnMask) =
            let mutable h = x

            for i in 0 .. numLayers - 1 do
                h <- layer.[i].forward (h, attentionMask, globalAttnMask)

            h

    /// `longformer.pooler.*` — first-token tanh pooler. Unused by
    /// LingMess but present in the checkpoint, so we register it.
    type LongformerPooler(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor>("LongformerPooler")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(H, H)

        do this.RegisterComponents()

        override _.forward(hidden) =
            let cls = hidden.select (1L, 0L)
            torch.tanh (dense.forward (cls))

    /// `longformer.*` — top-level container exposed to LingMess. Owned
    /// at attribute path `longformer` so the full state_dict path is
    /// `longformer.embeddings.…` / `longformer.encoder.…` / `longformer.pooler.…`.
    type LongformerModel(config: EncoderConfig) as this =
        inherit Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>("LongformerModel")

        let embeddings = new LongformerEmbeddings(config)
        let encoder = new LongformerEncoder(config)
        let pooler = new LongformerPooler(config)

        do this.RegisterComponents()

        /// Inputs: `(inputIds, attentionMask, globalAttentionMask)`.
        /// Returns the last hidden state `[B, L, H]`. The pooled output
        /// is exposed via `Pooler` for callers that need it.
        override _.forward(inputIds, attnMask, globalAttnMask) =
            let emb = embeddings.forward (inputIds)
            encoder.forward (emb, attnMask, globalAttnMask)

        member _.Embeddings = embeddings
        member _.Encoder = encoder
        member _.Pooler = pooler

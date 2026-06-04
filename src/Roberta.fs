namespace FastCoref

open TorchSharp
open TorchSharp.Modules
open FastCoref.Config
open FastCoref.Utils

/// Eager-attention RoBERTa encoder backbone.
///
/// Field names mirror HuggingFace's `state_dict` keys exactly so that
/// `TorchSharp.PyBridge.load_py` can map weights by name. The whole subtree
/// rooted in `RobertaModel` corresponds to the `roberta.*` prefix in the
/// FCoref `.bin` (the parent `FCorefModel` owns it under a field named
/// `roberta`).
module Roberta =

    // Cast helpers must precede `open type torch` so int64/float32 aren't
    // shadowed by torch.int64 / torch.float32 (ScalarType values).
    let inline private i64 (x: ^a) : int64 = int64 x
    let inline private f32 (x: ^a) : float32 = float32 x
    let inline private sqrtf (x: float32) : float32 = sqrt x |> float32
    let inline private dbl (x: ^a) : double = double x

    open type TorchSharp.torch
    open type TorchSharp.torch.nn

    // ----------------------------------------------------------------- Embeddings

    type RobertaEmbeddings(config: EncoderConfig) as this =
        inherit Module<Tensor, Tensor>("RobertaEmbeddings")

        let word_embeddings =
            Embedding(
                i64 (Vocab.value config.VocabSize),
                i64 (Hidden.value config.HiddenSize),
                padding_idx = System.Nullable<int64>(i64 (TokenId.value config.PadTokenId))
            )

        let position_embeddings =
            Embedding(i64 (PosEmb.value config.MaxPositionEmbeddings), i64 (Hidden.value config.HiddenSize))

        let token_type_embeddings =
            Embedding(i64 (Vocab.value config.TypeVocabSize), i64 (Hidden.value config.HiddenSize))

        let LayerNorm =
            LayerNorm([| i64 (Hidden.value config.HiddenSize) |], eps = dbl config.LayerNormEps)

        let dropout = Dropout(dbl config.HiddenDropoutProb)

        let padId = i64 (TokenId.value config.PadTokenId)

        do
            let positionIds =
                torch.arange(i64 (PosEmb.value config.MaxPositionEmbeddings)).unsqueeze(0L).to_type (ScalarType.Int64)

            this.register_buffer ("position_ids", positionIds) |> ignore
            this.RegisterComponents()

        override _.forward(inputIds: Tensor) : Tensor =
            // RoBERTa-specific position id construction (HF
            // `create_position_ids_from_input_ids`): positions start at
            // padding_idx + 1 and pad positions remain at padding_idx.
            let mask = inputIds.ne(Scalar.op_Implicit padId).to_type (ScalarType.Int64)
            let incremental = torch.cumsum(mask, 1L).to_type (ScalarType.Int64) * mask
            let positionIds = incremental.add (Scalar.op_Implicit padId)

            let tokenTypeIds =
                torch.zeros_like (inputIds, dtype = System.Nullable(ScalarType.Int64))

            let we = word_embeddings.forward (inputIds)
            let pe = position_embeddings.forward (positionIds)
            let te = token_type_embeddings.forward (tokenTypeIds)
            (we + pe + te) |> LayerNorm.forward |> dropout.forward

    // ----------------------------------------------------------------- Self-attention

    type RobertaSelfAttention(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaSelfAttention")

        let H = i64 (Hidden.value config.HiddenSize)

        let query = Linear(H, H)
        let key = Linear(H, H)
        let value = Linear(H, H)
        let dropout = Dropout(dbl config.AttentionProbsDropoutProb)

        let numHeads = i64 (Head.value config.NumAttentionHeads)
        // Per-head hidden dim. Encoded as `hidden / head` via the divisor
        // measures naturally; we strip to bare int64 here because TorchSharp's
        // `view` / `transpose` APIs work in measure-less shapes.
        let headDim = H / numHeads
        let hiddenSize = H
        let scale = sqrtf (f32 headDim)

        do base.RegisterComponents()

        override _.forward(x: Tensor, attentionMask: Tensor) : Tensor =
            let B = x.shape.[0]
            let L = x.shape.[1]

            let q = query.forward(x).view(B, L, numHeads, headDim).transpose (1L, 2L)
            let k = key.forward(x).view(B, L, numHeads, headDim).transpose (1L, 2L)
            let v = value.forward(x).view(B, L, numHeads, headDim).transpose (1L, 2L)

            let scores = q.matmul (k.transpose (-2L, -1L)) / scale

            // attentionMask: [B, L] with 1.0 = keep, 0.0 = pad. Convert to
            // additive bias broadcastable to [B, 1, 1, L].
            let extMask =
                attentionMask
                    .to_type(scores.dtype)
                    .mul(Scalar.op_Implicit -1.0f)
                    .add(Scalar.op_Implicit 1.0f)
                    .unsqueeze(1L)
                    .unsqueeze(2L)
                    .mul (Scalar.op_Implicit -1e4f)

            let attn = (scores + extMask) |> safeSoftmaxLastDim |> dropout.forward

            attn.matmul(v).transpose(1L, 2L).contiguous().view (B, L, hiddenSize)

    // ----------------------------------------------------------------- Self-output

    type RobertaSelfOutput(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaSelfOutput")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(H, H)

        let LayerNorm =
            LayerNorm([| H |], eps = dbl config.LayerNormEps)

        let dropout = Dropout(dbl config.HiddenDropoutProb)

        do base.RegisterComponents()

        override _.forward(hidden: Tensor, input: Tensor) : Tensor =
            let h = dense.forward (hidden) |> dropout.forward
            LayerNorm.forward (h + input)

    // ----------------------------------------------------------------- Attention wrapper

    type RobertaAttention(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaAttention")

        // Field literally named `self` to match HF state_dict key
        // `...attention.self.query.weight`. Not an F# keyword.
        let self = new RobertaSelfAttention(config)
        let output = new RobertaSelfOutput(config)

        do base.RegisterComponents()

        override _.forward(x: Tensor, mask: Tensor) : Tensor =
            let attnOut = self.forward (x, mask)
            output.forward (attnOut, x)

    // ----------------------------------------------------------------- Intermediate

    type RobertaIntermediate(config: EncoderConfig) =
        inherit Module<Tensor, Tensor>("RobertaIntermediate")

        let dense = Linear(i64 (Hidden.value config.HiddenSize), i64 (Ffnn.value config.IntermediateSize))

        do base.RegisterComponents()

        override _.forward(x: Tensor) : Tensor = dense.forward (x) |> functional.gelu

    // ----------------------------------------------------------------- Output

    type RobertaOutput(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaOutput")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(i64 (Ffnn.value config.IntermediateSize), H)

        let LayerNorm =
            LayerNorm([| H |], eps = dbl config.LayerNormEps)

        let dropout = Dropout(dbl config.HiddenDropoutProb)

        do base.RegisterComponents()

        override _.forward(hidden: Tensor, input: Tensor) : Tensor =
            let h = dense.forward (hidden) |> dropout.forward
            LayerNorm.forward (h + input)

    // ----------------------------------------------------------------- Layer

    type RobertaLayer(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaLayer")

        let attention = new RobertaAttention(config)
        let intermediate = new RobertaIntermediate(config)
        let output = new RobertaOutput(config)

        do base.RegisterComponents()

        override _.forward(x: Tensor, mask: Tensor) : Tensor =
            let a = attention.forward (x, mask)
            let i = intermediate.forward (a)
            output.forward (i, a)

    // ----------------------------------------------------------------- Encoder (stack)

    type RobertaEncoder(config: EncoderConfig) =
        inherit Module<Tensor, Tensor, Tensor>("RobertaEncoder")

        // Field name `layer` (singular) to match HF keys
        // `roberta.encoder.layer.{i}.*`.
        let layer =
            ModuleList<RobertaLayer>([| for _ in 1 .. Layer.value config.NumHiddenLayers -> new RobertaLayer(config) |])

        let nLayers = Layer.value config.NumHiddenLayers

        do base.RegisterComponents()

        override _.forward(x: Tensor, mask: Tensor) : Tensor =
            let mutable h = x

            for i in 0 .. nLayers - 1 do
                h <- layer.[i].forward (h, mask)

            h

    // ----------------------------------------------------------------- Pooler

    type RobertaPooler(config: EncoderConfig) =
        inherit Module<Tensor, Tensor>("RobertaPooler")

        let H = i64 (Hidden.value config.HiddenSize)
        let dense = Linear(H, H)

        do base.RegisterComponents()

        override _.forward(x: Tensor) : Tensor =
            // x: [B, L, H] -> first token rep, project, tanh -> [B, H]
            let first = x.select (1L, 0L)
            torch.tanh (dense.forward (first))

    // ----------------------------------------------------------------- Top-level model

    /// Owns the full `roberta.*` subtree (embeddings + encoder + pooler).
    /// Forward returns `sequence_output [B, L, H]`. The pooled vector is
    /// unused by the FCoref head, but the pooler is kept so PyBridge can
    /// bind its weights from the checkpoint.
    type RobertaModel(config: EncoderConfig) as this =
        inherit Module<Tensor, Tensor, Tensor>("RobertaModel")

        let embeddings = new RobertaEmbeddings(config)
        let encoder = new RobertaEncoder(config)
        let pooler = new RobertaPooler(config)

        do this.RegisterComponents()

        override _.forward(inputIds: Tensor, attentionMask: Tensor) : Tensor =
            let emb = embeddings.forward (inputIds)
            encoder.forward (emb, attentionMask)

        member _.Embeddings = embeddings
        member _.Encoder = encoder
        member _.Pooler = pooler

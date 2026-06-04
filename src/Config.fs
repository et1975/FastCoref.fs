namespace FastCoref

open System.Collections.Generic
open System.IO
open System.Text.Json
open FSharp.UMX

/// Parses the HuggingFace `config.json` shipped with `biu-nlp/f-coref`
/// (RoBERTa backbone) and `biu-nlp/lingmess-coref` (Longformer backbone).
module Config =

    // Phantom measure tags — lowercase per F# convention. Callers never
    // use these directly; they go through the PascalCase aliases and their
    // `[<RequireQualifiedAccess>]` companion modules declared below, which
    // are the sole entry points for tagging/untagging.
    [<RequireQualifiedAccess>]
    module Measures =
        [<Measure>] type tokenId
        [<Measure>] type layer
        [<Measure>] type head
        [<Measure>] type hidden
        [<Measure>] type ffnn
        [<Measure>] type vocab
        [<Measure>] type posEmb
        [<Measure>] type span
        [<Measure>] type segment

    /// Vocabulary-token ID (BPE vocab index). Distinct from `Clustering.TokenIdx`
    /// (a token *position* in a sequence): a token at position 5 has unrelated
    /// identity from vocab-id 5. Lives here rather than in `Tokenizer.fs`
    /// because `EncoderConfig.PadTokenId` / `BosTokenId` / `EosTokenId` are
    /// token IDs and `Config.fs` precedes `Tokenizer.fs` in compilation order.
    type TokenId = int<Measures.tokenId>

    [<RequireQualifiedAccess>]
    module TokenId =
        let inline ofInt (x: int) : TokenId = UMX.tag<Measures.tokenId> x
        let inline value (x: TokenId) : int = UMX.untag x
        let inline ofArray (xs: int[]) : TokenId[] = UMXArr.tag<Measures.tokenId> xs

    /// Transformer-layer index in the encoder stack.
    type Layer = int<Measures.layer>

    [<RequireQualifiedAccess>]
    module Layer =
        let inline ofInt (x: int) : Layer = UMX.tag<Measures.layer> x
        let inline value (x: Layer) : int = UMX.untag x

    /// Attention-head index within a transformer layer.
    type Head = int<Measures.head>

    [<RequireQualifiedAccess>]
    module Head =
        let inline ofInt (x: int) : Head = UMX.tag<Measures.head> x
        let inline value (x: Head) : int = UMX.untag x

    /// Encoder hidden dimension (`HiddenSize`).
    type Hidden = int<Measures.hidden>

    [<RequireQualifiedAccess>]
    module Hidden =
        let inline ofInt (x: int) : Hidden = UMX.tag<Measures.hidden> x
        let inline value (x: Hidden) : int = UMX.untag x

    /// FFN intermediate dimension (`IntermediateSize` / `FfnnSize`).
    type Ffnn = int<Measures.ffnn>

    [<RequireQualifiedAccess>]
    module Ffnn =
        let inline ofInt (x: int) : Ffnn = UMX.tag<Measures.ffnn> x
        let inline value (x: Ffnn) : int = UMX.untag x

    /// Vocabulary size — a *count* of tokens, not a token ID.
    type Vocab = int<Measures.vocab>

    [<RequireQualifiedAccess>]
    module Vocab =
        let inline ofInt (x: int) : Vocab = UMX.tag<Measures.vocab> x
        let inline value (x: Vocab) : int = UMX.untag x

    /// Maximum position-embedding count.
    type PosEmb = int<Measures.posEmb>

    [<RequireQualifiedAccess>]
    module PosEmb =
        let inline ofInt (x: int) : PosEmb = UMX.tag<Measures.posEmb> x
        let inline value (x: PosEmb) : int = UMX.untag x

    /// Maximum mention span length, in tokens. Distinct from `Segment` so
    /// an accidental swap with `MaxSegmentLen` fails to compile.
    type Span = int<Measures.span>

    [<RequireQualifiedAccess>]
    module Span =
        let inline ofInt (x: int) : Span = UMX.tag<Measures.span> x
        let inline value (x: Span) : int = UMX.untag x

    /// Encoder segment length, in tokens. Distinct from `Span`.
    type Segment = int<Measures.segment>

    [<RequireQualifiedAccess>]
    module Segment =
        let inline ofInt (x: int) : Segment = UMX.tag<Measures.segment> x
        let inline value (x: Segment) : int = UMX.untag x

    type CorefHeadConfig =
        { FfnnSize: Ffnn
          TopLambda: float32
          MaxSpanLength: Span
          MaxSegmentLen: Segment
          DropoutProb: float32 }

    /// Common shape for both RoBERTa- and Longformer-style encoders.
    /// Longformer-only fields are encoded as `Option`s.
    type EncoderConfig =
        { HiddenSize: Hidden
          NumHiddenLayers: Layer
          NumAttentionHeads: Head
          IntermediateSize: Ffnn
          MaxPositionEmbeddings: PosEmb
          TypeVocabSize: Vocab
          VocabSize: Vocab
          PadTokenId: TokenId
          BosTokenId: TokenId
          EosTokenId: TokenId
          LayerNormEps: float32
          HiddenDropoutProb: float32
          AttentionProbsDropoutProb: float32
          AttentionWindow: int[] option }

    /// Which coreference backbone a checkpoint is for. The HuggingFace
    /// `model_type` string is sniffed exactly once (`ModelKind.ofModelTypeString`)
    /// so downstream code can pattern-match instead of re-doing the substring
    /// check at every call site.
    type ModelKind =
        | FCoref
        | LingMess

    [<RequireQualifiedAccess>]
    module ModelKind =
        /// Sniff the HuggingFace `model_type` string. The only place that
        /// knows the substring rule.
        let ofModelTypeString (s: string) : ModelKind =
            let m = s.ToLowerInvariant()
            if m.Contains "lingmess" || m.Contains "longformer" then LingMess else FCoref

    type ModelConfig =
        { Kind: ModelKind
          RawModelType: string
          Encoder: EncoderConfig
          CorefHead: CorefHeadConfig }

    /// Result of loading a checkpoint via `TorchSharp.PyBridge.load_py`.
    /// Splits the rank-2 partition (loaded vs missing) out of the raw
    /// `Dictionary<string,bool>` that PyBridge populates, so callers don't
    /// have to re-walk it. `Missing` paths kept their random initialisation
    /// and almost always indicate a state-dict key mismatch worth surfacing.
    /// PyBridge's `strict = false` does not surface unexpected-keys so we do
    /// not model them.
    type LoadReport =
        { Loaded: IReadOnlyList<string>
          Missing: IReadOnlyList<string>
          Total: int }

    [<RequireQualifiedAccess>]
    module LoadReport =
        let ofPyBridgeDict (d: IReadOnlyDictionary<string, bool>) : LoadReport =
            let loaded = ResizeArray()
            let missing = ResizeArray()

            for KeyValue(k, v) in d do
                if v then loaded.Add k else missing.Add k

            { Loaded = loaded :> IReadOnlyList<string>
              Missing = missing :> IReadOnlyList<string>
              Total = d.Count }

        let isClean (r: LoadReport) = r.Missing.Count = 0

    /// Defaults used when a key is missing in `config.json`.
    /// Values differ between FCoref (RoBERTa-base) and LingMess (Longformer-large).
    let private corefHeadDefaults (kind: ModelKind) : CorefHeadConfig =
        match kind with
        | LingMess ->
            { FfnnSize = Ffnn.ofInt 2048
              TopLambda = 0.40f
              MaxSpanLength = Span.ofInt 30
              MaxSegmentLen = Segment.ofInt 4096
              DropoutProb = 0.3f }
        | FCoref ->
            { FfnnSize = Ffnn.ofInt 1024
              TopLambda = 0.25f
              MaxSpanLength = Span.ofInt 30
              MaxSegmentLen = Segment.ofInt 512
              DropoutProb = 0.3f }

    let private tryGet (root: JsonElement) (key: string) : JsonElement option =
        match root.TryGetProperty(key) with
        | true, v when v.ValueKind <> JsonValueKind.Null -> Some v
        | _ -> None

    let private getInt root key =
        (tryGet root key |> Option.get).GetInt32()

    let private getIntOr root key dflt =
        match tryGet root key with
        | Some v -> v.GetInt32()
        | None -> dflt

    let private getFloatOr root key dflt =
        match tryGet root key with
        | Some v -> v.GetSingle()
        | None -> dflt

    let private getStrOr root key dflt =
        match tryGet root key with
        | Some v -> v.GetString()
        | None -> dflt

    let private parseEncoder (root: JsonElement) : EncoderConfig =
        let attentionWindow =
            match tryGet root "attention_window" with
            | Some v when v.ValueKind = JsonValueKind.Array ->
                v.EnumerateArray() |> Seq.map (fun e -> e.GetInt32()) |> Array.ofSeq |> Some
            | _ -> None

        { HiddenSize = Hidden.ofInt (getInt root "hidden_size")
          NumHiddenLayers = Layer.ofInt (getInt root "num_hidden_layers")
          NumAttentionHeads = Head.ofInt (getInt root "num_attention_heads")
          IntermediateSize = Ffnn.ofInt (getIntOr root "intermediate_size" 3072)
          MaxPositionEmbeddings = PosEmb.ofInt (getIntOr root "max_position_embeddings" 514)
          TypeVocabSize = Vocab.ofInt (getIntOr root "type_vocab_size" 1)
          VocabSize = Vocab.ofInt (getIntOr root "vocab_size" 50265)
          PadTokenId = TokenId.ofInt (getIntOr root "pad_token_id" 1)
          BosTokenId = TokenId.ofInt (getIntOr root "bos_token_id" 0)
          EosTokenId = TokenId.ofInt (getIntOr root "eos_token_id" 2)
          LayerNormEps = getFloatOr root "layer_norm_eps" 1e-5f
          HiddenDropoutProb = getFloatOr root "hidden_dropout_prob" 0.1f
          AttentionProbsDropoutProb = getFloatOr root "attention_probs_dropout_prob" 0.1f
          AttentionWindow = attentionWindow }

    let private parseCorefHead (root: JsonElement) (defaults: CorefHeadConfig) : CorefHeadConfig =
        match tryGet root "coref_head" with
        | None -> defaults
        | Some head ->
            { FfnnSize = Ffnn.ofInt (getIntOr head "ffnn_size" (Ffnn.value defaults.FfnnSize))
              TopLambda = getFloatOr head "top_lambda" defaults.TopLambda
              MaxSpanLength = Span.ofInt (getIntOr head "max_span_length" (Span.value defaults.MaxSpanLength))
              MaxSegmentLen = Segment.ofInt (getIntOr head "max_segment_len" (Segment.value defaults.MaxSegmentLen))
              DropoutProb = getFloatOr head "dropout_prob" defaults.DropoutProb }

    let load (modelDir: string) : ModelConfig =
        let path = Utils.modelFile modelDir Utils.HfFiles.Config
        use stream = File.OpenRead(path)
        use doc = JsonDocument.Parse(stream)
        let root = doc.RootElement
        let modelTypeStr = getStrOr root "model_type" "fcoref"
        let kind = ModelKind.ofModelTypeString modelTypeStr
        let defaults = corefHeadDefaults kind

        { Kind = kind
          RawModelType = modelTypeStr
          Encoder = parseEncoder root
          CorefHead = parseCorefHead root defaults }

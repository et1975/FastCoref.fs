module FastCoref.Tests.ConfigProperties

open System
open System.IO
open System.Text.Json
open FsCheck
open FsCheck.FSharp
open FsCheck.Xunit
open FastCoref
open FastCoref.Config

// --- JSON writers ------------------------------------------------------------

let private writeEncoderFields (w: Utf8JsonWriter) (cfg: Config.ModelConfig) =
    w.WriteString("model_type", cfg.RawModelType)
    w.WriteNumber("hidden_size", Hidden.value cfg.Encoder.HiddenSize)
    w.WriteNumber("num_hidden_layers", Layer.value cfg.Encoder.NumHiddenLayers)
    w.WriteNumber("num_attention_heads", Head.value cfg.Encoder.NumAttentionHeads)
    w.WriteNumber("intermediate_size", Ffnn.value cfg.Encoder.IntermediateSize)
    w.WriteNumber("max_position_embeddings", PosEmb.value cfg.Encoder.MaxPositionEmbeddings)
    w.WriteNumber("type_vocab_size", Vocab.value cfg.Encoder.TypeVocabSize)
    w.WriteNumber("vocab_size", Vocab.value cfg.Encoder.VocabSize)
    w.WriteNumber("pad_token_id", TokenId.value cfg.Encoder.PadTokenId)
    w.WriteNumber("bos_token_id", TokenId.value cfg.Encoder.BosTokenId)
    w.WriteNumber("eos_token_id", TokenId.value cfg.Encoder.EosTokenId)
    w.WriteNumber("layer_norm_eps", cfg.Encoder.LayerNormEps)
    w.WriteNumber("hidden_dropout_prob", cfg.Encoder.HiddenDropoutProb)
    w.WriteNumber("attention_probs_dropout_prob", cfg.Encoder.AttentionProbsDropoutProb)
    match cfg.Encoder.AttentionWindow with
    | Some arr ->
        w.WriteStartArray("attention_window")
        for x in arr do w.WriteNumberValue(x)
        w.WriteEndArray()
    | None -> ()

let private writeConfigJson (path: string) (cfg: Config.ModelConfig) =
    use stream = File.Create path
    use w = new Utf8JsonWriter(stream)
    w.WriteStartObject()
    writeEncoderFields w cfg
    w.WriteStartObject("coref_head")
    w.WriteNumber("ffnn_size", Ffnn.value cfg.CorefHead.FfnnSize)
    w.WriteNumber("top_lambda", cfg.CorefHead.TopLambda)
    w.WriteNumber("max_span_length", Span.value cfg.CorefHead.MaxSpanLength)
    w.WriteNumber("max_segment_len", Segment.value cfg.CorefHead.MaxSegmentLen)
    w.WriteNumber("dropout_prob", cfg.CorefHead.DropoutProb)
    w.WriteEndObject()
    w.WriteEndObject()

let private writeConfigJsonNoHead (path: string) (cfg: Config.ModelConfig) =
    use stream = File.Create path
    use w = new Utf8JsonWriter(stream)
    w.WriteStartObject()
    writeEncoderFields w cfg
    w.WriteEndObject()

// --- Temp dir helper ---------------------------------------------------------

let private withTempDir (action: string -> 'a) : 'a =
    let dir = Path.Combine(Path.GetTempPath(),
                           "FastCoref.Tests." + Guid.NewGuid().ToString("N"))
    try
        Directory.CreateDirectory dir |> ignore
        action dir
    finally
        try Directory.Delete(dir, true) with _ -> ()

// --- Generators --------------------------------------------------------------

type ConfigArbs =
    static member ModelConfig () =
        gen {
            let! hidden    = Gen.choose (1, 4096)
            let! layers    = Gen.choose (1, 48)
            let! heads     = Gen.choose (1, 64)
            let! intermed  = Gen.choose (1, 16384)
            let! maxPos    = Gen.choose (1, 8192)
            let! typeVocab = Gen.choose (1, 8)
            let! vocab     = Gen.choose (1, 100000)
            let! pad       = Gen.choose (0, 100)
            let! bos       = Gen.choose (0, 100)
            let! eos       = Gen.choose (0, 100)
            let! lnEps     = Gen.choose (1, 1000) |> Gen.map (fun n -> float32 n * 1e-7f)
            let! hidDrop   = Gen.choose (0, 100) |> Gen.map (fun n -> float32 n / 100.0f)
            let! attnDrop  = Gen.choose (0, 100) |> Gen.map (fun n -> float32 n / 100.0f)
            let! attnWindow =
                Gen.frequency [
                    1, Gen.constant None
                    1, gen {
                        let! n = Gen.choose (1, 24)
                        let! ws = Gen.choose (32, 1024) |> Gen.listOfLength n
                        return Some (List.toArray ws)
                    }
                ]
            let encoder : Config.EncoderConfig =
                { HiddenSize = Hidden.ofInt hidden
                  NumHiddenLayers = Layer.ofInt layers
                  NumAttentionHeads = Head.ofInt heads
                  IntermediateSize = Ffnn.ofInt intermed
                  MaxPositionEmbeddings = PosEmb.ofInt maxPos
                  TypeVocabSize = Vocab.ofInt typeVocab
                  VocabSize = Vocab.ofInt vocab
                  PadTokenId = TokenId.ofInt pad
                  BosTokenId = TokenId.ofInt bos
                  EosTokenId = TokenId.ofInt eos
                  LayerNormEps = lnEps
                  HiddenDropoutProb = hidDrop
                  AttentionProbsDropoutProb = attnDrop
                  AttentionWindow = attnWindow }

            let! ffnnSize  = Gen.choose (1, 8192)
            let! topLambda = Gen.choose (1, 100) |> Gen.map (fun n -> float32 n / 100.0f)
            let! maxSpan   = Gen.choose (1, 100)
            let! maxSegLen = Gen.choose (1, 8192)
            let! dropoutP  = Gen.choose (0, 100) |> Gen.map (fun n -> float32 n / 100.0f)
            let corefHead : Config.CorefHeadConfig =
                { FfnnSize = Ffnn.ofInt ffnnSize
                  TopLambda = topLambda
                  MaxSpanLength = Span.ofInt maxSpan
                  MaxSegmentLen = Segment.ofInt maxSegLen
                  DropoutProb = dropoutP }

            let! modelType =
                Gen.elements
                    [ "fcoref"; "roberta"; "lingmess"; "longformer"; "lingmess_coref" ]

            return
                ({ Kind = Config.ModelKind.ofModelTypeString modelType
                   RawModelType = modelType
                   Encoder = encoder
                   CorefHead = corefHead } : Config.ModelConfig)
        }
        |> Arb.fromGen

// --- Properties --------------------------------------------------------------

let private closeF32 (a: float32) (b: float32) =
    abs (a - b) <= 1e-6f * max 1.0f (abs a)

[<Property(Arbitrary = [| typeof<ConfigArbs> |])>]
let ``ModelConfig round-trips through JSON`` (cfg: Config.ModelConfig) =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, Utils.HfFiles.Config)
        writeConfigJson path cfg
        let loaded = Config.load dir
        loaded.RawModelType = cfg.RawModelType
        && loaded.Kind = cfg.Kind
        && loaded.Encoder.HiddenSize             = cfg.Encoder.HiddenSize
        && loaded.Encoder.NumHiddenLayers        = cfg.Encoder.NumHiddenLayers
        && loaded.Encoder.NumAttentionHeads      = cfg.Encoder.NumAttentionHeads
        && loaded.Encoder.IntermediateSize       = cfg.Encoder.IntermediateSize
        && loaded.Encoder.MaxPositionEmbeddings  = cfg.Encoder.MaxPositionEmbeddings
        && loaded.Encoder.TypeVocabSize          = cfg.Encoder.TypeVocabSize
        && loaded.Encoder.VocabSize              = cfg.Encoder.VocabSize
        && loaded.Encoder.PadTokenId             = cfg.Encoder.PadTokenId
        && loaded.Encoder.BosTokenId             = cfg.Encoder.BosTokenId
        && loaded.Encoder.EosTokenId             = cfg.Encoder.EosTokenId
        && closeF32 loaded.Encoder.LayerNormEps              cfg.Encoder.LayerNormEps
        && closeF32 loaded.Encoder.HiddenDropoutProb         cfg.Encoder.HiddenDropoutProb
        && closeF32 loaded.Encoder.AttentionProbsDropoutProb cfg.Encoder.AttentionProbsDropoutProb
        && loaded.Encoder.AttentionWindow        = cfg.Encoder.AttentionWindow
        && loaded.CorefHead.FfnnSize             = cfg.CorefHead.FfnnSize
        && closeF32 loaded.CorefHead.TopLambda   cfg.CorefHead.TopLambda
        && loaded.CorefHead.MaxSpanLength        = cfg.CorefHead.MaxSpanLength
        && loaded.CorefHead.MaxSegmentLen        = cfg.CorefHead.MaxSegmentLen
        && closeF32 loaded.CorefHead.DropoutProb cfg.CorefHead.DropoutProb)

[<Property(Arbitrary = [| typeof<ConfigArbs> |])>]
let ``missing coref_head triggers model-type-aware defaults`` (cfg: Config.ModelConfig) =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, Utils.HfFiles.Config)
        writeConfigJsonNoHead path cfg
        let loaded = Config.load dir
        let h = loaded.CorefHead
        match cfg.Kind with
        | Config.LingMess ->
            Ffnn.value h.FfnnSize = 2048
            && h.TopLambda = 0.40f
            && Span.value h.MaxSpanLength = 30
            && Segment.value h.MaxSegmentLen = 4096
            && h.DropoutProb = 0.3f
        | Config.FCoref ->
            Ffnn.value h.FfnnSize = 1024
            && h.TopLambda = 0.25f
            && Span.value h.MaxSpanLength = 30
            && Segment.value h.MaxSegmentLen = 512
            && h.DropoutProb = 0.3f)

[<Property>]
let ``minimal config loads with documented defaults``
    (NonNegativeInt hsRaw) (NonNegativeInt lyRaw) (NonNegativeInt hdRaw) =
    let hs = (hsRaw % 4096) + 1
    let ly = (lyRaw % 48) + 1
    let hd = (hdRaw % 64) + 1
    withTempDir (fun dir ->
        let path = Path.Combine(dir, Utils.HfFiles.Config)
        let json =
            sprintf
                "{\"hidden_size\":%d,\"num_hidden_layers\":%d,\"num_attention_heads\":%d}"
                hs ly hd
        File.WriteAllText(path, json)
        let loaded = Config.load dir
        loaded.RawModelType = "fcoref"
        && loaded.Kind = Config.FCoref
        && Hidden.value loaded.Encoder.HiddenSize = hs
        && Layer.value loaded.Encoder.NumHiddenLayers = ly
        && Head.value loaded.Encoder.NumAttentionHeads = hd
        && Ffnn.value loaded.Encoder.IntermediateSize = 3072
        && PosEmb.value loaded.Encoder.MaxPositionEmbeddings = 514
        && Vocab.value loaded.Encoder.TypeVocabSize = 1
        && Vocab.value loaded.Encoder.VocabSize = 50265
        && TokenId.value loaded.Encoder.PadTokenId = 1
        && TokenId.value loaded.Encoder.BosTokenId = 0
        && TokenId.value loaded.Encoder.EosTokenId = 2
        && closeF32 loaded.Encoder.LayerNormEps 1e-5f
        && closeF32 loaded.Encoder.HiddenDropoutProb 0.1f
        && closeF32 loaded.Encoder.AttentionProbsDropoutProb 0.1f
        && loaded.Encoder.AttentionWindow = None
        && Ffnn.value loaded.CorefHead.FfnnSize = 1024
        && loaded.CorefHead.TopLambda = 0.25f
        && Segment.value loaded.CorefHead.MaxSegmentLen = 512)

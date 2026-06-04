namespace FastCoref

open System
open System.Collections.Generic
open TorchSharp
open FSharp.UMX

/// Public, user-facing surface for the FastCoref library. `CorefModel`
/// wraps the lower-level `FCorefInference` / `LingMessInference` modules
/// behind a single stateful class; `CorefKind` selects the backbone.
///
/// `module rec` is used so that `CorefResult.ToJson()` (defined inline on
/// the record) can reference the private `JsonDto` module defined further
/// down, without splitting the cohesive type definition.
module rec Api =

    open FastCoref.Config
    open FastCoref.Tokenizer
    open FastCoref.Clustering
    open FastCoref.FCorefModel
    open FastCoref.LingMessModel
    open FastCoref.FCorefInference
    open FastCoref.LingMessInference

    [<RequireQualifiedAccess>]
    module Measures =

        [<Measure>]
        type chr
        [<Measure>]
        type wrd

    /// Which coreference backbone a `CorefModel` should drive. Mirrors
    /// `Config.ModelKind`; the constructor cross-checks the two to surface
    /// "I asked for FCoref but the checkpoint is LingMess" mismatches.
    [<RequireQualifiedAccess>]
    type CorefKind =
        | FCoref
        | LingMess

    /// Character offset into the original input text (UTF-16 code-unit
    /// index). Distinct from `Clustering.TokenIdx` and `Config.TokenId`;
    /// a swap between any two is a compile error.
    type CharIdx = int<Measures.chr>

    [<RequireQualifiedAccess>]
    module CharIdx =
        let inline ofInt (x: int) : CharIdx = UMX.tag<Measures.chr> x
        let inline value (x: CharIdx) : int = UMX.untag x

    /// Half-open `[Start..End)` character span. `End` is exclusive so
    /// `text.Substring(int Start, int End - int Start)` recovers the
    /// surface mention verbatim.
    type CharSpan = { Start: CharIdx; End: CharIdx }

    /// One mention in a coreference cluster: the character span plus the
    /// surface substring extracted from the input text. Pre-cached so
    /// downstream code does not need the original text in hand.
    type Mention = { Span: CharSpan; Text: string }

    /// Word-index measure. Distinct from `chr` (character offset) and `tok`
    /// (post-BPE token position) so a swap between any two is a compile
    /// error.
    type WordIdx = int<Measures.wrd>

    [<RequireQualifiedAccess>]
    module WordIdx =
        let inline ofInt (x: int) : WordIdx = UMX.tag<Measures.wrd> x
        let inline value (x: WordIdx) : int = UMX.untag x

    /// Half-open `[Start..End)` word-index span. `End` is exclusive so
    /// `words.[int Start .. int End - 1]` recovers the surface words.
    type WordSpan = { Start: WordIdx; End: WordIdx }

    /// One mention from a pre-tokenized input: the word-index span plus the
    /// materialised surface text (space-joined).
    type WordMention = { Span: WordSpan; Text: string }

    /// Result of `CorefModel.Predict(words)`. `UnalignedMentions` carries any
    /// char-level mentions whose char span did not align cleanly on the
    /// space-joined word boundaries — surfaces tokenization edge cases
    /// rather than swallowing them silently. Clusters with fewer than 2
    /// aligned mentions after filtering are dropped (the `Cluster<_>`
    /// invariant).
    ///
    /// `Logits` is the *same* `CorefLogits` instance as the char-level
    /// `CorefResult` produced internally; `Release()` on it invalidates
    /// both the CharSpan and WordSpan lookups.
    type WordCorefResult =
        { Words: string[]
          Clusters: Cluster<WordMention> list
          UnalignedMentions: Mention list
          Logits: CorefLogits }

    /// One coref-resolution result for a single input text.
    type CorefResult =
        { Text: string
          Clusters: Cluster<Mention> list
          Logits: CorefLogits }

        /// Returns each cluster as a list of substring mentions.
        /// Equivalent to Python `CorefResult.get_clusters(as_strings=True)`.
        member this.GetClustersAsStrings() : string list list =
            this.Clusters
            |> List.map (fun c -> Cluster.toList c |> List.map (fun m -> m.Text))

        /// Substitute every non-head mention in each cluster with that cluster's
        /// head text, returning the resolved string. Verbatim substitution — no
        /// linguistic post-processing:
        ///   * no POS tagging
        ///   * no pronoun-case agreement / gender matching
        ///   * cluster head is always `Cluster.Head` (first mention by storage order)
        ///   * overlapping mentions: leftmost-longest wins, others skipped
        /// For pronoun-aware resolution, callers should post-process this output
        /// with their own NLP tooling.
        member this.ResolveText() : string =
            // 1. Collect every (span, replacement) pair: for each cluster,
            //    replacement = Head.Text; spans = every Rest mention's CharSpan.
            let replacements =
                this.Clusters
                |> List.collect (fun cluster ->
                    let head = cluster.Head
                    cluster.Rest |> List.map (fun m -> m.Span, head.Text))

            // 2. Sort by (Start asc, Length desc) so longest-overlap wins.
            let sorted =
                replacements
                |> List.sortWith (fun (a, _) (b, _) ->
                    let sa = CharIdx.value a.Start
                    let sb = CharIdx.value b.Start

                    if sa <> sb then
                        compare sa sb
                    else
                        let la = CharIdx.value a.End - sa
                        let lb = CharIdx.value b.End - sb
                        compare lb la) // longer first

            // 3. Walk left-to-right with cursor; skip overlapping; concatenate.
            let sb = System.Text.StringBuilder(this.Text.Length)
            let mutable cursor = 0

            for (span, replacement) in sorted do
                let s = CharIdx.value span.Start
                let e = CharIdx.value span.End

                if s >= cursor then
                    sb.Append(this.Text.Substring(cursor, s - cursor)) |> ignore
                    sb.Append(replacement) |> ignore
                    cursor <- e

            if cursor < this.Text.Length then
                sb.Append(this.Text.Substring(cursor)) |> ignore

            sb.ToString()

        /// Serialise to a single-line JSON string. Schema:
        ///   `{ "text", "text_idx", "clusters" [[start,end],...],
        ///      "clusters_strings" [["..."],...] }`
        /// Char-level (not Python's word-level `text_idx` shape). `text_idx`
        /// defaults to 0 for single-Predict callers; use
        /// `CorefModel.PredictToJsonl` for batched indices.
        member this.ToJson() : string =
            this |> JsonDto.ofResult 0 |> JsonDto.serialize

    [<RequireQualifiedAccess>]
    module JsonDto =
        open System.Text.Json
        open System.Text.Json.Serialization

        /// Python-fastcoref `text_idx`-compatible char-level shape.
        /// `clusters` uses `[[start,end], ...]` arrays per Python's
        /// convention; `clusters_strings` mirrors `GetClustersAsStrings`.
        type CorefResultDto =
            { [<JsonPropertyName("text")>]
              Text: string
              [<JsonPropertyName("text_idx")>]
              TextIdx: int
              [<JsonPropertyName("clusters")>]
              Clusters: int[][][]
              [<JsonPropertyName("clusters_strings")>]
              ClustersStrings: string[][] }

        let private opts =
            let o = JsonSerializerOptions()
            o.WriteIndented <- false
            o

        let ofResult (textIdx: int) (r: CorefResult) : CorefResultDto =
            let clusters =
                r.Clusters
                |> List.map (fun c ->
                    Cluster.toList c
                    |> List.map (fun m -> [| CharIdx.value m.Span.Start; CharIdx.value m.Span.End |])
                    |> List.toArray)
                |> List.toArray

            let strings =
                r.GetClustersAsStrings()
                |> List.map List.toArray
                |> List.toArray

            { Text = r.Text
              TextIdx = textIdx
              Clusters = clusters
              ClustersStrings = strings }

        let serialize (dto: CorefResultDto) : string =
            JsonSerializer.Serialize(dto, opts)

    /// CharSpan / WordSpan-typed convenience wrappers over
    /// `Clustering.CorefLogits`. The underlying type lives in `Clustering`
    /// to break the dependency cycle (it must be referenceable by
    /// `CorefPrediction`), but `CharSpan` and `WordSpan` are Api-layer
    /// types so the typed overloads live here.
    type Clustering.CorefLogits with
        /// Look up the antecedent-score logit for the (mention, antecedent) pair
        /// identified by their character spans into the input text. Returns `None`
        /// when either span doesn't match a pruned mention, when both spans are
        /// the same (self-link), or when `Release()` has dropped the matrix.
        member this.TryGet(spanI: CharSpan, spanJ: CharSpan) : float32 option =
            let toRaw (s: CharSpan) =
                struct (CharIdx.value s.Start, CharIdx.value s.End)

            this.TryGetByCharSpan(toRaw spanI, toRaw spanJ)

        /// Look up the antecedent-score logit for the (mention, antecedent) pair
        /// identified by their *word* spans (only meaningful for results
        /// produced via `CorefModel.Predict(words)`, which populates the
        /// WordSpan -> row-index map). Same `None` semantics as the CharSpan
        /// overload, plus: returns `None` if `Predict(words)` was never the
        /// origin of this `CorefLogits` (the word-index map is empty).
        member this.TryGet(spanI: WordSpan, spanJ: WordSpan) : float32 option =
            let toRaw (s: WordSpan) =
                struct (WordIdx.value s.Start, WordIdx.value s.End)

            this.TryGetByWordSpan(toRaw spanI, toRaw spanJ)

    let private mentionOf (text: string) (offsets: TokenOffset[]) (span: TokenSpan) : Mention option =
        // `tryCharSpan` works in raw token-position ints (it indexes the
        // `TokenOffset[]` array and does arithmetic on the underlying
        // `TextSpan` int fields), so strip the measure at this boundary.
        let ts = TokenIdx.value span.Start
        let te = TokenIdx.value span.End

        TokenOffset.tryCharSpan offsets ts te
        |> Option.map (fun cs ->
            { Span =
                { Start = CharIdx.ofInt cs.Start
                  End = CharIdx.ofInt cs.End }
              Text = text.Substring(cs.Start, cs.End - cs.Start) })

    let private toResult (pred: CorefPrediction) : CorefResult =
        let clusters =
            pred.Clusters
            |> List.choose (fun c ->
                let mentions = Cluster.toList c |> List.choose (mentionOf pred.Text pred.Offsets)

                match mentions with
                | head :: rest -> Some { Head = head; Rest = rest }
                | [] -> None)

        { Text = pred.Text
          Clusters = clusters
          Logits = pred.Logits }

    [<RequireQualifiedAccess>]
    type private Backend =
        | FCoref of FCorefModel.FCorefModel
        | LingMess of LingMessModel.LingMessModel

        interface IDisposable with
            member this.Dispose() =
                match this with
                | FCoref m -> (m :> IDisposable).Dispose()
                | LingMess m -> (m :> IDisposable).Dispose()

    let private kindMismatch (asked: CorefKind) (cfg: ModelKind) : string =
        sprintf "kind mismatch: caller requested %A but config.json declares %A" asked cfg

    let private checkKind (asked: CorefKind) (cfgKind: ModelKind) =
        let ok =
            match asked, cfgKind with
            | CorefKind.FCoref, ModelKind.FCoref
            | CorefKind.LingMess, ModelKind.LingMess -> true
            | _ -> false

        if not ok then
            invalidArg "kind" (kindMismatch asked cfgKind)

    /// Coreference resolution wrapper. Loads `config.json` +
    /// `pytorch_model.bin` from `modelDir` (typically a HuggingFace snapshot
    /// such as `biu-nlp/f-coref` or `biu-nlp/lingmess-coref`), selects the
    /// backbone via `kind`, and exposes `Predict` / `PredictBatch` returning
    /// `CorefResult`s with character-level cluster spans.
    ///
    /// The model is placed in eval-mode on construction and runs on
    /// `device` (default `torch.CPU`; pass `torch.CUDA` if available).
    ///
    /// Two input shapes are supported:
    ///
    /// * `Predict(text: string)` — raw string; spans returned as character
    ///   offsets.
    /// * `Predict(words: IReadOnlyList<string>)` — pre-tokenized convenience
    ///   that joins `words` with single ASCII spaces and projects spans back
    ///   to word indices. NOT strict HuggingFace `is_split_into_words`
    ///   parity — see the overload's own XML doc.
    type CorefModel(modelDir: string, kind: CorefKind, ?device: torch.Device) =
        let dev = defaultArg device torch.CPU
        let cfg = Config.load modelDir
        do checkKind kind cfg.Kind
        let tokenizer = RobertaTokenizer(modelDir)

        let backend, report =
            match kind with
            | CorefKind.FCoref ->
                let m, r = FCorefModel.load modelDir dev
                Backend.FCoref m, r
            | CorefKind.LingMess ->
                let m, r = LingMessModel.load modelDir dev
                Backend.LingMess m, r

        do
            match backend with
            | Backend.FCoref m -> m.eval ()
            | Backend.LingMess m -> m.eval ()

        let mutable disposed = false

        /// Run coreference resolution on a single string. Returns a
        /// `CorefResult` whose `Clusters` are character offsets into `text`.
        member _.Predict(text: string) : CorefResult =
            let pred =
                match backend with
                | Backend.FCoref m -> FCorefInference.predict m tokenizer text
                | Backend.LingMess m -> LingMessInference.predict m tokenizer text

            toResult pred

        /// Run coreference resolution on multiple strings sequentially.
        /// Equivalent to mapping `Predict` over `texts`; preserves order.
        member _.PredictBatch(texts: IReadOnlyList<string>) : CorefResult[] =
            let preds =
                match backend with
                | Backend.FCoref m -> FCorefInference.predictBatch m tokenizer texts
                | Backend.LingMess m -> LingMessInference.predictBatch m tokenizer texts

            preds |> Array.map toResult

        /// Pre-tokenized input convenience: joins `words` with single ASCII
        /// spaces and runs the regular `Predict` path, then projects
        /// char-level mention spans back to word indices. Mentions that
        /// don't align on word boundaries are surfaced via
        /// `WordCorefResult.UnalignedMentions` rather than silently dropped.
        ///
        /// **NOT strict HuggingFace `is_split_into_words` parity.** This is
        /// a convenience over the existing string path; the tokenizer sees
        /// `String.concat " " words` and may behave differently from a true
        /// pre-tokenized path on punctuation/contractions. For strict
        /// semantics, call `Predict(text: string)` directly.
        ///
        /// Throws `ArgumentException` for any word that is empty / null or
        /// contains whitespace (no silent corruption via sentinel
        /// substitution).
        member this.Predict(words: IReadOnlyList<string>) : WordCorefResult =
            // 1. Validate. Strict — null / empty / any-whitespace -> ArgumentException.
            let words = words |> Seq.toArray // defensive copy

            for w in words do
                if isNull w || w.Length = 0 then
                    invalidArg "words" "words must be non-empty"

                for ch in w do
                    if Char.IsWhiteSpace ch then
                        invalidArg
                            "words"
                            (sprintf
                                "word %A contains whitespace; pre-tokenized input must not embed whitespace in a single word"
                                w)

            // 2. Join with single space; record each word's start-char offset.
            let wordStarts = Array.zeroCreate words.Length
            let mutable cursor = 0

            for i in 0 .. words.Length - 1 do
                wordStarts.[i] <- cursor
                cursor <- cursor + words.[i].Length + (if i + 1 < words.Length then 1 else 0)

            let text = String.Join(" ", words)

            // 3. Char-level predict via the existing path.
            let charResult = this.Predict(text)

            // 4. CharSpan -> WordSpan alignment. Returns `ValueSome (ws, weExclusive)`
            //    iff the char span exactly starts on `wordStarts.[ws]` AND exactly
            //    ends on `wordStarts.[we-1] + words.[we-1].Length`. Linear scan
            //    from the binary-search hit is fine — coref mentions are short
            //    (typically <= ~10 words).
            let tryCharToWord (struct (cs, ce): struct (int * int)) : struct (int * int) voption =
                let wsIdx = System.Array.BinarySearch(wordStarts, cs)

                if wsIdx < 0 then
                    ValueNone
                else
                    let mutable we = wsIdx
                    let mutable found = ValueNone

                    while found.IsNone && we < words.Length do
                        let endChar = wordStarts.[we] + words.[we].Length

                        if endChar = ce then
                            found <- ValueSome(struct (wsIdx, we + 1))
                        elif endChar > ce then
                            we <- words.Length // overshoot — abort
                        else
                            we <- we + 1

                    found

            let tryAlignMention (m: Mention) : Choice<WordMention, Mention> =
                let raw = struct (CharIdx.value m.Span.Start, CharIdx.value m.Span.End)

                match tryCharToWord raw with
                | ValueSome (struct (ws, we)) ->
                    Choice1Of2
                        { Span =
                            { Start = WordIdx.ofInt ws
                              End = WordIdx.ofInt we }
                          Text = m.Text }
                | ValueNone -> Choice2Of2 m

            // 5. Walk each char-level cluster, partition its mentions into
            //    aligned / unaligned, drop clusters with < 2 aligned mentions
            //    (preserves the `Cluster<_>` invariant), accumulate unaligned.
            let unalignedAccum = ResizeArray<Mention>()

            let alignedClusters =
                charResult.Clusters
                |> List.choose (fun cluster ->
                    let alignedRev, unalignedRev =
                        Cluster.toList cluster
                        |> List.fold
                            (fun (a, u) m ->
                                match tryAlignMention m with
                                | Choice1Of2 wm -> wm :: a, u
                                | Choice2Of2 m -> a, m :: u)
                            ([], [])

                    for m in List.rev unalignedRev do
                        unalignedAccum.Add m

                    match List.rev alignedRev with
                    | head :: (_ :: _ as rest) -> Some { Head = head; Rest = rest }
                    | _ -> None)

            // 6. Populate WordSpan -> logit-row mapping on the shared
            //    CorefLogits, so `Logits.TryGet(wsI, wsJ)` works.
            charResult.Logits.WireWordIndex(tryCharToWord)

            { Words = words
              Clusters = alignedClusters
              UnalignedMentions = List.ofSeq unalignedAccum
              Logits = charResult.Logits }

        /// Batched pre-tokenized input. Each `documents.[i]` is one
        /// pre-tokenized document; preserves order in the result.
        member this.PredictBatch(documents: IReadOnlyList<IReadOnlyList<string>>) : WordCorefResult[] =
            documents |> Seq.toArray |> Array.map (fun words -> this.Predict(words))

        /// Run `PredictBatch texts` and write one JSON object per result,
        /// one per line (JSONL), to `outputPath`. Each result's `text_idx`
        /// matches its position in `texts`. Uses `StreamWriter` so output
        /// is buffered and flushed/closed on dispose.
        member this.PredictToJsonl(texts: IReadOnlyList<string>, outputPath: string) : unit =
            let results = this.PredictBatch(texts)
            use writer = new System.IO.StreamWriter(outputPath)

            results
            |> Array.iteri (fun i r ->
                let line = r |> JsonDto.ofResult i |> JsonDto.serialize
                writer.WriteLine(line))

        /// Diagnostic partition of parameter paths into `Loaded` / `Missing`.
        /// Non-empty `Missing` means those parameters kept their random
        /// initialisation rather than being overwritten from the checkpoint,
        /// almost always indicating a state-dict key mismatch worth surfacing.
        member _.LoadReport: Config.LoadReport = report

        /// The TorchSharp device the underlying model runs on.
        member _.Device: torch.Device = dev

        /// Which backbone this instance is driving.
        member _.Kind: CorefKind = kind

        interface IDisposable with
            /// Disposes the underlying TorchSharp model, releasing native
            /// (and GPU) memory. Safe to call multiple times.
            member _.Dispose() =
                if not disposed then
                    disposed <- true
                    (backend :> IDisposable).Dispose()

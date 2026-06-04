namespace FastCoref

open System.Collections.Generic
open System.Text

open FSharp.UMX
open FastCoref.Tokenizer

module Clustering =

    [<RequireQualifiedAccess>]
    module Measures =

        [<Measure>]
        type tok

    /// Token *position* inside the encoder's flattened sequence. Distinct
    /// from `Config.TokenId` (vocabulary id) and from `Api.CharIdx`
    /// (character offset) so a swap between any two of them is a compile
    /// error.
    type TokenIdx = int<Measures.tok>

    [<RequireQualifiedAccess>]
    module TokenIdx =
        let inline ofInt (x: int) : TokenIdx = UMX.tag<Measures.tok> x
        let inline value (x: TokenIdx) : int = UMX.untag x
        let inline ofArray (xs: int[]) : TokenIdx[] = UMXArr.tag<Measures.tok> xs

    /// Inclusive `[Start..End]` token span — matches the model's mention
    /// representation (start and end are both token positions; the mention
    /// covers all tokens in between).
    [<Struct>]
    type TokenSpan = { Start: TokenIdx; End: TokenIdx }

    /// A coreference cluster. `Head + Rest` enforces the invariant that
    /// every cluster has at least two mentions without dragging in a NuGet
    /// for non-empty lists.
    type Cluster<'span> = { Head: 'span; Rest: 'span list }

    [<RequireQualifiedAccess>]
    module Cluster =
        let toList (c: Cluster<'a>) : 'a list = c.Head :: c.Rest
        let length (c: Cluster<'a>) : int = 1 + List.length c.Rest

        let map (f: 'a -> 'b) (c: Cluster<'a>) : Cluster<'b> =
            { Head = f c.Head
              Rest = List.map f c.Rest }

    /// Opaque, char-keyed accessor for the per-(mention, antecedent) coref
    /// score matrix produced by inference. The matrix is computed once during
    /// `predict` / `predictBatch` and held by reference; `Release()` drops it
    /// so subsequent `TryGet` returns `None` (mirroring Python `release_logits`).
    ///
    /// The CharSpan-to-row-index dictionary is precomputed during inference by
    /// iterating the pruned `TokenSpan[]` and calling `TokenOffset.tryCharSpan`
    /// — there is no runtime CharSpan -> TokenSpan inversion.
    ///
    /// Note: this type is referenced from `module FastCoref.Api` which defines
    /// `CharSpan`; to avoid a forward reference, this module's TryGet takes
    /// raw `(int * int)` tuples and the Api layer wraps with the typed
    /// CharSpan/WordSpan overloads.
    type CorefLogits internal (charSpanToIndex: IReadOnlyDictionary<struct (int * int), int>, matrix: float32[,]) =
        let mutable matrixRef: float32[,] voption = ValueSome matrix
        let mutable charIndex = charSpanToIndex
        // Word-keyed dictionary is populated lazily by Api when WordCorefResult
        // is constructed; default empty so the type is usable for CharSpan-only paths.
        let mutable wordIndex: IReadOnlyDictionary<struct (int * int), int> =
            (System.Collections.Generic.Dictionary<struct (int * int), int>()) :> IReadOnlyDictionary<_, _>

        /// `(spanI_start, spanI_end)`, `(spanJ_start, spanJ_end)` -> logit (lower triangle).
        /// Self-link (i==j) -> None. Either span unknown -> None. Released -> None.
        member _.TryGetByCharSpan(spanI: struct (int * int), spanJ: struct (int * int)) : float32 option =
            match matrixRef with
            | ValueNone -> None
            | ValueSome m ->
                match charIndex.TryGetValue spanI, charIndex.TryGetValue spanJ with
                | (true, i), (true, j) when i <> j ->
                    let lo = min i j
                    let hi = max i j

                    if hi < Array2D.length1 m && lo < Array2D.length2 m then
                        Some m.[hi, lo]
                    else
                        None
                | _ -> None

        member _.TryGetByWordSpan(spanI: struct (int * int), spanJ: struct (int * int)) : float32 option =
            match matrixRef with
            | ValueNone -> None
            | ValueSome m ->
                match wordIndex.TryGetValue spanI, wordIndex.TryGetValue spanJ with
                | (true, i), (true, j) when i <> j ->
                    let lo = min i j
                    let hi = max i j

                    if hi < Array2D.length1 m && lo < Array2D.length2 m then
                        Some m.[hi, lo]
                    else
                        None
                | _ -> None

        /// Drop the matrix reference. Subsequent TryGet* returns None.
        member _.Release() = matrixRef <- ValueNone

        /// True iff Release() has not been called.
        member _.HasMatrix: bool = matrixRef.IsSome

        /// Walk the CharSpan -> row-index dictionary, mapping each CharSpan to a
        /// WordSpan via the supplied `tryMap` function; populate the WordSpan
        /// dictionary in-place. No-op if `Release()` already called.
        ///
        /// Called by the Api layer at the end of `Predict(words)` once it has
        /// the word-boundary information needed to translate each retained
        /// CharSpan into a WordSpan. CharSpans for which `tryMap` returns
        /// `ValueNone` (mention doesn't align on word boundaries) are silently
        /// omitted — `TryGetByWordSpan` will return `None` for them.
        member internal _.WireWordIndex(tryMap: struct (int * int) -> struct (int * int) voption) =
            match matrixRef with
            | ValueNone -> ()
            | ValueSome _ ->
                let m = System.Collections.Generic.Dictionary<struct (int * int), int>()

                for KeyValue(charSpan, idx) in charIndex do
                    match tryMap charSpan with
                    | ValueSome wordSpan -> m.[wordSpan] <- idx
                    | ValueNone -> ()

                wordIndex <- m :> IReadOnlyDictionary<_, _>

    [<RequireQualifiedAccess>]
    module CorefLogits =
        /// Inert sentinel for synthetic test fixtures. Always returns None.
        let empty: CorefLogits =
            let emptyDict =
                (System.Collections.Generic.Dictionary<struct (int * int), int>()) :> IReadOnlyDictionary<_, _>

            CorefLogits(emptyDict, Array2D.zeroCreate<float32> 0 0)

    /// Token-domain result of an end-to-end coref inference pipeline.
    /// `Clusters` carry the token-level cluster decoding; `TokenSpans` /
    /// `SpanToIndex` / `FinalLogits` expose the per-pruned-mention matrix
    /// so callers (`Api.CorefModel`) can map back to character offsets or
    /// recover pair logits. `Offsets` is the encoder's per-token offset
    /// map — owned here verbatim so downstream code does not need to
    /// re-tokenize. `Logits` is the opaque, char-keyed accessor over the
    /// same `FinalLogits` matrix (the user-facing way to read pair scores).
    type CorefPrediction =
        { Text: string
          Clusters: Cluster<TokenSpan> list
          TokenSpans: TokenSpan[]
          Offsets: TokenOffset[]
          FinalLogits: float32[,]
          SpanToIndex: IReadOnlyDictionary<TokenSpan, int>
          Logits: CorefLogits }

    /// Inputs to `extractClusters`. `MentionStartIds` / `MentionEndIds` are
    /// the inclusive token-index endpoints of each pruned mention (length
    /// `k`); `FinalLogits` is the `[k, k+1]` antecedent-score matrix whose
    /// last column is the "no antecedent" / null choice.
    type ClusteringInput =
        { MentionStartIds: TokenIdx[]
          MentionEndIds: TokenIdx[]
          FinalLogits: float32[,] }

    let private mkSpan (s: TokenIdx) (e: TokenIdx) : TokenSpan = { Start = s; End = e }

    let private spanCmp (a: TokenSpan) (b: TokenSpan) =
        let c = compare a.Start b.Start
        if c <> 0 then c else compare a.End b.End

    let extractClusters (input: ClusteringInput) : Cluster<TokenSpan> list =
        let mentionStartIds = input.MentionStartIds
        let mentionEndIds = input.MentionEndIds
        let finalLogits = input.FinalLogits
        let k = mentionStartIds.Length

        if k = 0 then
            []
        else
            let m2a = ResizeArray<TokenSpan * TokenSpan>()

            for i in 0 .. k - 1 do
                // Antecedent must be a strictly EARLIER mention (j < i) or the
                // null choice at column k. The Python f-coref / LingMess models
                // enforce this by masking logits[i, j>=i] to -inf before
                // softmax; we defensively restrict the argmax search here so
                // arbitrary inputs (e.g. property tests) can never produce a
                // self-link (singleton cluster) or a forward link (mention
                // appearing in two clusters).
                let mutable best = k

                for j in 0 .. i - 1 do
                    if finalLogits.[i, j] > finalLogits.[i, best] then
                        best <- j

                if best < k then
                    let mention = mkSpan mentionStartIds.[i] mentionEndIds.[i]
                    let antecedent = mkSpan mentionStartIds.[best] mentionEndIds.[best]
                    m2a.Add(mention, antecedent)

            let clusters = ResizeArray<ResizeArray<TokenSpan>>()
            let mentionToCluster = Dictionary<TokenSpan, int>()

            for (mention, antecedent) in m2a do
                match mentionToCluster.TryGetValue antecedent with
                | true, idx ->
                    clusters.[idx].Add(mention)
                    mentionToCluster.[mention] <- idx
                | false, _ ->
                    let idx = clusters.Count
                    let c = ResizeArray<TokenSpan>()
                    c.Add(antecedent)
                    c.Add(mention)
                    clusters.Add(c)
                    mentionToCluster.[mention] <- idx
                    mentionToCluster.[antecedent] <- idx

            clusters
            |> Seq.choose (fun c ->
                let sorted = c |> Seq.distinct |> Seq.sortWith spanCmp |> List.ofSeq

                match sorted with
                | [] -> None
                | head :: rest -> Some { Head = head; Rest = rest })
            |> List.ofSeq

    let prettyPrint (clusters: Cluster<TokenSpan> list) : string =
        let sb = StringBuilder()
        sb.AppendFormat("{0} cluster(s):", clusters.Length) |> ignore

        clusters
        |> List.iteri (fun i c ->
            sb.AppendLine() |> ignore
            sb.AppendFormat("  [{0}]", i) |> ignore

            for span in Cluster.toList c do
                let s = TokenIdx.value span.Start
                let e = TokenIdx.value span.End
                sb.AppendFormat(" ({0},{1})", s, e) |> ignore)

        sb.ToString()

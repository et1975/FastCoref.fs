module FastCoref.Tests.CorefLogitsTests

// Tests for `FastCoref.Clustering.CorefLogits` and the
// `Api.CorefResult.Logits` field surfaced through `Api.CorefModel`.
//
// Two groups:
//   * Pure tests exercise the synthetic `CorefLogits.empty` sentinel and a
//     hand-built fixture using the internal constructor (the test project
//     has `InternalsVisibleTo`). They run unconditionally.
//   * Gated tests load a real `f-coref` checkpoint and assert round-trip
//     properties on the predicted matrix. They are gated on the
//     `FASTCOREF_MODELS_DIR` env var via the same idiom as `FCorefTests.fs`
//     (silent vacuous pass when unset / missing).
//
// Word-level (`WordSpan`) overload tests are at the bottom of this file,
// under the `// ========== WordSpan tests ==========` banner. They exercise
// the `WireWordIndex` wiring plus the typed `TryGet(WordSpan, WordSpan)`
// overload defined in `FastCoref.Api`.

open System
open System.Collections.Generic
open System.IO
open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Clustering
open FastCoref.Api

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

let private charSpan (startIdx: int) (endIdx: int) : CharSpan =
    { Start = CharIdx.ofInt startIdx
      End = CharIdx.ofInt endIdx }

/// Build a fresh non-empty `CorefLogits` with two CharSpan rows and a 2x1
/// lower-triangle matrix. Row 0 -> charSpan (0,5); row 1 -> charSpan (10,15).
/// `matrix.[1, 0] = 10.0f`, `matrix.[0, 0] = 0.0f`.
let private mkFixture () : CorefLogits * float32[,] =
    let dict = Dictionary<struct (int * int), int>()
    dict.[struct (0, 5)] <- 0
    dict.[struct (10, 15)] <- 1
    let matrix = Array2D.init 2 1 (fun i j -> float32 (i * 10 + j))
    let logits = CorefLogits(dict :> IReadOnlyDictionary<_, _>, matrix)
    logits, matrix

// ---------------------------------------------------------------------------
// Gating — mirrors FCorefTests.fs
// ---------------------------------------------------------------------------

let private modelsDir () : string option =
    match Environment.GetEnvironmentVariable "FASTCOREF_MODELS_DIR" with
    | null
    | "" -> None
    | path -> Some path

let private fcorefDir () : string option =
    modelsDir ()
    |> Option.map (fun root -> Path.Combine(root, "f-coref"))
    |> Option.filter (fun p -> File.Exists(Path.Combine(p, Utils.HfFiles.Config)))

/// Text known to produce at least one coref cluster (Alice / She).
let private sampleText = "Alice walked into the room. She sat down."

// ---------------------------------------------------------------------------
// Pure tests
// ---------------------------------------------------------------------------

[<Fact>]
let ``CorefLogits.empty always returns None on TryGet`` () =
    let s1 = charSpan 0 5
    let s2 = charSpan 10 15
    test <@ CorefLogits.empty.TryGet(s1, s2) = None @>
    // Symmetric query: same answer.
    test <@ CorefLogits.empty.TryGet(s2, s1) = None @>
    // Self-link: still None.
    test <@ CorefLogits.empty.TryGet(s1, s1) = None @>

[<Fact>]
let ``CorefLogits.empty TryGetByCharSpan returns None`` () =
    test <@ CorefLogits.empty.TryGetByCharSpan(struct (0, 5), struct (10, 15)) = None @>

[<Fact>]
let ``CorefLogits.empty HasMatrix is true initially`` () =
    // `empty` is constructed with a real (0x0) matrix, not released — pin
    // that current behaviour. `Release()` is what flips `HasMatrix` to
    // false, not "matrix is empty".
    test <@ CorefLogits.empty.HasMatrix = true @>

[<Fact>]
let ``Release makes HasMatrix false and TryGet None`` () =
    let logits, _ = mkFixture ()

    // Before Release: known pair is Some.
    test <@ logits.HasMatrix = true @>
    test <@ logits.TryGetByCharSpan(struct (0, 5), struct (10, 15)) |> Option.isSome @>

    logits.Release()

    // After Release: HasMatrix false and the same pair returns None.
    test <@ logits.HasMatrix = false @>
    test <@ logits.TryGetByCharSpan(struct (0, 5), struct (10, 15)) = None @>

[<Fact>]
let ``TryGetByCharSpan respects lower-triangle and self-link rule`` () =
    let logits, matrix = mkFixture ()
    let s0 = struct (0, 5)
    let s1 = struct (10, 15)
    let unknown = struct (999, 1000)

    // Self-link forbidden.
    test <@ logits.TryGetByCharSpan(s0, s0) = None @>

    // Lower-triangle: (row 0, row 1) -> m.[max,min] = m.[1, 0].
    test <@ logits.TryGetByCharSpan(s0, s1) = Some matrix.[1, 0] @>

    // Symmetric query produces the same lower-triangle entry.
    test <@ logits.TryGetByCharSpan(s1, s0) = Some matrix.[1, 0] @>

    // Unknown span on either side -> None.
    test <@ logits.TryGetByCharSpan(s0, unknown) = None @>
    test <@ logits.TryGetByCharSpan(unknown, s0) = None @>

// ---------------------------------------------------------------------------
// Gated tests — real f-coref load
// ---------------------------------------------------------------------------

[<Fact>]
let ``Real prediction: every (head, mention) pair has Some logit`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let result = coref.Predict sampleText

        // Sanity: the sample text should produce at least one cluster, else
        // there is nothing to test. We assert this so a degenerate prediction
        // does not silently mask the round-trip check.
        test <@ result.Clusters.Length >= 1 @>

        for cluster in result.Clusters do
            for mention in cluster.Rest do
                let logit = result.Logits.TryGet(cluster.Head.Span, mention.Span)
                test <@ Option.isSome logit @>

[<Fact>]
let ``Real prediction: shifted-by-one CharSpan returns None`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let result = coref.Predict sampleText

        test <@ result.Clusters.Length >= 1 @>

        let headSpan = result.Clusters.Head.Head.Span

        let shifted: CharSpan =
            { Start = CharIdx.ofInt (CharIdx.value headSpan.Start + 1)
              End = CharIdx.ofInt (CharIdx.value headSpan.End + 1) }

        // Defensive: a span shifted by one character is overwhelmingly
        // unlikely to align with any pruned mention's CharSpan key, since
        // pruning works on token-level spans and char offsets land on token
        // boundaries.
        test <@ result.Logits.TryGet(headSpan, shifted) = None @>

[<Fact>]
let ``Real prediction: Release frees TryGet`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let result = coref.Predict sampleText

        test <@ result.Clusters.Length >= 1 @>

        // Capture the (head, rest-mention) pairs that returned Some before
        // Release; assert each one flips to None after.
        let pairs =
            [ for cluster in result.Clusters do
                  for mention in cluster.Rest do
                      let before = result.Logits.TryGet(cluster.Head.Span, mention.Span)

                      if Option.isSome before then
                          yield cluster.Head.Span, mention.Span ]

        // The previous gated test pins this, but assert locally so a failure
        // here points the finger at Release semantics, not at the upstream
        // round-trip property.
        test <@ pairs.Length >= 1 @>

        result.Logits.Release()

        for head, mention in pairs do
            test <@ result.Logits.TryGet(head, mention) = None @>

        test <@ result.Logits.HasMatrix = false @>

// ---------------------------------------------------------------------------
// ========== WordSpan tests ==========
// ---------------------------------------------------------------------------
//
// Cover the `TryGet(WordSpan, WordSpan)` overload (Api.fs) and the
// `WireWordIndex` wiring (Clustering.fs, internal — visible here via
// `InternalsVisibleTo`). The fixture from `mkFixture` is reused; note that
// `mkFixture` does *not* call `WireWordIndex`, so every test below that
// exercises the WordSpan path must wire it explicitly.

let private wordSpan (s: int) (e: int) : WordSpan =
    { Start = WordIdx.ofInt s
      End = WordIdx.ofInt e }

[<Fact>]
let ``TryGet(WordSpan,WordSpan) is None before WireWordIndex`` () =
    let logits, _ = mkFixture ()

    // `wordIndex` defaults to an empty dictionary; the CharSpan path still
    // works (pinned by an earlier test) but the WordSpan path must miss.
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 1 2) = None @>
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 2 3) = None @>

[<Fact>]
let ``WireWordIndex wires every CharSpan whose translator returns ValueSome`` () =
    let logits, matrix = mkFixture ()

    // Translator: divide both endpoints by 5 — charSpan (0,5)  -> wordSpan (0,1)
    //                                         charSpan (10,15) -> wordSpan (2,3)
    logits.WireWordIndex(fun (struct (cs, ce)) -> ValueSome(struct (cs / 5, ce / 5)))

    // Lower-triangle lookup: (row 0, row 1) -> m.[1, 0] = 10.0f.
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = Some matrix.[1, 0] @>
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 2 3) = Some matrix.[1, 0] @>

    // CharSpan path is unaffected by WireWordIndex.
    test <@ logits.TryGetByCharSpan(struct (0, 5), struct (10, 15)) = Some matrix.[1, 0] @>

[<Fact>]
let ``WireWordIndex skips CharSpans whose translator returns ValueNone`` () =
    let logits, _ = mkFixture ()

    // Drop charSpan (10,15) from the WordSpan index; keep (0,5) -> (0,1).
    logits.WireWordIndex(fun (struct (cs, ce)) ->
        if cs = 10 then ValueNone
        else ValueSome(struct (cs / 5, ce / 5)))

    // Self-link on the surviving span — still None per the i==j rule.
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 0 1) = None @>

    // Partner span was dropped (translator returned ValueNone) — its
    // WordSpan key is absent from `wordIndex`, so the cross-lookup misses
    // even though the underlying matrix row exists.
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = None @>
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 2 3) = None @>

[<Fact>]
let ``TryGet(WordSpan,WordSpan) self-link returns None`` () =
    let logits, _ = mkFixture ()

    logits.WireWordIndex(fun (struct (cs, ce)) -> ValueSome(struct (cs / 5, ce / 5)))

    // Both endpoints resolve to the same row -> i==j -> None.
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 0 1) = None @>
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 2 3) = None @>

[<Fact>]
let ``Release invalidates WordSpan lookup`` () =
    let logits, matrix = mkFixture ()

    logits.WireWordIndex(fun (struct (cs, ce)) -> ValueSome(struct (cs / 5, ce / 5)))

    // Sanity: the wiring took.
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = Some matrix.[1, 0] @>

    logits.Release()

    // Same query now returns None — `matrixRef` short-circuits both paths.
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = None @>
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 2 3) = None @>
    test <@ logits.HasMatrix = false @>

[<Fact>]
let ``WireWordIndex replaces (not merges) word-index on second call`` () =
    let logits, matrix = mkFixture ()

    // First wiring: charSpan (0,5)  -> wordSpan (0,1)
    //               charSpan (10,15) -> wordSpan (2,3)
    logits.WireWordIndex(fun (struct (cs, ce)) -> ValueSome(struct (cs / 5, ce / 5)))
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = Some matrix.[1, 0] @>

    // Second wiring: completely different word ranges.
    //   charSpan (0,5)  -> wordSpan (100, 200)
    //   charSpan (10,15) -> wordSpan (300, 400)
    logits.WireWordIndex(fun (struct (cs, _)) ->
        if cs = 0 then ValueSome(struct (100, 200))
        else ValueSome(struct (300, 400)))

    // Self-link on the new mapping — same row twice -> None.
    test <@ logits.TryGet(wordSpan 100 200, wordSpan 100 200) = None @>

    // Old keys are gone (replacement, not merge).
    test <@ logits.TryGet(wordSpan 0 1, wordSpan 2 3) = None @>
    test <@ logits.TryGet(wordSpan 2 3, wordSpan 0 1) = None @>

    // New cross-pair resolves to the lower-triangle entry.
    test <@ logits.TryGet(wordSpan 100 200, wordSpan 300 400) = Some matrix.[1, 0] @>
    test <@ logits.TryGet(wordSpan 300 400, wordSpan 100 200) = Some matrix.[1, 0] @>

// ---------------------------------------------------------------------------
// Gated WordSpan tests — real f-coref load
// ---------------------------------------------------------------------------

[<Fact>]
let ``Real prediction via Predict(words): every (head, mention) WordSpan pair has Some logit`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)

        let words =
            [| "Alice"; "walked"; "home"; "."; "She"; "sat"; "down"; "." |]

        let result = coref.Predict(words)

        // On this short text it's possible the model returns no clusters; if
        // so, the round-trip property is vacuous — skip rather than fail.
        if result.Clusters.Length >= 1 then
            for cluster in result.Clusters do
                for mention in cluster.Rest do
                    let logit = result.Logits.TryGet(cluster.Head.Span, mention.Span)
                    test <@ Option.isSome logit @>

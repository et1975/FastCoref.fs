module FastCoref.Tests.WordInputTests

// Tests for `FastCoref.Api.CorefModel.Predict(words: IReadOnlyList<string>)`
// and its `PredictBatch` sibling — added in v2 as a pre-tokenized convenience
// over the existing char-string `Predict` path.
//
// Every test gates on `FASTCOREF_MODELS_DIR` and silently vacuous-passes when
// unset, matching the idiom established by `FCorefTests.fs` and
// `CorefLogitsTests.fs`. We can't exercise the validation phase without a
// `CorefModel` instance (its constructor needs the model dir), so even the
// "throws on bad input" facts are gated — there is no separate "pure" group.
//
// Sibling agent `word-logits-tests-4b` covers the WordSpan side of
// `CorefLogits.TryGet` from the logits angle; the overlap here (test 9) is
// intentional and tests the WordSpan wiring from the result-construction
// angle.

open System
open System.Collections.Generic
open System.IO
open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Clustering
open FastCoref.Api

// ---------------------------------------------------------------------------
// Gating — mirrors FCorefTests.fs / CorefLogitsTests.fs
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

// ---------------------------------------------------------------------------
// 1. Round-trip parity with the char `Predict`
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict(words) matches Predict(text) in cluster count and mention text`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        // Whitespace-separated, no contractions — every char-level mention
        // should align on a word boundary, so cluster shape stays identical
        // and `UnalignedMentions` is empty.
        let text = "Alice walked home . She sat down ."
        let words = text.Split(' ')
        let charResult = coref.Predict text
        let wordResult = coref.Predict words

        test <@ wordResult.Words = words @>
        test <@ wordResult.UnalignedMentions = [] @>
        test <@ wordResult.Clusters.Length = charResult.Clusters.Length @>

        for cw, cc in List.zip wordResult.Clusters charResult.Clusters do
            test <@ Cluster.length cw = Cluster.length cc @>

            let wmTexts = Cluster.toList cw |> List.map (fun m -> m.Text)
            let cmTexts = Cluster.toList cc |> List.map (fun m -> m.Text)
            test <@ wmTexts = cmTexts @>

// ---------------------------------------------------------------------------
// 2. WordSpan reflects word indices for known mentions
// ---------------------------------------------------------------------------

[<Fact>]
let ``WordSpan reflects word indices for known mentions`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        // words.[0] = "Alice"  -> WordSpan { Start=0; End=1 }
        // words.[4] = "She"    -> WordSpan { Start=4; End=5 }
        let words = [| "Alice"; "walked"; "home"; "."; "She"; "sat"; "down"; "." |]
        let result = coref.Predict words

        let allMentions = result.Clusters |> List.collect Cluster.toList
        let aliceMention = allMentions |> List.tryFind (fun m -> m.Text = "Alice")
        let sheMention = allMentions |> List.tryFind (fun m -> m.Text = "She")

        // Only assert when the model identified them — model-quality is not
        // part of the word-projection contract being tested here.
        match aliceMention with
        | Some m ->
            test <@ WordIdx.value m.Span.Start = 0 @>
            test <@ WordIdx.value m.Span.End = 1 @>
        | None -> ()

        match sheMention with
        | Some m ->
            test <@ WordIdx.value m.Span.Start = 4 @>
            test <@ WordIdx.value m.Span.End = 5 @>
        | None -> ()

// ---------------------------------------------------------------------------
// 3. UnalignedMentions — defensive shape check
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict with contractions: result well-formed, any unaligned mention does not land on word boundaries`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        // "doesn't" / "." may produce a char span the GPT-2 BPE splits inside
        // the apostrophe — we don't know in advance which mentions, if any,
        // will fail to align. The assertion below is conditional on
        // unalignment actually happening; empty UnalignedMentions is a valid
        // outcome.
        let words = [| "Alice"; "doesn't"; "like"; "Bob"; "."; "She"; "left"; "." |]
        let result = coref.Predict words

        test <@ result.Words = words @>

        // Cluster<_> invariant: every retained cluster has >= 2 mentions.
        for c in result.Clusters do
            test <@ Cluster.length c >= 2 @>

        // Re-derive `wordStarts` exactly as the impl does so we can independently
        // verify that each unaligned mention's char span is NOT word-aligned.
        let wordStarts = Array.zeroCreate words.Length
        let mutable cursor = 0

        for i in 0 .. words.Length - 1 do
            wordStarts.[i] <- cursor
            cursor <- cursor + words.[i].Length + (if i + 1 < words.Length then 1 else 0)

        let wordEnds =
            wordStarts
            |> Array.mapi (fun i s -> s + words.[i].Length)

        for m in result.UnalignedMentions do
            let cs = CharIdx.value m.Span.Start
            let ce = CharIdx.value m.Span.End
            let startsOnBoundary = Array.contains cs wordStarts
            let endsOnBoundary = Array.contains ce wordEnds
            // Genuine unalignment: either start or end (or both) is off-boundary.
            test <@ not (startsOnBoundary && endsOnBoundary) @>

// ---------------------------------------------------------------------------
// 4. Validation — empty word
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict throws ArgumentException on empty individual word`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let words = [| "Alice"; ""; "Bob" |]

        let ex =
            Assert.Throws<ArgumentException>(fun () -> coref.Predict words |> ignore)

        test <@ ex.Message.Contains "non-empty" @>

// ---------------------------------------------------------------------------
// 5. Validation — whitespace inside a word (space AND tab)
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict throws ArgumentException on space inside word`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let words = [| "Alice"; "wal ked"; "Bob" |]

        let ex =
            Assert.Throws<ArgumentException>(fun () -> coref.Predict words |> ignore)

        test <@ ex.Message.Contains "whitespace" @>

[<Fact>]
let ``Predict throws ArgumentException on tab inside word`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let words = [| "Alice"; "wal\tked"; "Bob" |]

        let ex =
            Assert.Throws<ArgumentException>(fun () -> coref.Predict words |> ignore)

        test <@ ex.Message.Contains "whitespace" @>

// ---------------------------------------------------------------------------
// 6. Validation — null word (caught by `isNull w || w.Length = 0`)
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict throws ArgumentException on null word`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        // Build via mutation so the F# literal `[| "Alice"; null; "Bob" |]`
        // doesn't trip the F# 9 nullable-reference type warnings on `null`.
        let words: string[] = Array.zeroCreate 3
        words.[0] <- "Alice"
        words.[1] <- null
        words.[2] <- "Bob"

        Assert.Throws<ArgumentException>(fun () -> coref.Predict words |> ignore)
        |> ignore

// ---------------------------------------------------------------------------
// 7. Empty word LIST is allowed (validation only rejects empty individual
//    words, per src/Api.fs lines 371–381). Returns an empty result.
// ---------------------------------------------------------------------------

[<Fact>]
let ``Predict with empty word list returns empty result`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        // Per src/Api.fs lines 367–391: validation iterates `words` and only
        // throws on per-word problems; an empty top-level array trivially
        // passes that loop, joins to "", and runs `Predict("")` downstream.
        // The expected outcome is a structurally empty WordCorefResult.
        let result = coref.Predict([||]: string[])

        test <@ result.Words = [||] @>
        test <@ result.Clusters = [] @>
        test <@ result.UnalignedMentions = [] @>

// ---------------------------------------------------------------------------
// 8. PredictBatch parity — order preserved, Words round-trip
// ---------------------------------------------------------------------------

[<Fact>]
let ``PredictBatch returns same Words arrays as Predict in order`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let words1 = [| "Alice"; "walked"; "home"; "."; "She"; "sat"; "down"; "." |]
        let words2 = [| "Bob"; "was"; "happy"; "."; "He"; "smiled"; "." |]

        let docs: IReadOnlyList<IReadOnlyList<string>> =
            [| (words1 :> IReadOnlyList<string>); (words2 :> IReadOnlyList<string>) |]

        let batch = coref.PredictBatch docs

        test <@ batch.Length = 2 @>
        // Compare on the cheap, deterministic field — full WordCorefResult
        // equality would cover Clusters too but isn't required by the
        // batch-ordering contract.
        test <@ batch.[0].Words = words1 @>
        test <@ batch.[1].Words = words2 @>

// ---------------------------------------------------------------------------
// 9. Logits survives the word path: HasMatrix true AND WordSpan lookup works.
//    Validates `WireWordIndex(tryCharToWord)` is called inside Predict(words)
//    (src/Api.fs line 460).
// ---------------------------------------------------------------------------

[<Fact>]
let ``Logits HasMatrix true and TryGet by WordSpan returns Some for head-rest pair`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
        let words = [| "Alice"; "walked"; "into"; "the"; "room"; "."; "She"; "sat"; "down"; "." |]
        let result = coref.Predict words

        test <@ result.Logits.HasMatrix = true @>
        test <@ result.Clusters.Length >= 1 @>

        // Pick the first cluster's (head, first-rest) pair and verify the
        // WordSpan-keyed lookup returns Some. If the WordSpan -> row-index
        // map were not wired, this would be None.
        let c = result.Clusters.Head
        // Explicit annotation: WordSpan and CharSpan are structurally identical
        // — pin the overload F# resolves against.
        let headSpan: WordSpan = c.Head.Span
        let restSpan: WordSpan = c.Rest.Head.Span
        let logit = result.Logits.TryGet(headSpan, restSpan)

        test <@ Option.isSome logit @>

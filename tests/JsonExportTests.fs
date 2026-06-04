module FastCoref.Tests.JsonExportTests

// Pure tests for the v2 JSON export surface on `Api.CorefResult`:
//   * `CorefResult.ToJson() : string` — single-line JSON, char-level shape.
//
// All tests synthesize `CorefResult` values via `CorefLogits.empty` — no
// model load, no env-var gating. The `PredictToJsonl` integration is a thin
// wrapper around `PredictBatch` + per-line `JsonDto.ofResult i + serialize`;
// since it requires a live model, that contract belongs alongside the gated
// `FCorefTests.fs` / `LingMessTests.fs` smoke tests. This file pins the
// per-result JSON shape that `PredictToJsonl` writes one of per line.

open System.Text.Json
open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Api
open FastCoref.Clustering

// ---- Helpers --------------------------------------------------------------

let private mkSpan (s: int) (e: int) : CharSpan =
    { Start = CharIdx.ofInt s
      End = CharIdx.ofInt e }

let private mkMention (s: int) (e: int) (text: string) : Mention =
    { Span = mkSpan s e; Text = text }

let private mkCluster (head: Mention) (rest: Mention list) : Cluster<Mention> =
    { Head = head; Rest = rest }

let private mkResult (text: string) (clusters: Cluster<Mention> list) : CorefResult =
    { Text = text
      Clusters = clusters
      Logits = CorefLogits.empty }

// `JsonElement` is a struct — Unquote quotations cannot capture struct
// locals (FS3155: "may not involve … taking the address of a captured local
// variable"). So every helper here returns reference / scalar values
// extracted from the JSON *before* the test quotation runs.

let private textOf (json: string) : string =
    use doc = JsonDocument.Parse json
    doc.RootElement.GetProperty("text").GetString()

let private textIdxOf (json: string) : int =
    use doc = JsonDocument.Parse json
    doc.RootElement.GetProperty("text_idx").GetInt32()

/// Read `clusters` as `int[][][]` (cluster -> mention -> [start; end]).
let private clustersOf (json: string) : int[][][] =
    use doc = JsonDocument.Parse json

    [| for cluster in doc.RootElement.GetProperty("clusters").EnumerateArray() ->
           [| for mention in cluster.EnumerateArray() ->
                  [| for pt in mention.EnumerateArray() -> pt.GetInt32() |] |] |]

let private clustersStringsOf (json: string) : string[][] =
    use doc = JsonDocument.Parse json

    [| for cluster in doc.RootElement.GetProperty("clusters_strings").EnumerateArray() ->
           [| for s in cluster.EnumerateArray() -> s.GetString() |] |]

// ---- Tests ----------------------------------------------------------------

[<Fact>]
let ``ToJson: empty clusters produces empty arrays and echoes text`` () =
    let json = (mkResult "Hello world" []).ToJson()

    test <@ textOf json = "Hello world" @>
    test <@ textIdxOf json = 0 @>
    test <@ clustersOf json = [||] @>
    test <@ clustersStringsOf json = [||] @>

[<Fact>]
let ``ToJson: single cluster with two mentions round-trips spans and strings`` () =
    // "Alice. She."
    //  0    5  7  10
    let result =
        mkResult "Alice. She." [ mkCluster (mkMention 0 5 "Alice") [ mkMention 7 10 "She" ] ]

    let json = result.ToJson()

    test <@ clustersOf json = [| [| [| 0; 5 |]; [| 7; 10 |] |] |] @>
    test <@ clustersStringsOf json = [| [| "Alice"; "She" |] |] @>

[<Fact>]
let ``ToJson: multiple clusters preserve construction order`` () =
    // "John saw Mary. He waved at her."
    //  0    5   9    14 15  18    27 30
    let result =
        mkResult
            "John saw Mary. He waved at her."
            [ mkCluster (mkMention 0 4 "John") [ mkMention 15 17 "He" ]
              mkCluster (mkMention 9 13 "Mary") [ mkMention 27 30 "her" ] ]

    let json = result.ToJson()
    let clusters = clustersOf json
    let strings = clustersStringsOf json

    test <@ clusters.Length = 2 @>
    test <@ clusters.[0] = [| [| 0; 4 |]; [| 15; 17 |] |] @>
    test <@ clusters.[1] = [| [| 9; 13 |]; [| 27; 30 |] |] @>
    test <@ strings = [| [| "John"; "He" |]; [| "Mary"; "her" |] |] @>

[<Fact>]
let ``ToJson: output is single-line (no newlines, no carriage returns)`` () =
    let result =
        mkResult
            "John saw Mary. He waved at her."
            [ mkCluster (mkMention 0 4 "John") [ mkMention 15 17 "He" ]
              mkCluster (mkMention 9 13 "Mary") [ mkMention 27 30 "her" ] ]

    let json = result.ToJson()
    test <@ json.Contains('\n') = false @>
    test <@ json.Contains('\r') = false @>

[<Fact>]
let ``ToJson: text_idx defaults to 0 for single-Predict callers`` () =
    // Two distinct results — both produced via `ToJson()` (not the JSONL
    // path) — should each carry `text_idx = 0`. Per-line indexing under
    // `CorefModel.PredictToJsonl` is exercised in the gated FCoref /
    // LingMess smoke tests where a real model is loadable.
    let a = (mkResult "first" []).ToJson()
    let b = (mkResult "second" []).ToJson()

    test <@ textIdxOf a = 0 @>
    test <@ textIdxOf b = 0 @>

[<Fact>]
let ``ToJson: result is parseable by JsonDocument and round-trips text`` () =
    let result =
        mkResult "Alice. She." [ mkCluster (mkMention 0 5 "Alice") [ mkMention 7 10 "She" ] ]

    let json = result.ToJson()
    // `textOf` parses internally — if `JsonDocument.Parse` throws, this fails.
    test <@ textOf json = result.Text @>

[<Fact>]
let ``ToJson: clusters_strings uses Mention.Text verbatim, not a re-extraction`` () =
    // The `Mention.Text` field is the canonical surface form — JSON must
    // echo it, NOT re-slice `text.Substring(span.Start, span.End - ...)`.
    // Construct a deliberate mismatch: span points at lowercase "aliased"
    // but Mention.Text is "ALIASED".
    let result =
        mkResult
            "aliased and others."
            [ mkCluster (mkMention 0 7 "ALIASED") [ mkMention 12 18 "OTHERS" ] ]

    let strings = clustersStringsOf (result.ToJson())
    test <@ strings = [| [| "ALIASED"; "OTHERS" |] |] @>

[<Fact>]
let ``ToJson: half-open span [0,5) serialises as [0, 5]`` () =
    // Pin the half-open `[Start..End)` convention — `End` is exclusive
    // and surfaces verbatim into JSON. Off-by-one in either direction
    // would silently break Python `fastcoref` compatibility.
    let result =
        mkResult "Alice." [ mkCluster (mkMention 0 5 "Alice") [ mkMention 0 5 "Alice" ] ]

    let clusters = clustersOf (result.ToJson())

    test <@ clusters.Length = 1 @>
    test <@ clusters.[0].[0] = [| 0; 5 |] @>
    test <@ clusters.[0].[1] = [| 0; 5 |] @>

module FastCoref.Tests.LingMessTests

// Smoke tests for `FastCoref.LingMessInference` + `FastCoref.LingMessModel`,
// driven through `FastCoref.Api.CorefModel` for the char-domain assertions.
//
// The TorchSharp runtime currently HANGS on macOS 12 + Intel libtorch builds,
// so any test that actually loads weights or runs a forward pass is gated on
// the `FASTCOREF_MODELS_DIR` environment variable. The expected layout is:
//
//   $FASTCOREF_MODELS_DIR/
//     lingmess-coref/
//       config.json
//       pytorch_model.bin
//       vocab.json
//       merges.txt
//
// When the variable is unset (or the directory lacks `lingmess-coref/
// config.json`) every gated test short-circuits to a silent pass. This is the
// same convention used by `TokenizerTests.fs`; it trades the "skipped" status
// for not pulling in the extra `Xunit.SkippableFact` package.

open System
open System.IO
open Xunit
open Swensen.Unquote
open TorchSharp
open FastCoref
open FastCoref.Clustering
open FastCoref.Api
open FastCoref.Tokenizer
open FastCoref.LingMessModel
open FastCoref.LingMessInference

/// Canonical sample text mirroring the Python `fastcoref` test suite.
let private aliceText =
    "Alice goes down the rabbit hole. \
     Where she would discover a new reality beyond her expectations."

let private modelsDir () : string option =
    match Environment.GetEnvironmentVariable "FASTCOREF_MODELS_DIR" with
    | null
    | "" -> None
    | path -> Some path

let private lingmessDir () : string option =
    modelsDir ()
    |> Option.map (fun root -> Path.Combine(root, "lingmess-coref"))
    |> Option.filter (fun p -> File.Exists(Path.Combine(p, Utils.HfFiles.Config)))

let private mentionStrings (c: Cluster<Mention>) =
    Cluster.toList c |> List.map (fun m -> m.Text)

[<Fact>]
let ``skip path is exercised when FASTCOREF_MODELS_DIR is unset`` () =
    // Sanity-check the gating helper itself: when the env var is missing,
    // `lingmessDir ()` must return `None` so the heavy tests below can
    // early-return cleanly. When CI sets the variable, this test still
    // passes — both branches are valid.
    match lingmessDir () with
    | None -> test <@ true @>
    | Some dir -> test <@ Directory.Exists dir @>

[<Fact>]
let ``LingMess loads and predicts on Alice text`` () =
    match lingmessDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.LingMess)
        let result = coref.Predict aliceText
        test <@ result.Text = aliceText @>
        test <@ result.Clusters.Length >= 1 @>

        for cluster in result.Clusters do
            test <@ Cluster.length cluster >= 2 @>

            for m in Cluster.toList cluster do
                let s = CharIdx.value m.Span.Start
                let e = CharIdx.value m.Span.End
                test <@ 0 <= s && s <= e && e <= aliceText.Length @>
                test <@ m.Text.Trim().Length > 0 @>

[<Fact>]
let ``LingMess discovers Alice-she-her cluster`` () =
    match lingmessDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.LingMess)
        let result = coref.Predict aliceText

        let aliceCluster =
            result.Clusters
            |> List.tryFind (fun c ->
                mentionStrings c
                |> List.exists (fun m -> m.Contains("Alice", StringComparison.Ordinal)))

        test <@ aliceCluster.IsSome @>

        let texts =
            aliceCluster.Value
            |> mentionStrings
            |> List.map (fun m -> m.ToLowerInvariant())

        test <@ texts |> List.exists (fun m -> m.Contains "she") @>

[<Fact>]
let ``PredictBatch returns one result per input`` () =
    match lingmessDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.LingMess)

        let inputs =
            [| "John saw Mary. He waved at her."; "The car broke down. It was old." |]

        let results = coref.PredictBatch inputs
        test <@ results.Length = 2 @>

        for r in results do
            test <@ r.Clusters.Length >= 1 @>

[<Fact>]
let ``Tokenizer encodes Alice text into BOS-prefixed ids`` () =
    match lingmessDir () with
    | None -> ()
    | Some dir ->
        let tokenizer = new RobertaTokenizer(dir)
        let enc = tokenizer.Encode aliceText
        test <@ enc.InputIds.[0] = tokenizer.BosId @>
        test <@ enc.InputIds.[enc.InputIds.Length - 1] = tokenizer.EosId @>
        test <@ enc.InputIds.Length = enc.AttentionMask.Length @>
        test <@ enc.InputIds.Length = enc.Offsets.Length @>
        test <@ enc.Offsets.[0] = TokenOffset.Special @>

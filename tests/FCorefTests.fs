module FastCoref.Tests.FCorefTests

// Smoke tests for `FastCoref.FCorefInference` + `FastCoref.FCorefModel`,
// driven through `FastCoref.Api.CorefModel` for the char-domain assertions.
//
// The TorchSharp runtime currently HANGS on macOS 12 + Intel libtorch builds,
// so any test that actually loads weights or runs a forward pass is gated on
// the `FASTCOREF_MODELS_DIR` environment variable. The expected layout is:
//
//   $FASTCOREF_MODELS_DIR/
//     f-coref/
//       config.json
//       pytorch_model.bin
//       vocab.json
//       merges.txt
//
// When the variable is unset (or the directory lacks `f-coref/config.json`)
// every gated test short-circuits to a silent pass. This is the same
// convention used by `TokenizerTests.fs`; it trades the "skipped" status
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
open FastCoref.FCorefModel
open FastCoref.FCorefInference

/// Canonical sample text mirroring the Python `fastcoref` test suite.
let private aliceText =
    "Alice goes down the rabbit hole. \
     Where she would discover a new reality beyond her expectations."

let private modelsDir () : string option =
    match Environment.GetEnvironmentVariable "FASTCOREF_MODELS_DIR" with
    | null
    | "" -> None
    | path -> Some path

let private fcorefDir () : string option =
    modelsDir ()
    |> Option.map (fun root -> Path.Combine(root, "f-coref"))
    |> Option.filter (fun p -> File.Exists(Path.Combine(p, Utils.HfFiles.Config)))

let private mentionStrings (c: Cluster<Mention>) =
    Cluster.toList c |> List.map (fun m -> m.Text)

[<Fact>]
let ``skip path is exercised when FASTCOREF_MODELS_DIR is unset`` () =
    // Sanity-check the gating helper itself: when the env var is missing,
    // `fcorefDir ()` must return `None` so the heavy tests below can
    // early-return cleanly. When CI sets the variable, this test still
    // passes — both branches are valid.
    match fcorefDir () with
    | None -> test <@ true @>
    | Some dir -> test <@ Directory.Exists dir @>

[<Fact>]
let ``FCoref loads and predicts on Alice text`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
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
let ``FCoref discovers Alice-she-her cluster`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)
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
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        use coref = new CorefModel(dir, CorefKind.FCoref)

        let inputs =
            [| "John saw Mary. He waved at her."; "The car broke down. It was old." |]

        let results = coref.PredictBatch inputs
        test <@ results.Length = 2 @>

        for r in results do
            test <@ r.Clusters.Length >= 1 @>

[<Fact>]
let ``Tokenizer encodes Alice text into BOS-prefixed ids`` () =
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        let tokenizer = new RobertaTokenizer(dir)
        let enc = tokenizer.Encode aliceText
        test <@ enc.InputIds.[0] = tokenizer.BosId @>
        test <@ enc.InputIds.[enc.InputIds.Length - 1] = tokenizer.EosId @>
        test <@ enc.InputIds.Length = enc.AttentionMask.Length @>
        test <@ enc.InputIds.Length = enc.Offsets.Length @>
        test <@ enc.Offsets.[0] = TokenOffset.Special @>

[<Fact>]
let ``FinalLogits last column is null antecedent (=0)`` () =
    // Sanity check that the null-antecedent column convention is intact:
    // the FCoref head appends a zero-valued column at index k as the "no
    // antecedent" choice. Only meaningful when at least one mention was
    // extracted, so we guard on `TokenSpans.Length > 0`.
    match fcorefDir () with
    | None -> ()
    | Some dir ->
        let tokenizer = new RobertaTokenizer(dir)
        let model, _ = FCorefModel.load dir torch.CPU
        let pred = FCorefInference.predict model tokenizer aliceText

        if pred.TokenSpans.Length > 0 then
            let lastCol = Array2D.length2 pred.FinalLogits - 1
            test <@ pred.FinalLogits.[0, lastCol] = 0.0f @>

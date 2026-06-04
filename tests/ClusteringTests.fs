module FastCoref.Tests.ClusteringTests

open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Clustering

let private mkLogits (rows: float32[][]) : float32[,] =
    let r = rows.Length
    let c = if r = 0 then 0 else rows.[0].Length
    Array2D.init r c (fun i j -> rows.[i].[j])

let private toks = TokenIdx.ofArray

let private asTuples (clusters: Cluster<TokenSpan> list) : (int * int) list list =
    clusters
    |> List.map (fun c ->
        Cluster.toList c
        |> List.map (fun s ->
            let a = TokenIdx.value s.Start
            let b = TokenIdx.value s.End
            (a, b)))

[<Fact>]
let ``empty input yields no clusters`` () =
    let clusters =
        Clustering.extractClusters
            { MentionStartIds = [||]
              MentionEndIds = [||]
              FinalLogits = Array2D.zeroCreate 0 1 }

    test <@ asTuples clusters = [] @>

[<Fact>]
let ``single mention with null antecedent yields no clusters`` () =
    let starts = toks [| 0 |]
    let ends = toks [| 1 |]
    let finalLogits = mkLogits [| [| -1.0f; 0.0f |] |]

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = starts
              MentionEndIds = ends
              FinalLogits = finalLogits }

    test <@ asTuples clusters = [] @>

[<Fact>]
let ``two mentions, second points to first`` () =
    let starts = toks [| 0; 5 |]
    let ends = toks [| 2; 7 |]
    let finalLogits = mkLogits [| [| -1.0f; -1.0f; 0.0f |]; [| 2.0f; -1.0f; -1.0f |] |]

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = starts
              MentionEndIds = ends
              FinalLogits = finalLogits }

    test <@ asTuples clusters = [ [ (0, 2); (5, 7) ] ] @>

[<Fact>]
let ``chain of three merges into single cluster`` () =
    let starts = toks [| 0; 5; 10 |]
    let ends = toks [| 2; 7; 12 |]

    let finalLogits =
        mkLogits
            [| [| -1.0f; -1.0f; -1.0f; 0.0f |]
               [| 2.0f; -1.0f; -1.0f; -1.0f |]
               [| -1.0f; 2.0f; -1.0f; -1.0f |] |]

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = starts
              MentionEndIds = ends
              FinalLogits = finalLogits }

    test <@ asTuples clusters = [ [ (0, 2); (5, 7); (10, 12) ] ] @>

[<Fact>]
let ``two independent chains yield two clusters`` () =
    let starts = toks [| 0; 5; 10; 20 |]
    let ends = toks [| 2; 7; 12; 22 |]

    let finalLogits =
        mkLogits
            [| [| -1.0f; -1.0f; -1.0f; -1.0f; 0.0f |]
               [| 2.0f; -1.0f; -1.0f; -1.0f; -1.0f |]
               [| -1.0f; -1.0f; -1.0f; -1.0f; 0.0f |]
               [| -1.0f; -1.0f; 2.0f; -1.0f; -1.0f |] |]

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = starts
              MentionEndIds = ends
              FinalLogits = finalLogits }

    test <@ asTuples clusters = [ [ (0, 2); (5, 7) ]; [ (10, 12); (20, 22) ] ] @>

[<Fact>]
let ``forward and self links are ignored (antecedent must be earlier)`` () =
    let starts = toks [| 0; 5 |]
    let ends = toks [| 2; 7 |]
    // Row 0 argmax is mention 1 (a forward link); row 1 argmax is the null
    // column. The real models mask logits[i, j>=i] to -inf before softmax,
    // and `extractClusters` defensively mirrors that — so both rows fall
    // through to "no antecedent" and no cluster is emitted.
    let finalLogits = mkLogits [| [| -1.0f; 2.0f; -1.0f |]; [| -1.0f; -1.0f; 0.0f |] |]

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = starts
              MentionEndIds = ends
              FinalLogits = finalLogits }

    test <@ asTuples clusters = [] @>

[<Fact>]
let ``prettyPrint contains spans`` () =
    let mkSpan (s: int) (e: int) : TokenSpan = { Start = TokenIdx.ofInt s; End = TokenIdx.ofInt e }
    let cluster = { Head = mkSpan 0 2; Rest = [ mkSpan 5 7 ] }
    let s = Clustering.prettyPrint [ cluster ]
    test <@ s.Contains("(0,2)") && s.Contains("(5,7)") @>

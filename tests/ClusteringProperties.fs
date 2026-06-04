module FastCoref.Tests.ClusteringProperties

open System.Collections.Generic
open FsCheck
open FsCheck.FSharp
open FsCheck.Xunit
open FastCoref
open FastCoref.Clustering

type Inputs =
    { Starts: TokenIdx[]
      Ends: TokenIdx[]
      FinalLogits: float32[,] }

let private runClustering (inp: Inputs) =
    Clustering.extractClusters
        { MentionStartIds = inp.Starts
          MentionEndIds = inp.Ends
          FinalLogits = inp.FinalLogits }

let rec private genInputs (maxK: int) : Gen<Inputs> =
    gen {
        let! k = Gen.choose (0, maxK)
        let! startsList = Gen.choose (0, 1000) |> Gen.listOfLength (max 1 (k * 4))

        let distinctStarts =
            startsList |> List.distinct |> List.truncate k |> List.sort

        if distinctStarts.Length < k then
            return! genInputs maxK
        else
            let starts = distinctStarts |> List.toArray
            let! endOffsets = Gen.choose (0, 5) |> Gen.listOfLength k
            let ends = Array.map2 (fun s d -> s + d) starts (List.toArray endOffsets)
            let cols = k + 1
            let! flat = Gen.choose (-1000, 1000) |> Gen.listOfLength (k * cols)

            let logits =
                Array2D.init k cols (fun i j -> float32 flat.[i * cols + j] / 100.0f)

            return
                { Starts = TokenIdx.ofArray starts
                  Ends = TokenIdx.ofArray ends
                  FinalLogits = logits }
    }

type ClusteringArbs =
    static member Inputs() = Arb.fromGen (genInputs 8)

let private asTuples (clusters: Cluster<TokenSpan> list) : (int * int) list list =
    clusters
    |> List.map (fun c ->
        Cluster.toList c
        |> List.map (fun s ->
            let a = TokenIdx.value s.Start
            let b = TokenIdx.value s.End
            (a, b)))

let private referenceClusters
    (starts: TokenIdx[])
    (ends: TokenIdx[])
    (logits: float32[,])
    : (int * int) list list =
    let k = starts.Length

    if k = 0 then
        []
    else
        let cols = logits.GetLength(1)

        // Antecedent must be strictly earlier (j < i) or null (column k);
        // mirrors the masking enforced in `extractClusters`.
        let argmax i =
            let mutable best = k

            for j in 0 .. i - 1 do
                if logits.[i, j] > logits.[i, best] then
                    best <- j

            best

        let links =
            [| for i in 0 .. k - 1 do
                   let a = argmax i
                   if a < k then
                       yield i, a |]

        let mention i : int * int = TokenIdx.value starts.[i], TokenIdx.value ends.[i]
        let clusters = ResizeArray<ResizeArray<int * int>>()
        let cluOf = Dictionary<int * int, int>()

        for (i, a) in links do
            let mi, ma = mention i, mention a

            match cluOf.TryGetValue ma with
            | true, idx ->
                clusters.[idx].Add mi
                cluOf.[mi] <- idx
            | _ ->
                let idx = clusters.Count
                let c = ResizeArray()
                c.Add ma
                c.Add mi
                clusters.Add c
                cluOf.[ma] <- idx
                cluOf.[mi] <- idx

        clusters
        |> Seq.map (fun c -> c |> Seq.distinct |> Seq.sort |> List.ofSeq)
        |> List.ofSeq

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``every mention is in at most one cluster`` (inp: Inputs) =
    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    let all = clusters |> List.collect id
    List.length all = (all |> List.distinct |> List.length)

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``each cluster is sorted ascending and distinct`` (inp: Inputs) =
    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    clusters |> List.forall (fun c -> c = (c |> List.distinct |> List.sort))

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``every cluster has at least two mentions`` (inp: Inputs) =
    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    clusters |> List.forall (fun c -> List.length c >= 2)

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``null-only logits yield empty clusters`` (inp: Inputs) =
    let k = inp.Starts.Length
    let cols = k + 1

    let nullDominant =
        Array2D.init k cols (fun _ j -> if j = k then 1000.0f else -1000.0f)

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = nullDominant }
        |> asTuples

    clusters = []

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``all cluster mentions appear in the input mention set`` (inp: Inputs) =
    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    let mentionSet =
        set [ for i in 0 .. inp.Starts.Length - 1 -> TokenIdx.value inp.Starts.[i], TokenIdx.value inp.Ends.[i] ]

    clusters
    |> List.forall (fun c -> c |> List.forall (fun m -> Set.contains m mentionSet))

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``reference implementation matches actual`` (inp: Inputs) =
    let actual =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    let expected = referenceClusters inp.Starts inp.Ends inp.FinalLogits
    actual = expected

[<Property(Arbitrary = [| typeof<ClusteringArbs> |])>]
let ``cluster count is at most number of linked mentions`` (inp: Inputs) =
    let k = inp.Starts.Length
    let cols = inp.FinalLogits.GetLength(1)

    let argmax i =
        let mutable best = 0

        for j in 1 .. cols - 1 do
            if inp.FinalLogits.[i, j] > inp.FinalLogits.[i, best] then
                best <- j

        best

    let linked =
        if k = 0 then [||]
        else [| 0 .. k - 1 |] |> Array.filter (fun i -> argmax i < k)

    let clusters =
        Clustering.extractClusters
            { MentionStartIds = inp.Starts
              MentionEndIds = inp.Ends
              FinalLogits = inp.FinalLogits }
        |> asTuples

    List.length clusters <= linked.Length

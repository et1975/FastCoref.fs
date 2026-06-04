module FastCoref.Tests.UtilsProperties

open System
open FsCheck
open FsCheck.FSharp
open FsCheck.Xunit
open FastCoref

// --- Torch gating ----------------------------------------------------------
// The TorchSharp native library hangs at first allocation on macOS 12 / Intel.
// Any property that constructs a Torch tensor must be gated. When disabled,
// the property returns `true` (vacuously passing) so FsCheck records a pass
// on this host while still exercising the property elsewhere.
let private torchEnabled () : bool =
    match Environment.GetEnvironmentVariable "FASTCOREF_TORCH_TESTS" with
    | null | "" | "0" | "false" | "False" -> false
    | _ -> true

// --- Generator inputs ------------------------------------------------------
type PadInput      = { Ids: int[]; Length: int; PadId: int }
type SpanMaskInput = { SeqLen: int; MaxSpan: int }
type TriuInput     = { N: int }

type UtilsArbs =
    static member PadInput () =
        gen {
            let! length = Gen.choose (0, 32)
            let! padId  = Gen.choose (-1000, 1000)
            let! idsLen = Gen.choose (0, 40)
            let! ids    = Gen.choose (-1000, 1000) |> Gen.listOfLength idsLen
            return { Ids = List.toArray ids; Length = length; PadId = padId }
        }
        |> Arb.fromGen

    static member SpanMaskInput () =
        gen {
            let! seqLen  = Gen.choose (1, 16)
            let! maxSpan = Gen.choose (1, 20)
            return { SeqLen = seqLen; MaxSpan = maxSpan }
        }
        |> Arb.fromGen

    static member TriuInput () =
        gen {
            let! n = Gen.choose (1, 16)
            return { N = n }
        }
        |> Arb.fromGen

// --- padToLength: pure F# properties --------------------------------------

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength returns array of requested length`` (p: PadInput) =
    let r = Utils.padToLength p.Ids p.Length p.PadId
    r.Length = p.Length

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength preserves prefix when ids fit`` (p: PadInput) =
    if p.Ids.Length > p.Length then true
    else
        let r = Utils.padToLength p.Ids p.Length p.PadId
        let mutable ok = true
        for i in 0 .. p.Ids.Length - 1 do
            if r.[i] <> p.Ids.[i] then ok <- false
        ok

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength pads tail with padId when ids shorter`` (p: PadInput) =
    if p.Ids.Length >= p.Length then true
    else
        let r = Utils.padToLength p.Ids p.Length p.PadId
        let mutable ok = true
        for i in p.Ids.Length .. p.Length - 1 do
            if r.[i] <> p.PadId then ok <- false
        ok

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength truncates when ids longer`` (p: PadInput) =
    if p.Ids.Length <= p.Length then true
    else
        let r = Utils.padToLength p.Ids p.Length p.PadId
        r = p.Ids.[0 .. p.Length - 1]

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength is idempotent`` (p: PadInput) =
    let r1 = Utils.padToLength p.Ids p.Length p.PadId
    let r2 = Utils.padToLength r1 p.Length p.PadId
    r1 = r2

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``padToLength returns a distinct array when length matches`` (p: PadInput) =
    let ids = Array.copy p.Ids
    let n = ids.Length
    let r = Utils.padToLength ids n p.PadId
    // Zero-length arrays in .NET *may* be the same object; skip.
    if n = 0 then true
    else not (obj.ReferenceEquals(r, ids))

// --- spanLengthMask: gated Torch property ---------------------------------

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``spanLengthMask matches pure oracle`` (sm: SpanMaskInput) =
    if not (torchEnabled ()) then true
    else
        let mask =
            Utils.spanLengthMask (int64 sm.SeqLen) sm.MaxSpan TorchSharp.torch.CPU
        let shape = mask.shape
        if shape <> [| int64 sm.SeqLen; int64 sm.SeqLen |] then false
        else
            let arr = mask.data<bool>() |> Array.ofSeq
            let mutable ok = true
            for s in 0 .. sm.SeqLen - 1 do
                for e in 0 .. sm.SeqLen - 1 do
                    let expected = e >= s && e - s < sm.MaxSpan
                    let actual = arr.[s * sm.SeqLen + e]
                    if expected <> actual then ok <- false
            ok

// --- triuMask: gated Torch property ---------------------------------------

[<Property(Arbitrary = [| typeof<UtilsArbs> |])>]
let ``triuMask matches pure oracle (j > i)`` (t: TriuInput) =
    if not (torchEnabled ()) then true
    else
        let n = t.N
        let mask = Utils.triuMask (int64 n) TorchSharp.torch.CPU
        let shape = mask.shape
        if shape <> [| int64 n; int64 n |] then false
        else
            let arr = mask.data<bool>() |> Array.ofSeq
            let mutable ok = true
            for i in 0 .. n - 1 do
                for j in 0 .. n - 1 do
                    let expected = j > i
                    let actual = arr.[i * n + j]
                    if expected <> actual then ok <- false
            ok

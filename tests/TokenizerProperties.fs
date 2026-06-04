module FastCoref.Tests.TokenizerProperties

open System
open System.IO
open Xunit
open FsCheck
open FsCheck.FSharp
open FsCheck.Xunit
open FastCoref
open FastCoref.Tokenizer

// ---------------------------------------------------------------------------
// Gating — mirrors tests/TokenizerTests.fs. Properties that need a live vocab
// short-circuit to `true` when FASTCOREF_MODELS_DIR is not set or does not
// point at a directory with both `vocab.json` and `merges.txt`.
// ---------------------------------------------------------------------------
let private liveModelDir () : string option =
    match Environment.GetEnvironmentVariable("FASTCOREF_MODELS_DIR") with
    | null
    | "" -> None
    | root ->
        let fcorefDir = Path.Combine(root, "f-coref")
        if File.Exists(Path.Combine(fcorefDir, Utils.HfFiles.Vocab))
           && File.Exists(Path.Combine(fcorefDir, Utils.HfFiles.Merges)) then
            Some fcorefDir
        elif File.Exists(Path.Combine(root, Utils.HfFiles.Vocab))
             && File.Exists(Path.Combine(root, Utils.HfFiles.Merges)) then
            Some root
        else
            None

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode spec, reconstructed here for documentation purposes.
// The production table in `Tokenizer.fs` is module-private and so not directly
// reachable from tests; this spec mirrors it so we can at least assert the
// shape invariants the real table must also satisfy.
// ---------------------------------------------------------------------------
let private buildByteEncoderSpec () : char[] =
    let bs = ResizeArray<int>()
    for i in int '!' .. int '~' do
        bs.Add i
    for i in 0xA1 .. 0xAC do
        bs.Add i
    for i in 0xAE .. 0xFF do
        bs.Add i
    let cs = ResizeArray<int>(bs)
    let mutable extra = 0
    for b in 0 .. 255 do
        if not (bs.Contains b) then
            bs.Add b
            cs.Add(256 + extra)
            extra <- extra + 1
    let table = Array.create 256 ' '
    for i in 0 .. bs.Count - 1 do
        table.[bs.[i]] <- char cs.[i]
    table

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
type AsciiText = { Value: string }

type TokenizerArbs =
    static member AsciiText() =
        gen {
            let! len = Gen.choose (0, 40)
            let! chars = Gen.choose (32, 126) |> Gen.listOfLength len
            let s =
                chars
                |> List.map char
                |> List.toArray
                |> System.String
            return { Value = s }
        }
        |> Arb.fromGen

// ---------------------------------------------------------------------------
// Group A — pure properties (no vocab needed)
// ---------------------------------------------------------------------------

[<Fact>]
let ``byte encoder spec is a bijection of size 256`` () =
    let table = buildByteEncoderSpec ()
    Assert.Equal(256, table.Length)
    Assert.Equal(256, table |> Array.distinct |> Array.length)

[<Property>]
let ``utf8 length matches BCL for valid scalar code points`` (NonNegativeInt n) =
    let cp = n % 0x110000
    if cp >= 0xD800 && cp <= 0xDFFF then
        true
    else
        let s = Char.ConvertFromUtf32 cp
        let bclLen = System.Text.Encoding.UTF8.GetByteCount s
        let expected =
            if cp < 0x80 then 1
            elif cp < 0x800 then 2
            elif cp < 0x10000 then 3
            else 4
        bclLen = expected

// ---------------------------------------------------------------------------
// Group B — gated properties (need a live RoBERTa vocab on disk)
// ---------------------------------------------------------------------------

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode returns InputIds, AttentionMask, Offsets of equal length`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        enc.InputIds.Length = enc.AttentionMask.Length
        && enc.InputIds.Length = enc.Offsets.Length

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode AttentionMask is all ones`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        enc.AttentionMask |> Array.forall id

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode first and last offsets are special markers`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        let n = enc.Offsets.Length
        enc.Offsets.[0] = TokenOffset.Special
        && enc.Offsets.[n - 1] = TokenOffset.Special

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode first and last InputIds are BOS and EOS`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        let n = enc.InputIds.Length
        enc.InputIds.[0] = tok.BosId
        && enc.InputIds.[n - 1] = tok.EosId

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode content offsets are well-formed and in input bounds`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let s = text.Value
        let enc = tok.Encode s
        let mutable ok = true
        for i in 1 .. enc.Offsets.Length - 2 do
            match enc.Offsets.[i] with
            | TokenOffset.Content cs ->
                if not (0 <= cs.Start && cs.Start <= cs.End && cs.End <= s.Length) then
                    ok <- false
            | TokenOffset.Special -> ok <- false
        ok

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode content offsets are non-decreasing in start`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        let n = enc.Offsets.Length
        if n <= 3 then
            true
        else
            let starts =
                enc.Offsets
                |> Array.skip 1
                |> Array.truncate (n - 2)
                |> Array.choose (function TokenOffset.Content cs -> Some cs.Start | TokenOffset.Special -> None)
            seq {
                for i in 1 .. starts.Length - 1 do
                    yield starts.[i] >= starts.[i - 1]
            }
            |> Seq.forall id

[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode/Decode ASCII round-trips after leading-space trim`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let tok = Tokenizer.RobertaTokenizer dir
        let enc = tok.Encode text.Value
        let decoded = tok.Decode enc.InputIds
        decoded.TrimStart() = text.Value.TrimStart()

// ---------------------------------------------------------------------------
// Regression property for the GPT-2 prefix-space fix (v2 limitation L5).
// After the fix, no Content offset may start on an ASCII space (U+0020):
// the first sub-token of a word-prefixed piece shifts past the leading space,
// and pure-whitespace pieces become Special. We assert this over arbitrary
// ASCII strings (which include plenty of literal spaces). Vacuously true when
// FASTCOREF_MODELS_DIR is unset.
// ---------------------------------------------------------------------------
[<Property(Arbitrary = [| typeof<TokenizerArbs> |])>]
let ``Encode Content offsets never start on ASCII whitespace`` (text: AsciiText) =
    match liveModelDir () with
    | None -> true
    | Some dir ->
        let s = text.Value
        if s.Length = 0 then
            true
        else
            let tok = Tokenizer.RobertaTokenizer dir
            let enc = tok.Encode s
            enc.Offsets
            |> Array.forall (fun o ->
                match o with
                | TokenOffset.Special -> true
                | TokenOffset.Content cs ->
                    cs.Start >= 0
                    && cs.Start < cs.End
                    && cs.End <= s.Length
                    && s.[cs.Start] <> ' ')

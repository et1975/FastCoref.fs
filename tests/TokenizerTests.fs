module FastCoref.Tests.TokenizerTests

open System
open System.IO
open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Config
open FastCoref.Tokenizer

// Live-vocab tests are gated on FASTCOREF_MODELS_DIR pointing at a folder
// that contains both `vocab.json` and `merges.txt`. When unset, those tests
// short-circuit to a pass — the pure-logic tests below always run.
let private liveModelDir () : string option =
    match Environment.GetEnvironmentVariable("FASTCOREF_MODELS_DIR") with
    | null
    | "" -> None
    | d when
        Directory.Exists d
        && File.Exists(Path.Combine(d, Utils.HfFiles.Vocab))
        && File.Exists(Path.Combine(d, Utils.HfFiles.Merges))
        ->
        Some d
    | _ -> None

[<Fact>]
let ``RobertaTokenizer encodes Hello world with BOS/EOS and aligned shapes`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let enc = tok.Encode "Hello world"
        test <@ enc.InputIds.[0] = tok.BosId @>
        test <@ enc.InputIds.[enc.InputIds.Length - 1] = tok.EosId @>
        test <@ enc.InputIds.Length = enc.AttentionMask.Length @>
        test <@ enc.InputIds.Length = enc.Offsets.Length @>
        test <@ enc.AttentionMask |> Array.forall id @>
        // Standard RoBERTa vocab produces [BOS, "ĠHello"=31414, "Ġworld"=232, EOS].
        test <@ enc.InputIds = [| tok.BosId; TokenId.ofInt 31414; TokenId.ofInt 232; tok.EosId |] @>

[<Fact>]
let ``RobertaTokenizer specials carry Special offsets and content tokens cover input chars`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "Hello world"
        let enc = tok.Encode text
        test <@ enc.Offsets.[0] = TokenOffset.Special @>
        test <@ enc.Offsets.[enc.Offsets.Length - 1] = TokenOffset.Special @>
        for i in 1 .. enc.Offsets.Length - 2 do
            match enc.Offsets.[i] with
            | TokenOffset.Content s -> test <@ s.Start >= 0 && s.End <= text.Length && s.Start < s.End @>
            | TokenOffset.Special -> test <@ false @>
        test <@ enc.Offsets.[1] = TokenOffset.Content { Start = 0; End = 5 } @>
        match enc.Offsets.[2] with
        | TokenOffset.Content s -> test <@ text.Substring(s.Start, s.End - s.Start) = " world" @>
        | TokenOffset.Special -> test <@ false @>

[<Fact>]
let ``RobertaTokenizer Decode round-trips ASCII through Encode`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "Hello world"
        let enc = tok.Encode text
        let decoded = tok.Decode enc.InputIds
        // RoBERTa always reintroduces a leading space from the prefix-space
        // convention; trim it before comparing.
        test <@ decoded.TrimStart() = text @>

// ---------------------------------------------------------------------------
// Regression tests for the GPT-2 prefix-space fix (v2 limitation L5).
//
// The fix in Tokenizer.Encode skips the leading ASCII space of a word-prefixed
// piece when reporting the FIRST sub-token's offset, and emits Special for
// whitespace-only pieces. These tests pin down the post-fix invariants so they
// cannot silently regress.
// ---------------------------------------------------------------------------

/// All Content offsets in an encoding, in token order.
let private contentSpans (enc: Tokenizer.Encoding) : Tokenizer.TextSpan[] =
    enc.Offsets
    |> Array.choose (function
        | TokenOffset.Content s -> Some s
        | TokenOffset.Special -> None)

[<Fact>]
let ``Encode strips leading prefix-space from interior word mentions`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "Alice and Mira talked."
        let enc = tok.Encode text
        let spans = contentSpans enc
        // No content sub-token should start on an ASCII space.
        for s in spans do
            test <@ text.[s.Start] <> ' ' @>
        // Each interior word's first sub-token should anchor at the word's
        // first letter — not the preceding space.
        let andStart = text.IndexOf "and"
        let miraStart = text.IndexOf "Mira"
        let talkedStart = text.IndexOf "talked"
        test <@ andStart = 6 && text.[andStart] = 'a' @>
        test <@ miraStart = 10 && text.[miraStart] = 'M' @>
        test <@ talkedStart = 15 && text.[talkedStart] = 't' @>
        test <@ spans |> Array.exists (fun s -> s.Start = andStart) @>
        test <@ spans |> Array.exists (fun s -> s.Start = miraStart) @>
        test <@ spans |> Array.exists (fun s -> s.Start = talkedStart) @>

[<Fact>]
let ``Encode preserves zero-offset for first word with no virtual prefix needed`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "Mira walks."
        let enc = tok.Encode text
        let spans = contentSpans enc
        test <@ spans.Length > 0 @>
        // The very first content sub-token covers the leading 'M'.
        test <@ spans.[0].Start = 0 @>
        test <@ text.[spans.[0].Start] = 'M' @>

[<Fact>]
let ``Encode emits Special for whitespace-only pieces`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "hello   world"   // three spaces between words
        let enc = tok.Encode text
        // No content offset starts on a whitespace character.
        for s in contentSpans enc do
            test <@ not (Char.IsWhiteSpace text.[s.Start]) @>
        // At least one interior offset is Special (the pure-whitespace piece(s)
        // emitted from the regex's \s+ branches). Skip BOS at [0] and EOS at
        // [n-1] — those are always Special by construction and don't prove the
        // fix.
        let interiorSpecials =
            enc.Offsets
            |> Array.skip 1
            |> Array.truncate (enc.Offsets.Length - 2)
            |> Array.filter (function TokenOffset.Special -> true | _ -> false)
        test <@ interiorSpecials.Length >= 1 @>

[<Fact>]
let ``Encode handles input starting with whitespace`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        let text = "   leading spaces"
        let enc = tok.Encode text
        let spans = contentSpans enc
        test <@ spans.Length > 0 @>
        // First content offset must anchor at the 'l' of "leading" (index 3),
        // never at one of the leading-space indices 0..2.
        test <@ spans.[0].Start = 3 @>
        test <@ text.[spans.[0].Start] = 'l' @>
        // Defence in depth: NO content offset starts inside the leading-space
        // run.
        for s in spans do
            test <@ s.Start >= 3 @>

[<Fact>]
let ``Encode handles non-ASCII whitespace without skipping`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        // 'a', NO-BREAK SPACE (U+00A0), 'b'. The fix uses LITERAL ' ' (U+0020),
        // so the NBSP must NOT trigger the prefix-skip. The 'b' offset should
        // land cleanly at index 2 and 'a' at index 0.
        let text = "a\u00A0b"
        let enc = tok.Encode text
        let spans = contentSpans enc
        // A Content span covering 'a' at [0, 1) must exist.
        test <@ spans |> Array.exists (fun s -> s.Start = 0 && s.End = 1) @>
        // A Content span covering 'b' at [2, 3) must exist.
        test <@ spans |> Array.exists (fun s -> s.Start = 2 && s.End = 3) @>
        // No Content span starts AT the U+00A0 position (index 1).
        for s in spans do
            test <@ s.Start <> 1 @>

[<Fact>]
let ``Encode handles surrogate-pair character (emoji)`` () =
    match liveModelDir () with
    | None -> ()
    | Some d ->
        let tok = Tokenizer.RobertaTokenizer(d)
        // 😀 (U+1F600) encodes as a UTF-16 surrogate pair. text indices:
        //   0:'h' 1:'i' 2:' ' 3:high-surr 4:low-surr 5:'!'.
        let text = "hi \U0001F600!"
        let enc = tok.Encode text
        let spans = contentSpans enc
        // Post-fix invariant: no Content span starts on whitespace.
        for s in spans do
            test <@ not (Char.IsWhiteSpace text.[s.Start]) @>
        // Surrogate sanity: any Content span whose Start is a LOW surrogate
        // must be immediately preceded by another offset ending exactly at
        // that Start (i.e., we never slice between the two halves of a
        // codepoint). In practice charStartInPiece always points at the high
        // surrogate, so this should be vacuously true.
        for s in spans do
            if Char.IsLowSurrogate text.[s.Start] then
                test <@ spans |> Array.exists (fun other -> other.End = s.Start) @>
        // And: no Content span Start IS a low surrogate (the strong form).
        for s in spans do
            test <@ not (Char.IsLowSurrogate text.[s.Start]) @>

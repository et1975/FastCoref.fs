namespace FastCoref

open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open System.Collections.Generic
open FastCoref.Utils
open FastCoref.Config

/// RoBERTa / Longformer byte-level BPE tokenizer ported from the HuggingFace
/// slow tokenizer (`vocab.json` + `merges.txt`). Pure F# — no Torch, no IO
/// other than reading the vocab files at construction time.
module Tokenizer =

    /// Raw character span `[Start..End)` into the original input text.
    /// Lower layer than `Api.CharSpan`, which wraps the same endpoints in
    /// `CharOffset` for cross-type safety; this record stays unwrapped so
    /// the tokenizer can keep its char-offset arithmetic primitive.
    type TextSpan = { Start: int; End: int }

    /// Per-token mapping back to source text. Special tokens (BOS, EOS,
    /// PAD) carry no source span; content tokens carry a non-empty
    /// `TextSpan` (`End > Start`).
    [<RequireQualifiedAccess>]
    type TokenOffset =
        | Special
        | Content of TextSpan

    [<RequireQualifiedAccess>]
    module TokenOffset =
        /// Recover the contiguous character span covered by an inclusive
        /// `[startTok..endTok]` token range. Returns `None` when either
        /// endpoint is out of bounds, is `Special`, or the resulting span
        /// has non-positive length (strict `End > Start`).
        let tryCharSpan (offsets: TokenOffset[]) (startTok: int) (endTok: int) : TextSpan option =
            if startTok < 0
               || endTok < 0
               || startTok >= offsets.Length
               || endTok >= offsets.Length then
                None
            else
                match offsets.[startTok], offsets.[endTok] with
                | TokenOffset.Content s0, TokenOffset.Content s1 when s1.End > s0.Start ->
                    Some { Start = s0.Start; End = s1.End }
                | _ -> None

    /// One encoded text: token ids, per-token attention mask, and a per-token
    /// offset back to the original input. `AttentionMask` is `true` for real
    /// tokens and `false` for pad; `Offsets` is `Special` for BOS/EOS/PAD.
    type Encoding =
        { InputIds: TokenId[]
          AttentionMask: bool[]
          Offsets: TokenOffset[] }

    // GPT-2 pre-tokenizer regex. Splits on contractions, runs of letters,
    // runs of digits, runs of other non-space chars, and whitespace runs
    // (with a lookahead to keep a trailing space attached to the next word).
    let private gpt2Pattern =
        Regex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)

    /// GPT-2 / RoBERTa `bytes_to_unicode` map: a 256-entry table sending each
    /// raw byte to a printable Unicode character so byte-level BPE can be
    /// performed on a string without losing information.
    let private buildByteEncoder () : char[] =
        let bs = ResizeArray<int>()

        for i in int '!' .. int '~' do
            bs.Add i

        for i in 0xA1..0xAC do
            bs.Add i

        for i in 0xAE..0xFF do
            bs.Add i

        let cs = ResizeArray<int>(bs)
        let mutable extra = 0

        for b in 0..255 do
            if not (bs.Contains b) then
                bs.Add b
                cs.Add(256 + extra)
                extra <- extra + 1

        let table = Array.create 256 ' '

        for i in 0 .. bs.Count - 1 do
            table.[bs.[i]] <- char cs.[i]

        table

    let private byteEncoder: char[] = buildByteEncoder ()

    let private byteDecoder: Dictionary<char, byte> =
        let d = Dictionary<char, byte>()

        for b in 0..255 do
            d.[byteEncoder.[b]] <- byte b

        d

    /// Encode one Unicode code point into its UTF-8 byte length.
    let private utf8LenOfCodePoint (cp: int) : int =
        if cp < 0x80 then 1
        elif cp < 0x800 then 2
        elif cp < 0x10000 then 3
        else 4

    type RobertaTokenizer(modelDir: string) =
        let vocabPath = Utils.modelFile modelDir Utils.HfFiles.Vocab
        let mergesPath = Utils.modelFile modelDir Utils.HfFiles.Merges

        let vocab: Dictionary<string, int> =
            let root = Utils.readJsonFile vocabPath
            let d = Dictionary<string, int>(StringComparer.Ordinal)

            for p in root.EnumerateObject() do
                d.[p.Name] <- p.Value.GetInt32()

            d

        let idToTokenMap: Dictionary<int, string> =
            let d = Dictionary<int, string>(vocab.Count)

            for KeyValue(k, v) in vocab do
                d.[v] <- k

            d

        // Rank of each adjacent-pair merge; lower rank = higher priority.
        let mergeRanks: Dictionary<struct (string * string), int> =
            let d = Dictionary<struct (string * string), int>()
            let lines = File.ReadAllLines mergesPath
            let mutable rank = 0

            for line in lines do
                if line.Length > 0 && not (line.StartsWith "#") then
                    let parts = line.Split(' ')

                    if parts.Length = 2 then
                        d.[struct (parts.[0], parts.[1])] <- rank
                        rank <- rank + 1

            d

        let bpeCache = Dictionary<string, string[]>(StringComparer.Ordinal)

        // Apply BPE merges to a single byte-encoded piece, returning the
        // resulting sub-tokens (each is a contiguous slice of the input
        // string, so the joined lengths equal the input length).
        let bpe (token: string) : string[] =
            match bpeCache.TryGetValue token with
            | true, cached -> cached
            | _ ->
                let result =
                    if token.Length <= 1 then
                        [| token |]
                    else
                        let mutable parts = Array.init token.Length (fun i -> string token.[i])
                        let mutable keepMerging = true

                        while keepMerging && parts.Length > 1 do
                            let mutable bestRank = Int32.MaxValue
                            let mutable bestIdx = -1

                            for i in 0 .. parts.Length - 2 do
                                match mergeRanks.TryGetValue(struct (parts.[i], parts.[i + 1])) with
                                | true, r when r < bestRank ->
                                    bestRank <- r
                                    bestIdx <- i
                                | _ -> ()

                            if bestIdx < 0 then
                                keepMerging <- false
                            else
                                let merged = parts.[bestIdx] + parts.[bestIdx + 1]
                                let next = Array.zeroCreate (parts.Length - 1)
                                Array.blit parts 0 next 0 bestIdx
                                next.[bestIdx] <- merged
                                Array.blit parts (bestIdx + 2) next (bestIdx + 1) (parts.Length - bestIdx - 2)
                                parts <- next

                        parts

                bpeCache.[token] <- result
                result

        let tryLookup (token: string) (fallback: TokenId) : TokenId =
            match vocab.TryGetValue token with
            | true, v -> TokenId.ofInt v
            | _ -> fallback

        let bosId = tryLookup "<s>" (TokenId.ofInt 0)
        let padId = tryLookup "<pad>" (TokenId.ofInt 1)
        let eosId = tryLookup "</s>" (TokenId.ofInt 2)
        let unkId = tryLookup "<unk>" (TokenId.ofInt 3)
        let maskId = tryLookup "<mask>" (TokenId.ofInt 50264)

        member _.PadId : TokenId = padId
        member _.BosId : TokenId = bosId
        member _.EosId : TokenId = eosId
        member _.MaskId : TokenId = maskId
        member _.UnknownId : TokenId = unkId
        member _.VocabSize = vocab.Count

        member _.IdToToken(id: TokenId) : string =
            match idToTokenMap.TryGetValue (TokenId.value id) with
            | true, v -> v
            | _ -> "<unk>"

        member _.TokenToId(token: string) : TokenId option =
            match vocab.TryGetValue token with
            | true, v -> Some (TokenId.ofInt v)
            | _ -> None

        /// Encode one string into `Encoding`. Always prefixes BOS and suffixes
        /// EOS. Follows the RoBERTa `add_prefix_space = true` convention: a
        /// virtual leading space is inserted when needed so the first content
        /// word starts with the BPE `Ġ` marker. Offsets are reported relative
        /// to the original (un-prefixed) input string, with the virtual space
        /// excluded from the first token's span.
        member _.Encode(text: string) : Encoding =
            let needsPrefix = text.Length = 0 || not (Char.IsWhiteSpace text.[0])
            let work = if needsPrefix then " " + text else text
            let prefixShift = if needsPrefix then 1 else 0

            let ids = ResizeArray<TokenId>()
            let offsets = ResizeArray<TokenOffset>()
            ids.Add bosId
            offsets.Add TokenOffset.Special

            for m in gpt2Pattern.Matches work do
                let piece = m.Value
                let pieceStart = m.Index
                let bytes = System.Text.Encoding.UTF8.GetBytes piece

                // byteToCharStart[i] = index into `piece` of the first UTF-16
                // code unit of the code point that produced byte i.
                let byteToCharStart = Array.zeroCreate bytes.Length
                let mutable bi = 0
                let mutable ci = 0

                while ci < piece.Length do
                    let isSurrogatePair = Char.IsHighSurrogate piece.[ci] && ci + 1 < piece.Length

                    let cp =
                        if isSurrogatePair then
                            Char.ConvertToUtf32(piece.[ci], piece.[ci + 1])
                        else
                            int piece.[ci]

                    let cpUtf16Len = if isSurrogatePair then 2 else 1
                    let cpUtf8Len = utf8LenOfCodePoint cp

                    for k in 0 .. cpUtf8Len - 1 do
                        byteToCharStart.[bi + k] <- ci

                    bi <- bi + cpUtf8Len
                    ci <- ci + cpUtf16Len

                let encoded =
                    let sb = StringBuilder(bytes.Length)

                    for b in bytes do
                        sb.Append byteEncoder.[int b] |> ignore

                    sb.ToString()

                // BPE sub-tokens are 1:1 with the byte-encoded chars, which are
                // 1:1 with the original UTF-8 bytes. So a sub-token of length L
                // covers exactly L bytes of `piece`.
                let subTokens = bpe encoded
                let mutable byteOffset = 0

                for sub in subTokens do
                    let byteStart = byteOffset
                    let byteEnd = byteOffset + sub.Length
                    byteOffset <- byteEnd
                    let charStartInPiece = byteToCharStart.[byteStart]
                    let lastCharStart = byteToCharStart.[byteEnd - 1]

                    let charEndInPiece =
                        if Char.IsHighSurrogate piece.[lastCharStart] && lastCharStart + 1 < piece.Length then
                            lastCharStart + 2
                        else
                            lastCharStart + 1

                    let absStartRaw = pieceStart + charStartInPiece - prefixShift
                    let absEndRaw = pieceStart + charEndInPiece - prefixShift

                    // GPT-2 prefix-space marker: byte 0x20 maps to 'Ġ' in the BPE
                    // alphabet. Only the FIRST sub-token of a piece can own this
                    // prefix byte (subsequent sub-tokens have charStartInPiece > 0
                    // and start on a letter). When that sub-token's piece begins
                    // with an ASCII space AND has any non-space content following,
                    // shift the start past the space so the mention reads "Mira"
                    // not " Mira". Use literal U+0020 only — Char.IsWhiteSpace
                    // would over-fire on U+00A0 etc., which are NOT used by GPT-2
                    // word prefixing.
                    let pieceHasContentAfterPrefix =
                        charStartInPiece + 1 < piece.Length && piece.[charStartInPiece] = ' '

                    let absStart =
                        let s = if pieceHasContentAfterPrefix then absStartRaw + 1 else absStartRaw
                        max 0 s

                    let absEnd = max absStart absEndRaw

                    // If the entire piece is whitespace (the regex's \s+(?!\S) /
                    // \s+ branches), emit Special so this token can never anchor
                    // a mention span via tryCharSpan.
                    let mutable isWhitespaceOnly = true
                    let mutable k = charStartInPiece

                    while isWhitespaceOnly && k < charEndInPiece do
                        if not (Char.IsWhiteSpace piece.[k]) then
                            isWhitespaceOnly <- false

                        k <- k + 1

                    ids.Add(tryLookup sub unkId)

                    if isWhitespaceOnly then
                        offsets.Add TokenOffset.Special
                    else
                        offsets.Add(TokenOffset.Content { Start = absStart; End = absEnd })

            ids.Add eosId
            offsets.Add TokenOffset.Special
            let inputIds = ids.ToArray()

            { InputIds = inputIds
              AttentionMask = Array.create inputIds.Length true
              Offsets = offsets.ToArray() }

        /// Decode a list of token ids back to text (debugging aid; strips BOS,
        /// EOS, PAD; inverse of byte-level BPE).
        member _.Decode(ids: TokenId[]) : string =
            let sb = StringBuilder()

            for id in ids do
                if id <> bosId && id <> eosId && id <> padId then
                    match idToTokenMap.TryGetValue (TokenId.value id) with
                    | true, t -> sb.Append t |> ignore
                    | _ -> ()

            let joined = sb.ToString()
            let bytes = Array.zeroCreate joined.Length

            for i in 0 .. joined.Length - 1 do
                match byteDecoder.TryGetValue joined.[i] with
                | true, b -> bytes.[i] <- b
                | _ -> bytes.[i] <- byte '?'

            System.Text.Encoding.UTF8.GetString bytes

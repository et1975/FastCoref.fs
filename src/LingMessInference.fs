namespace FastCoref

open System
open System.Collections.Generic
open TorchSharp
open TorchSharp.Modules

/// End-to-end LingMess (Longformer + 7-expert antecedent scorer) inference.
/// Faithful port of `fastcoref.coref_models.modeling_lingmess.LingMessModel.
/// forward(..., return_all_outputs=True)` plus the surrounding clustering
/// driver in `fastcoref.modeling.CorefModel._batch_inference`.
///
/// Public surface mirrors `FCorefInference` (single-doc `predict` returning
/// a `Clustering.CorefPrediction`, plus `predictBatch`) so downstream code
/// can treat the two backbones interchangeably.
module LingMessInference =

    // Cast helpers — MUST be declared before `open type torch` so that
    // `int64` / `float32` remain the BCL conversions (Torch shadows them
    // with `ScalarType` values).
    let inline private i64 (x: ^a) : int64 = int64 x
    let inline private toInt (x: ^a) : int = int x

    type ScalarHelper =
        static member Op(x: float) : Scalar = Scalar.op_Implicit x
        static member Op(x: float32) : Scalar = Scalar.op_Implicit x
        static member Op(x: int64) : Scalar = Scalar.op_Implicit x
        static member Op(x: int) : Scalar = Scalar.op_Implicit x

    let inline private scalar (x: ^T) : Scalar =
        ((^T or ScalarHelper): (static member Op: ^T -> Scalar) x)

    open type TorchSharp.torch

    open FastCoref.Utils
    open FastCoref.Config
    open FastCoref.Tokenizer
    open FastCoref.CorefHead
    open FastCoref.Clustering
    open FastCoref.LingMessModel

    /// The seven antecedent expert categories. The first six are linguistic
    /// pair categories assigned by `buildCategoriesMaskTensor`; `All` is the
    /// extra plane that fires for every assigned pair (mirroring the Python
    /// reference's category index 6). `toIndex` is the only place that knows
    /// the numeric ordering — it MUST agree with the per-category planes
    /// baked into the LingMess checkpoint.
    type private Category =
        | PronPronComp
        | PronPronNoComp
        | PronEnt
        | Match
        | Contain
        | Other
        | All

    [<RequireQualifiedAccess>]
    module private Category =
        let toIndex =
            function
            | PronPronComp -> 0
            | PronPronNoComp -> 1
            | PronEnt -> 2
            | Match -> 3
            | Contain -> 4
            | Other -> 5
            | All -> 6

        /// Total expert planes including `All`; must equal `LingMessModel.NumCats`.
        let count = 7

    // Startup sanity check: keep the F# DU plane count in lockstep with the
    // LingMess checkpoint's baked-in `NumCats`. Lives at module init so a
    // mismatch fails fast on first inference call rather than producing
    // silently-wrong tensors.
    let private _categoryCountOk: unit =
        if i64 Category.count <> NumCats then
            invalidOp (
                sprintf "Category.count (%d) must equal LingMessModel.NumCats (%d)" Category.count (toInt NumCats)
            )

    // The `PRONOUNS_GROUPS` map clusters surface pronouns by referent identity;
    // two singleton-pronoun spans are referent-compatible iff their group ids
    // match. `STOPWORDS` are dropped before computing the entity word-set used
    // to decide `match` / `contain` / `other`. (Mirrors Python
    // `fastcoref.utilities.consts`.)

    module private Consts =

        let PRONOUNS_GROUPS: IReadOnlyDictionary<string, int> =
            let d = Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)

            for (w, g) in
                [ "i", 0
                  "me", 0
                  "my", 0
                  "mine", 0
                  "myself", 0
                  "you", 1
                  "your", 1
                  "yours", 1
                  "yourself", 1
                  "yourselves", 1
                  "he", 2
                  "him", 2
                  "his", 2
                  "himself", 2
                  "she", 3
                  "her", 3
                  "hers", 3
                  "herself", 3
                  "it", 4
                  "its", 4
                  "itself", 4
                  "we", 5
                  "us", 5
                  "our", 5
                  "ours", 5
                  "ourselves", 5
                  "they", 6
                  "them", 6
                  "their", 6
                  "themselves", 6
                  "that", 7
                  "this", 7 ] do
                d.[w] <- g

            d :> _

        let STOPWORDS: HashSet<string> =
            let s = HashSet<string>(StringComparer.OrdinalIgnoreCase)

            for w in
                [ "'s"
                  "a"
                  "all"
                  "an"
                  "and"
                  "at"
                  "for"
                  "from"
                  "in"
                  "into"
                  "more"
                  "of"
                  "on"
                  "or"
                  "some"
                  "the"
                  "these"
                  "those" ] do
                s.Add(w) |> ignore

            s

    /// Result of running LingMess end-to-end on a single text is the
    /// shared `Clustering.CorefPrediction` — see `FCorefInference`.

    // ---------------------------------------------------------------------
    // Tensor helpers (private; we don't want to pollute `Utils` with
    // LingMess-specific code).
    // ---------------------------------------------------------------------

    /// Port of `fastcoref.utilities.util.mask_tensor`:
    ///   `t' = clamp(t + (1 - mask) * -10000, -10000, 10000)`.
    let private maskTensor (t: Tensor) (mask: Tensor) : Tensor =
        let m = mask.to_type (t.dtype)
        let oneMinus = m.mul(scalar -1.0).add (scalar 1.0)
        let penalty = oneMinus.mul (scalar -10000.0)
        t.add(penalty).clamp (scalar -10000.0, scalar 10000.0)

    /// `[seqLen, seqLen]` float mask with `mask[s, e] = 1` iff
    /// `s ≤ e ≤ s + maxSpanLen - 1`. Identical to the mask used by FCoref.
    let private mentionMask (seqLen: int64) (maxSpanLen: Span) (dtype: ScalarType) (device: Device) : Tensor =
        torch
            .ones([| seqLen; seqLen |], dtype = dtype, device = device)
            .triu(0L)
            .tril (i64 (Span.value maxSpanLen) - 1L)

    /// CPU copy of a 2-D tensor as `float32[,]`.
    let private toF32Matrix (t: Tensor) : float32[,] =
        use t32 = t.to_type(ScalarType.Float32).cpu().contiguous ()
        let rows = toInt t32.shape.[0]
        let cols = toInt t32.shape.[1]
        let arr = Array2D.zeroCreate<float32> rows cols
        let span = t32.data<float32> ()

        for i in 0 .. rows - 1 do
            for j in 0 .. cols - 1 do
                arr.[i, j] <- span.[i64 (i * cols + j)]

        arr

    /// CPU copy of a 1-D int-like tensor as `int[]`.
    let private toIntArray (t: Tensor) : int[] =
        use t64 = t.to_type(ScalarType.Int64).cpu().contiguous ()
        let n = toInt t64.shape.[0]
        let arr = Array.zeroCreate<int> n
        let span = t64.data<int64> ()

        for i in 0 .. n - 1 do
            arr.[i] <- toInt span.[i64 i]

        arr

    let private deviceOf (model: LingMessModel) : Device =
        let p = model.parameters (true) |> Seq.head
        p.device

    // ---------------------------------------------------------------------
    // Linguistic per-pair category logic.
    // ---------------------------------------------------------------------

    /// Compute the surface word set (lower-cased, stop-words removed) and
    /// the pronoun-group id for a token span `[startTok .. endTok]`. The
    /// pronoun check inspects the RAW word set (BEFORE stop-word removal),
    /// matching Python's `get_pronoun_id(span)` semantics — `span` is what
    /// `_get_categories_labels` passes in, not `span - STOPWORDS`. Invalid
    /// / special-token spans collapse to `(-1, ∅)`.
    let private spanWordsAndPronoun
        (text: string)
        (offsets: TokenOffset[])
        (startTok: int)
        (endTok: int)
        : int * HashSet<string> =
        let empty () =
            HashSet<string>(StringComparer.OrdinalIgnoreCase)

        match TokenOffset.tryCharSpan offsets startTok endTok with
        | None -> -1, empty ()
        | Some cs ->
            let surface = text.Substring(cs.Start, cs.End - cs.Start).ToLowerInvariant()

            let pieces =
                surface.Split([| ' '; '\t'; '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)

            let raw = HashSet<string>(pieces, StringComparer.OrdinalIgnoreCase)

            let pronounId =
                if raw.Count = 1 then
                    let w = Seq.head raw

                    match Consts.PRONOUNS_GROUPS.TryGetValue w with
                    | true, gid -> gid
                    | _ -> -1
                else
                    -1

            let words = HashSet<string>(raw, StringComparer.OrdinalIgnoreCase)
            words.ExceptWith Consts.STOPWORDS
            pronounId, words

    /// Build the `[1, NumCats, maxK, maxK]` float mask consumed by
    /// `_calc_coref_logits`. Row `i`, col `j` (with `j < i`) carries the
    /// pair's category id; the upper triangle and diagonal stay at `-1`
    /// and contribute zero. Category `6` (ALL) marks every assigned pair.
    let private buildCategoriesMaskTensor
        (text: string)
        (offsets: TokenOffset[])
        (startIdsHost: int[])
        (endIdsHost: int[])
        (device: Device)
        (dtype: ScalarType)
        : Tensor =
        let maxK = startIdsHost.Length
        let pids = Array.zeroCreate maxK
        let words = Array.zeroCreate<HashSet<string>> maxK

        for i in 0 .. maxK - 1 do
            let pid, ws = spanWordsAndPronoun text offsets startIdsHost.[i] endIdsHost.[i]
            pids.[i] <- pid
            words.[i] <- ws

        // Pronoun-aware default fill for every j < i pair.
        let labels: Category option[,] = Array2D.create maxK maxK None

        for i in 0 .. maxK - 1 do
            for j in 0 .. i - 1 do
                let pi, pj = pids.[i], pids.[j]

                if pi >= 0 && pj >= 0 then
                    labels.[i, j] <- Some(if pi = pj then PronPronComp else PronPronNoComp)
                elif pi >= 0 || pj >= 0 then
                    labels.[i, j] <- Some PronEnt
                else
                    // ent-ent default, refined below to Match/Contain where applicable.
                    labels.[i, j] <- Some Other

        // Refine ent-ent pairs to `match` / `contain` using an inverted
        // word→entity-span index (matches Python's optimisation).
        let wordToEnts =
            Dictionary<string, ResizeArray<int>>(StringComparer.OrdinalIgnoreCase)

        for i in 0 .. maxK - 1 do
            if pids.[i] = -1 then
                for w in words.[i] do
                    match wordToEnts.TryGetValue w with
                    | true, lst -> lst.Add i
                    | _ ->
                        let lst = ResizeArray<int>()
                        lst.Add i
                        wordToEnts.[w] <- lst

        for i in 0 .. maxK - 1 do
            if pids.[i] = -1 then
                let spanI = words.[i]
                let candidates = HashSet<int>()

                for w in spanI do
                    match wordToEnts.TryGetValue w with
                    | true, lst ->
                        for j in lst do
                            candidates.Add j |> ignore
                    | _ -> ()

                candidates.Remove i |> ignore

                for j in candidates do
                    if j < i then
                        let spanJ = words.[j]

                        if spanI.SetEquals spanJ then
                            labels.[i, j] <- Some Match
                        elif spanI.IsSubsetOf spanJ || spanJ.IsSubsetOf spanI then
                            labels.[i, j] <- Some Contain

        // Materialise `[N, maxK, maxK]` row-major then upload. The `All`
        // plane fires for every assigned pair (Python's index-6 catch-all).
        let flat = Array.zeroCreate<float32> (Category.count * maxK * maxK)
        let allPlane = Category.toIndex All

        for i in 0 .. maxK - 1 do
            for j in 0 .. maxK - 1 do
                match labels.[i, j] with
                | None -> ()
                | Some cat ->
                    let c = Category.toIndex cat
                    flat.[c * maxK * maxK + i * maxK + j] <- 1.0f
                    flat.[allPlane * maxK * maxK + i * maxK + j] <- 1.0f

        torch
            .tensor(flat, dtype = ScalarType.Float32, device = device)
            .view([| 1L; NumCats; i64 maxK; i64 maxK |])
            .to_type (dtype)

    // ---------------------------------------------------------------------
    // Driver: full forward pass for one document.
    // ---------------------------------------------------------------------

    let private predictOne (model: LingMessModel) (tokenizer: RobertaTokenizer) (text: string) : CorefPrediction =

        let device = deviceOf model
        let cfg = model.Config
        let maxSpan = cfg.CorefHead.MaxSpanLength
        let topLambda = cfg.CorefHead.TopLambda

        // 1. Tokenize. We do NOT segment: Longformer handles long docs in
        //    one pass. The Python reference also feeds the encoder a single
        //    `[1, L]` batch (`PadCollator` only pads within a batch).
        let enc = tokenizer.Encode text
        let seqLen = i64 enc.InputIds.Length

        // 2. Build the encoder input tensors. SIMPLIFIED v1: the Python
        //    `fastcoref` inference driver passes no `global_attention_mask`;
        //    we mirror that with an all-zeros mask so behaviour matches
        //    bit-for-bit. TODO(v2): emit `[1, 0, 0, …]` BOS-global per the
        //    Longformer paper for better long-document quality.
        let inputIds =
            torch
                .tensor(
                    enc.InputIds |> Array.map (fun x -> i64 (TokenId.value x)),
                    dtype = ScalarType.Int64,
                    device = device
                )
                .unsqueeze (0L)

        let attentionMask =
            torch
                .tensor(
                    enc.AttentionMask |> Array.map (fun b -> if b then 1L else 0L),
                    dtype = ScalarType.Int64,
                    device = device
                )
                .unsqueeze (0L)

        let globalAttentionMask =
            torch.zeros ([| 1L; seqLen |], dtype = ScalarType.Int64, device = device)

        // 3. Encoder → `[1, L, H]`. Coref MLPs are applied LATER, on the
        //    gathered top-k reps only (per the Python reference), so we
        //    deliberately do NOT call `ForwardReps`.
        let hidden = model.Longformer.forward (inputIds, attentionMask, globalAttentionMask)
        let H = hidden.shape.[2]

        // 4. Mention logits over the full sequence.
        let sMent = model.StartMentionMlp.forward (hidden)
        let eMent = model.EndMentionMlp.forward (hidden)

        let mentionLogitsRaw =
            CorefHead.computeMentionLogits sMent eMent model.MentionS2EClf model.MentionStartClf model.MentionEndClf

        let menMask =
            mentionMask seqLen maxSpan mentionLogitsRaw.dtype device
            |> fun m -> m.unsqueeze (0L) // [1, L, L]

        let mentionLogits = maskTensor mentionLogitsRaw menMask

        // 5. Top-k pruning. `k = floor(actual_seq_len * top_lambda)` per doc.
        let actualLen = attentionMask.sum (dim = -1L) // [1] int64

        let kTensor =
            actualLen.to_type(ScalarType.Float32).mul(scalar topLambda).to_type (ScalarType.Int64) // [1] int64

        let kCpu = kTensor.cpu().data<int64> ()
        let maxK = toInt kCpu.[0L]

        if maxK <= 0 then
            { Text = text
              Clusters = []
              TokenSpans = [||]
              Offsets = enc.Offsets
              FinalLogits = Array2D.zeroCreate<float32> 0 1
              SpanToIndex = (Dictionary<TokenSpan, int>() :> IReadOnlyDictionary<_, _>)
              Logits = CorefLogits.empty }
        else

            let flat = mentionLogits.view ([| 1L; seqLen * seqLen |])
            let struct (_, topkIdx) = torch.topk (flat, maxK, 1, true, true) // [1, maxK] int64

            // span_mask: arange(maxK) < k_per_doc. For batch=1 it's all-1, but
            // we follow the same shape Python uses so the masking math stays
            // identical and the code is robust to future batching.
            let arangeK = torch.arange(i64 maxK, device = device).unsqueeze (0L)
            let spanMask = arangeK.lt(kTensor.unsqueeze (1L)).to_type (ScalarType.Int64) // [1, maxK]

            let invalidK =
                spanMask.mul(scalar -1.0).add(scalar 1.0).to_type(ScalarType.Int64).mul (scalar (seqLen * seqLen - 1L))

            let topkIdx = topkIdx.mul(spanMask).add (invalidK)
            let struct (sortedIdx, _) = topkIdx.sort (dim = -1L, descending = false)

            let startIds = sortedIdx.div (scalar seqLen, torch.RoundingMode.floor) // [1, maxK]
            let endIds = sortedIdx.remainder (scalar seqLen)

            // Per-mention scalars → [1, maxK, maxK] additive matrix.
            let topkMentionScalars =
                mentionLogits.view([| 1L; seqLen * seqLen |]).gather (1L, sortedIdx)

            let topkMentionLogitsKK =
                topkMentionScalars.unsqueeze (-1L) + topkMentionScalars.unsqueeze (-2L)

            // 6. Gather hidden reps at the chosen positions, then apply the
            //    coref-side per-category MLPs (`coref_*_all_mlps`). This MUST
            //    happen here, on the gathered reps — Python applies them
            //    AFTER pruning, not on the full sequence.
            let expandHidden (ids: Tensor) =
                ids.unsqueeze(-1L).expand ([| 1L; i64 maxK; H |])

            let topkStartReps = hidden.gather (1L, expandHidden startIds) // [1, maxK, H]
            let topkEndReps = hidden.gather (1L, expandHidden endIds)

            let F = i64 cfg.CorefHead.FfnnSize
            let allStartsFlat = model.CorefStartAllMlps.forward (topkStartReps) // [1, maxK, N*F]
            let allEndsFlat = model.CorefEndAllMlps.forward (topkEndReps)
            // [1, maxK, N, F] → permute to bnkf = [1, N, maxK, F].
            let allStarts =
                allStartsFlat.view([| 1L; i64 maxK; NumCats; F |]).permute(0L, 2L, 1L, 3L).contiguous ()

            let allEnds =
                allEndsFlat.view([| 1L; i64 maxK; NumCats; F |]).permute(0L, 2L, 1L, 3L).contiguous ()

            // 7. Four bilinear einsums summed → per-category antecedent logits.
            //    `Parameter` IS-A `Tensor`, so it passes straight to `einsum`.
            let wS2S: Tensor = model.AntecedentS2SWeights
            let wE2E: Tensor = model.AntecedentE2EWeights
            let wS2E: Tensor = model.AntecedentS2EWeights
            let wE2S: Tensor = model.AntecedentE2SWeights
            let s2s = torch.einsum ("bnkf,nfg,bnlg->bnkl", [| allStarts; wS2S; allStarts |])
            let e2e = torch.einsum ("bnkf,nfg,bnlg->bnkl", [| allEnds; wE2E; allEnds |])
            let s2e = torch.einsum ("bnkf,nfg,bnlg->bnkl", [| allStarts; wS2E; allEnds |])
            let e2s = torch.einsum ("bnkf,nfg,bnlg->bnkl", [| allEnds; wE2S; allStarts |])
            let categoriesLogitsCore = s2s + e2e + s2e + e2s // [1, N, maxK, maxK]

            // 8. Four bias einsums. The S2E bias contracts against `ends`
            //    and the E2S bias against `starts` — intentional asymmetry
            //    matching Python `_calc_coref_logits` lines 308–311.
            let bS2S: Tensor = model.AntecedentS2SBiases
            let bE2E: Tensor = model.AntecedentE2EBiases
            let bS2E: Tensor = model.AntecedentS2EBiases
            let bE2S: Tensor = model.AntecedentE2SBiases
            let biasS2S = torch.einsum("bnkf,nf->bnk", [| allStarts; bS2S |]).unsqueeze (-2L)
            let biasE2E = torch.einsum("bnkf,nf->bnk", [| allEnds; bE2E |]).unsqueeze (-2L)
            let biasS2E = torch.einsum("bnkf,nf->bnk", [| allEnds; bS2E |]).unsqueeze (-2L)
            let biasE2S = torch.einsum("bnkf,nf->bnk", [| allStarts; bE2S |]).unsqueeze (-2L)
            let categoriesLogits = categoriesLogitsCore + biasS2S + biasE2E + biasS2E + biasE2S

            // 9. Linguistic per-pair categories → mask tensor, then sum over
            //    categories and add the mention-score matrix.
            let startIdsHost = toIntArray (startIds.view ([| i64 maxK |]))
            let endIdsHost = toIntArray (endIds.view ([| i64 maxK |]))

            let catMasks =
                buildCategoriesMaskTensor text enc.Offsets startIdsHost endIdsHost device categoriesLogits.dtype

            let summed = (categoriesLogits * catMasks).sum (dim = 1L) // [1, maxK, maxK]
            let finalLogitsPre = summed + topkMentionLogitsKK

            // 10. Antecedent mask: strict lower-triangle ∧ valid-span mask.
            let antecedentMask =
                torch.ones_like(finalLogitsPre).tril(-1L).mul (spanMask.to_type(finalLogitsPre.dtype).unsqueeze (-1L))

            let finalLogitsMasked = maskTensor finalLogitsPre antecedentMask

            let zeros =
                torch.zeros ([| 1L; i64 maxK; 1L |], dtype = finalLogitsMasked.dtype, device = device)

            let finalLogits = torch.cat ([| finalLogitsMasked; zeros |], dim = -1L) // [1, maxK, maxK+1]

            // 11. CPU copies + clustering + char-offset mapping.
            let finalMat = toF32Matrix (finalLogits.squeeze (Nullable<int64>(0L)))

            let startTokArr = TokenIdx.ofArray startIdsHost
            let endTokArr = TokenIdx.ofArray endIdsHost

            let clusters =
                Clustering.extractClusters
                    { MentionStartIds = startTokArr
                      MentionEndIds = endTokArr
                      FinalLogits = finalMat }

            let tokenSpans = Array.zeroCreate<TokenSpan> maxK
            let spanToIndex = Dictionary<TokenSpan, int>()

            for i in 0 .. maxK - 1 do
                let span =
                    { Start = startTokArr.[i]
                      End = endTokArr.[i] }

                tokenSpans.[i] <- span

                if not (spanToIndex.ContainsKey span) then
                    spanToIndex.[span] <- i

            // Build the char-keyed index up front so `CorefLogits.TryGetByCharSpan`
            // is O(1) without doing a runtime CharSpan -> TokenSpan inversion.
            let charSpanToIndex = Dictionary<struct (int * int), int>()

            for KeyValue(tokenSpan, idx) in spanToIndex do
                let ts = TokenIdx.value tokenSpan.Start
                let te = TokenIdx.value tokenSpan.End

                match TokenOffset.tryCharSpan enc.Offsets ts te with
                | Some cs -> charSpanToIndex.[struct (cs.Start, cs.End)] <- idx
                | None -> ()

            let logits = CorefLogits(charSpanToIndex :> IReadOnlyDictionary<_, _>, finalMat)

            { Text = text
              Clusters = clusters
              TokenSpans = tokenSpans
              Offsets = enc.Offsets
              FinalLogits = finalMat
              SpanToIndex = (spanToIndex :> IReadOnlyDictionary<_, _>)
              Logits = logits }

    /// Run LingMess inference on a single string. Switches the model to
    /// eval mode and wraps the forward pass in `torch.no_grad()`.
    let predict (model: LingMessModel) (tokenizer: RobertaTokenizer) (text: string) : CorefPrediction =
        model.eval ()
        use _ = torch.no_grad ()
        predictOne model tokenizer text

    /// Batch overload. Documents are NOT batched along the encoder's
    /// first dim — Longformer documents are typically long enough that
    /// padding-the-shorter-to-longer wastes too much compute. Use this
    /// for convenience, not raw throughput.
    let predictBatch
        (model: LingMessModel)
        (tokenizer: RobertaTokenizer)
        (texts: IReadOnlyList<string>)
        : CorefPrediction[] =
        model.eval ()
        use _ = torch.no_grad ()
        texts |> Seq.map (predictOne model tokenizer) |> Array.ofSeq

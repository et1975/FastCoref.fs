namespace FastCoref

open System
open System.Collections.Generic
open TorchSharp
open TorchSharp.Modules

/// End-to-end FCoref inference: tokenize → segment with the LeftOversCollator
/// pattern → run the RoBERTa encoder + coreference head → top-k mention
/// pruning → 4-direction antecedent bilinear scoring → cluster decoding →
/// map token-index spans back to character offsets in the original input.
///
/// This module is a faithful port of `fastcoref.modeling.CorefModel.
/// _batch_inference` driving `fastcoref.coref_models.modeling_fcoref.
/// FCorefModel.forward(..., return_all_outputs=True)`.
module FCorefInference =

    // Cast helpers MUST come before `open type torch`, otherwise `int64`/
    // `float32` resolve to `torch.int64`/`torch.float32` (ScalarType values)
    // instead of the F# conversion functions.
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
    open FastCoref.FCorefModel

    // -----------------------------------------------------------------
    // Tensor helpers — kept private to this module so we don't pollute the
    // Utils surface with FCoref-specific code.
    // -----------------------------------------------------------------

    /// Port of `fastcoref.utilities.util.mask_tensor`:
    ///   t' = clamp(t + (1 - mask) * -10000, -10000, 10000)
    let private maskTensor (t: Tensor) (mask: Tensor) : Tensor =
        let m = mask.to_type (t.dtype)
        let oneMinus = m.mul(scalar -1.0).add (scalar 1.0)
        let penalty = oneMinus.mul (scalar -10000.0)
        t.add(penalty).clamp (scalar -10000.0, scalar 10000.0)

    /// `[seqLen, seqLen]` float mask with `mask[s, e] = 1` iff
    /// `s ≤ e ≤ s + maxSpanLen - 1`, i.e. mention spans of allowed length.
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

    // -----------------------------------------------------------------
    // Segmentation — mirrors `LeftOversCollator`.
    // -----------------------------------------------------------------

    /// Internal structure carrying the per-document segmentation result.
    /// Ids are pre-cast to int64 ready for `torch.tensor`; masks stay as
    /// `bool[]` (matching `Encoding.AttentionMask`) and are converted to
    /// int64 only at the torch upload site.
    type private Segmented =
        { MainIds: int64[] // segments * segLen
          MainMask: bool[] // segments * segLen
          SegmentCount: int // number of main segments (>= 1)
          SegLen: int
          LeftoverIds: int64[] // empty if no leftover
          LeftoverMask: bool[] // empty if no leftover
          LeftoverLen: int } // 0 if no leftover

    /// Convert a bool attention mask to int64, the dtype the encoder
    /// stack expects for `torch.tensor` uploads.
    let private maskToI64 (mask: bool[]) : int64[] =
        mask |> Array.map (fun b -> if b then 1L else 0L)

    /// Pads `enc.InputIds` up to the next multiple of `segLen` (using `padId`,
    /// with the matching attention-mask positions set to `false`), chunks the
    /// result into segments of size `segLen`, then peels the final segment
    /// off as a leftover iff (a) there is more than one segment and (b) the
    /// final segment contains any padding.
    let private segment (enc: Encoding) (padId: TokenId) (segLen: Segment) : Segmented =
        let total = enc.InputIds.Length
        let segLenI = Segment.value segLen
        let nSegments = if total = 0 then 1 else (total + segLenI - 1) / segLenI
        let paddedLen = nSegments * segLenI

        let idsPad: TokenId[] = Array.create paddedLen padId
        let maskPad = Array.create paddedLen false
        Array.blit enc.InputIds 0 idsPad 0 total
        Array.blit enc.AttentionMask 0 maskPad 0 total

        let lastStart = (nSegments - 1) * segLenI

        let lastChunkFull =
            seq { lastStart .. lastStart + segLenI - 1 }
            |> Seq.forall (fun i -> maskPad.[i])

        let hasLeftover = nSegments > 1 && not lastChunkFull

        if hasLeftover then
            let mainSegs = nSegments - 1
            let mainLen = mainSegs * segLenI
            let mainIds = Array.init mainLen (fun i -> i64 (TokenId.value idsPad.[i]))
            let mainMsk = Array.sub maskPad 0 mainLen
            let lftIds = Array.init segLenI (fun i -> i64 (TokenId.value idsPad.[mainLen + i]))
            let lftMsk = Array.sub maskPad mainLen segLenI

            { MainIds = mainIds
              MainMask = mainMsk
              SegmentCount = mainSegs
              SegLen = segLenI
              LeftoverIds = lftIds
              LeftoverMask = lftMsk
              LeftoverLen = segLenI }
        else
            let mainIds = Array.init paddedLen (fun i -> i64 (TokenId.value idsPad.[i]))
            let mainMsk = Array.copy maskPad

            { MainIds = mainIds
              MainMask = mainMsk
              SegmentCount = nSegments
              SegLen = segLenI
              LeftoverIds = [||]
              LeftoverMask = [||]
              LeftoverLen = 0 }

    // -----------------------------------------------------------------
    // Encoder forward — collapses [B, S, segLen] → [B*S, segLen] for the
    // RoBERTa call, reshapes back, then concatenates the leftover branch.
    // -----------------------------------------------------------------

    /// Runs the encoder + four projection MLPs and returns
    /// `(sMent, eMent, sCoref, eCoref, attentionMask)` where each rep tensor
    /// is `[1, L, F]` and the mask is `[1, L]` with
    /// `L = segments*segLen + leftoverLen`.
    let private encoderForward
        (model: FCorefModel)
        (seg: Segmented)
        (device: Device)
        : Tensor * Tensor * Tensor * Tensor * Tensor =
        // Main branch: shape [segments, segLen] for the encoder.
        let mainIds =
            torch
                .tensor(seg.MainIds, dtype = ScalarType.Int64, device = device)
                .view ([| i64 seg.SegmentCount; i64 seg.SegLen |])

        let mainMsk =
            torch
                .tensor(maskToI64 seg.MainMask, dtype = ScalarType.Int64, device = device)
                .view ([| i64 seg.SegmentCount; i64 seg.SegLen |])

        let mSm, mEm, mSc, mEc = model.ForwardReps(mainIds, mainMsk)
        // Reshape [segments, segLen, F] → [1, segments*segLen, F].
        let mainLen = i64 seg.SegmentCount * i64 seg.SegLen
        let f = mSm.shape.[2]
        let reshape (t: Tensor) = t.view ([| 1L; mainLen; f |])
        let sMentMain = reshape mSm
        let eMentMain = reshape mEm
        let sCorefMain = reshape mSc
        let eCorefMain = reshape mEc
        let mainMskFlat = mainMsk.view ([| 1L; mainLen |])

        if seg.LeftoverLen = 0 then
            sMentMain, eMentMain, sCorefMain, eCorefMain, mainMskFlat
        else
            let lftIds =
                torch
                    .tensor(seg.LeftoverIds, dtype = ScalarType.Int64, device = device)
                    .view ([| 1L; i64 seg.LeftoverLen |])

            let lftMsk =
                torch
                    .tensor(maskToI64 seg.LeftoverMask, dtype = ScalarType.Int64, device = device)
                    .view ([| 1L; i64 seg.LeftoverLen |])

            let lSm, lEm, lSc, lEc = model.ForwardReps(lftIds, lftMsk)
            let cat (a: Tensor) (b: Tensor) = torch.cat ([| a; b |], dim = 1L)
            cat sMentMain lSm, cat eMentMain lEm, cat sCorefMain lSc, cat eCorefMain lEc, cat mainMskFlat lftMsk

    // -----------------------------------------------------------------
    // Driver: full forward pass for one document.
    // -----------------------------------------------------------------

    let private deviceOf (model: FCorefModel) : Device =
        let p = model.parameters (true) |> Seq.head
        p.device

    let private predictOne (model: FCorefModel) (tokenizer: RobertaTokenizer) (text: string) : CorefPrediction =
        let device = deviceOf model
        let segLen = model.Config.CorefHead.MaxSegmentLen
        let maxSpan = model.Config.CorefHead.MaxSpanLength
        let topLambda = model.Config.CorefHead.TopLambda

        let enc = tokenizer.Encode text
        let seg = segment enc tokenizer.PadId segLen

        // Encoder + projection MLPs.
        let sMent, eMent, sCoref, eCoref, attnMask = encoderForward model seg device
        let seqLen = sMent.shape.[1]
        let f = sMent.shape.[2]

        // --- Mention logits + length mask --------------------------------
        let mentionLogits =
            CorefHead.computeMentionLogits sMent eMent model.MentionS2EClf model.MentionStartClf model.MentionEndClf

        let menMask =
            mentionMask seqLen maxSpan mentionLogits.dtype device
            |> fun m -> m.unsqueeze (0L) // [1, L, L]

        let mentionLogits = maskTensor mentionLogits menMask

        // --- Top-k pruning ---------------------------------------------
        // k = floor(actual_seq_len * top_lambda), per doc.
        let actualLen = attnMask.sum (dim = -1L) // [1]

        let kTensor =
            actualLen.to_type(ScalarType.Float32).mul(scalar topLambda).to_type (ScalarType.Int64) // [1]

        let kCpu = kTensor.cpu().data<int64> ()
        let maxK = toInt kCpu.[0L]
        // Guard against degenerate inputs.
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
            // Build a [1, maxK] mask of valid positions (here always all-1 since
            // batch=1 and k == maxK); the python code accommodates ragged docs
            // in a batch.
            let arangeK = torch.arange(i64 maxK, device = device).unsqueeze (0L)
            let spanMask = arangeK.lt(kTensor.unsqueeze (1L)).to_type (ScalarType.Int64) // [1, maxK]

            let invalidK =
                spanMask.mul(scalar -1.0).add(scalar 1.0).to_type(ScalarType.Int64).mul (scalar (seqLen * seqLen - 1L))

            let topkIdx = topkIdx.mul(spanMask).add (invalidK)
            // Sort ascending to recover document order.
            let struct (sortedIdx, _) = topkIdx.sort (dim = -1L, descending = false)

            let startIds = sortedIdx.div (scalar seqLen, torch.RoundingMode.floor)
            let endIds = sortedIdx.remainder (scalar seqLen)

            // Gather the [1, maxK] mention scalars, then form the [1, maxK, maxK]
            // additive matrix.
            let topkMentionScalars =
                mentionLogits.view([| 1L; seqLen * seqLen |]).gather (1L, sortedIdx) // [1, maxK]

            let topkMentionLogits =
                topkMentionScalars.unsqueeze (-1L) + topkMentionScalars.unsqueeze (-2L)

            // --- Coref reps gather -----------------------------------------
            let expandFor (ids: Tensor) =
                ids.unsqueeze(-1L).expand ([| 1L; i64 maxK; f |])

            let topkStartCoref = sCoref.gather (1L, expandFor startIds) // [1, maxK, F]
            let topkEndCoref = eCoref.gather (1L, expandFor endIds) // [1, maxK, F]

            // --- 4-direction antecedent bilinear sum -----------------------
            let bilinear (proj: Linear) (a: Tensor) (b: Tensor) : Tensor =
                proj.forward(a).bmm (b.transpose (-2L, -1L))

            let s2s = bilinear model.AntecedentS2S topkStartCoref topkStartCoref
            let e2e = bilinear model.AntecedentE2E topkEndCoref topkEndCoref
            let s2e = bilinear model.AntecedentS2E topkStartCoref topkEndCoref
            let e2s = bilinear model.AntecedentE2S topkEndCoref topkStartCoref
            let corefLogits = s2s + e2e + s2e + e2s // [1, maxK, maxK]

            // --- Combine + antecedent mask + null column -------------------
            let combined = topkMentionLogits + corefLogits

            let antecedentMask =
                torch.ones_like(combined).tril(-1L).mul (spanMask.to_type(combined.dtype).unsqueeze (-1L))

            let combined = maskTensor combined antecedentMask

            let zeros =
                torch.zeros ([| 1L; i64 maxK; 1L |], dtype = combined.dtype, device = device)

            let finalLogits = torch.cat ([| combined; zeros |], dim = -1L) // [1, maxK, maxK+1]

            let startArr = toIntArray (startIds.view ([| i64 maxK |]))
            let endArr = toIntArray (endIds.view ([| i64 maxK |]))
            let finalMat = toF32Matrix (finalLogits.squeeze (Nullable<int64>(0L)))

            let startTokArr = TokenIdx.ofArray startArr
            let endTokArr = TokenIdx.ofArray endArr

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
                // First occurrence wins if top-k ever yields duplicates.
                if not (spanToIndex.ContainsKey span) then
                    spanToIndex.[span] <- i

            // Build the char-keyed index up front so `CorefLogits.TryGetByCharSpan`
            // is O(1) without doing a runtime CharSpan -> TokenSpan inversion.
            // Spans whose `tryCharSpan` fails (special tokens, malformed) are
            // simply omitted from the dict — TryGet returns None for them.
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

    /// Run FCoref inference on a single input string. The model is switched
    /// to eval mode and the forward pass is wrapped in `torch.no_grad()`.
    let predict (model: FCorefModel) (tokenizer: RobertaTokenizer) (text: string) : CorefPrediction =
        model.eval ()
        use _ = torch.no_grad ()
        predictOne model tokenizer text

    /// Batch overload — runs each text independently. Documents are NOT
    /// batched along the encoder's first dim (FCoref's docs are typically
    /// long enough that padding-the-shorter-to-longer wastes too much
    /// compute); use this for convenience, not raw throughput.
    let predictBatch
        (model: FCorefModel)
        (tokenizer: RobertaTokenizer)
        (texts: IReadOnlyList<string>)
        : CorefPrediction[] =
        model.eval ()
        use _ = torch.no_grad ()
        texts |> Seq.map (predictOne model tokenizer) |> Array.ofSeq

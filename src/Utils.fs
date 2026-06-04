namespace FastCoref

// FS0042: this is the IL identity / retype pattern FSharp.UMX itself uses
// (and suppresses identically) for tag/untag. Sound here because units of
// measure are erased at runtime.
#nowarn "42"

open System
open System.IO
open System.Text.Json
open TorchSharp

/// Reinterpret a primitive array as the same array with a unit of measure attached.
/// Units of measure are erased at runtime, so the input and output share the
/// same `System.Int32[]` (resp. `Single[]`) — no allocation, no element walk.
[<RequireQualifiedAccess>]
module UMXArr =
    let inline tag<[<Measure>] 'm> (xs: int[]) : int<'m>[] = (# "" xs : int<'m>[] #)
    let inline tagF32<[<Measure>] 'm> (xs: float32[]) : float32<'m>[] = (# "" xs : float32<'m>[] #)

/// Tensor / IO / JSON helpers shared across the FastCoref pipeline.
module Utils =

    // Cast helpers MUST be defined before `open type torch`, otherwise
    // `int64` / `float32` get shadowed by `torch.int64` / `torch.float32`
    // (which are ScalarType values, not conversion functions).
    let inline i64 (x: ^a) : int64 = int64 x
    let inline f32 (x: ^a) : float32 = float32 x
    let inline toInt (x: ^a) : int = int x
    let inline sqrtf (x: float32) : float32 = sqrt x |> float32

    open type TorchSharp.torch

    // --- Debug helpers --------------------------------------------------

    /// Logs shape, dtype and the first few values to stderr; returns the
    /// tensor unchanged so it can be spliced into a pipeline.
    let peek (name: string) (t: Tensor) : Tensor =
        let shapeStr = t.shape |> Array.map string |> String.concat ","
        let total = t.shape |> Array.fold (fun acc d -> acc * d) 1L
        let n = if total < 8L then total else 8L

        let preview =
            if total = 0L then
                "<empty>"
            else
                t.reshape(-1L).narrow(0L, 0L, n).to_type(ScalarType.Float32).data<float32> ()
                |> Seq.map (sprintf "%g")
                |> String.concat ","

        eprintfn "%s: shape=[%s] dtype=%A device=%s first=[%s]" name shapeStr t.dtype (t.device.ToString()) preview
        t

    let logShape (name: string) (t: Tensor) : unit =
        let shapeStr = t.shape |> Array.map string |> String.concat ","
        eprintfn "%s: shape=[%s] dtype=%A" name shapeStr t.dtype

    // --- Mask builders --------------------------------------------------

    /// `[n,n]` bool mask with `mask[i,j] = j > i`.
    let triuMask (n: int64) (device: Device) : Tensor =
        torch.ones([| n; n |], dtype = ScalarType.Bool, device = device).triu (1L)

    /// `[seqLen, seqLen]` bool mask with
    /// `mask[s,e] = (e >= s) && (e - s < maxSpanLen)`.
    let spanLengthMask (seqLen: int64) (maxSpanLen: int) (device: Device) : Tensor =
        let idx = torch.arange (seqLen, device = device)
        let starts = idx.unsqueeze (1L)
        let ends = idx.unsqueeze (0L)
        let diff = ends - starts
        (diff.ge (Scalar.op_Implicit 0L)).logical_and (diff.lt (Scalar.op_Implicit (i64 maxSpanLen)))

    /// Out-of-place masked-fill: writes `value` where `mask` is true.
    let batchMaskedFill (t: Tensor) (mask: Tensor) (value: float) : Tensor =
        t.masked_fill (mask, Scalar.op_Implicit value)

    // --- Sequence / sort helpers ---------------------------------------

    /// Pad with `padId` or truncate-from-end so the result has exactly `length` ids.
    let padToLength (ids: int[]) (length: int) (padId: int) : int[] =
        if ids.Length = length then
            Array.copy ids
        elif ids.Length > length then
            ids.[0 .. length - 1]
        else
            let out = Array.create length padId
            Array.blit ids 0 out 0 ids.Length
            out

    let argsortDescending (t: Tensor) (dim: int) : Tensor =
        torch.argsort (t, i64 dim, descending = true)

    /// Mirrors fastcoref's numerically-safer softmax: cast to fp32, softmax,
    /// then cast back to the input dtype.
    let safeSoftmaxLastDim (t: Tensor) : Tensor =
        t.``to``(ScalarType.Float32).softmax(-1L).``to`` (t.dtype)

    // --- JSON / FS helpers ---------------------------------------------

    /// Parses a JSON file and returns a cloned root element (so the caller
    /// owns it independently of the underlying JsonDocument).
    let readJsonFile (path: string) : JsonElement =
        use stream = File.OpenRead(path)
        use doc = JsonDocument.Parse(stream)
        doc.RootElement.Clone()

    let modelFile (modelDir: string) (filename: string) : string =
        if not (Directory.Exists modelDir) then
            raise (DirectoryNotFoundException(sprintf "Model directory not found: '%s'" modelDir))

        Path.Combine(modelDir, filename)

    /// Canonical HuggingFace snapshot file names. Centralised so the
    /// model / config / tokenizer loaders all agree on the on-disk layout.
    [<RequireQualifiedAccess>]
    module HfFiles =
        [<Literal>]
        let Weights = "pytorch_model.bin"

        [<Literal>]
        let Config = "config.json"

        [<Literal>]
        let Vocab = "vocab.json"

        [<Literal>]
        let Merges = "merges.txt"

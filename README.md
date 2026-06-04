# FastCoref.fs

> Coreference resolution for English in pure .NET — F# port of the Python [`fastcoref`](https://github.com/shon-otmazgin/fastcoref) library (Otmazgin, Cattan, Goldberg — AACL 2022).

`FastCoref.fs` runs the two BIU-NLP coreference checkpoints (`f-coref` and `lingmess-coref`) directly from .NET, with no Python runtime at inference time. The original PyTorch weights are loaded as-is via [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge) — there is no conversion step, no ONNX export, no intermediate format. A single `CorefModel` class parameterised by `CorefKind` (`FCoref` | `LingMess`) drives both checkpoints; `Predict` / `PredictBatch` return `CorefResult`s with character-level cluster spans so existing Python recipes translate over with minimal effort.

## Features

- **FCoref** (RoBERTa-6L, ~90 M params) — fast checkpoint for high-throughput pipelines.
- **LingMess** (Longformer-large, 24 layers, ~590 M params) — higher quality with the 7-expert antecedent scorer.
- Loads original `biu-nlp/*` PyTorch weights directly via `TorchSharp.PyBridge` — no conversion step.
- Pure .NET: no Python runtime, no `transformers` required at inference time.
- Both raw-string and pre-tokenized (`IReadOnlyList<string>`) input shapes; outputs carry char- or word-indexed cluster spans accordingly.
- Antecedent logit lookups (`Logits.TryGet(spanI, spanJ)`) for callers that want the underlying mention-pair scores.
- JSON / JSONL export compatible with the Python `fastcoref` `text_idx` schema.
- Pronoun-style text resolution (`CorefResult.ResolveText()`) — verbatim head substitution (no POS / gender agreement).
- xUnit test suite gated on `FASTCOREF_MODELS_DIR` so CI without GPU/checkpoints stays green.

## Installation

No NuGet package is published yet — build from source.

```bash
git clone <repo-url> FastCoref.fs
cd FastCoref.fs
./build.fsx -t Setup        # dotnet restore + download both checkpoints
./build.fsx -t Build        # release build via the slnx
```

Requires the **.NET 10 SDK** (the project sets `LangVersion=preview`).

To reference the library from your own project, add a `ProjectReference` to `src/FastCoref.fs.fsproj`:

```xml
<ProjectReference Include="path/to/FastCoref.fs/src/FastCoref.fs.fsproj" />
```

### Build script targets

All repo workflows go through the FAKE 5 `build.fsx` at the repo root. Pass any target with `-t`; `./build.fsx --list` enumerates everything.

| Target | What it does |
|---|---|
| `Setup` | `Restore` + `DownloadModels`. The one-shot post-clone target. |
| `Restore` / `Build` | `dotnet restore` / `dotnet build -c Release` on `FastCoref.fs.slnx`. |
| `Tests` | `dotnet test` on the test project; auto-defaults `FASTCOREF_MODELS_DIR` to `~/.cache/fastcoref` when unset and that directory exists. |
| `Package` | `dotnet pack` → `src/bin/Release/FastCoref.fs.<version>.nupkg`. |
| `DownloadModels` | Download both checkpoints (see below). |
| `DownloadFCoref` / `DownloadLingMess` | Download a single checkpoint. |
| `CheckModels` | Verify both checkpoints are present; list any missing files. |
| `GenerateDocs` / `WatchDocs` | Build (or live-rebuild) the `fsdocs` site under `output/`. |
| `Clean` | Wipe `bin/`, `obj/`, `output/`, `.fsdocs/`. |
| `All` | `Clean` → `Restore` → `Meta` → `Build` → `Tests` → `Package` → `GenerateDocs`. Default target. |
| `PublishNuget` / `ReleaseDocs` / `Release` | Release-time targets; need `nugetkey` and a `gh-pages` branch respectively. |

## Downloading the models

The library does **not** download checkpoints at runtime. Pull them via the build script:

```bash
./build.fsx -t DownloadModels      # both models
./build.fsx -t DownloadFCoref      # just FCoref   (~360 MB)
./build.fsx -t DownloadLingMess    # just LingMess (~2.4 GB)
./build.fsx -t CheckModels         # verify the layout, list anything missing
```

This shells out to `huggingface-cli download` and drops the files under `$FASTCOREF_MODELS_DIR` (defaults to `~/.cache/fastcoref`). It expects `huggingface-cli` on `PATH`; install once with `pip install -U "huggingface_hub[cli]"` if you don't have it.

Each model directory ends up with:

- `config.json`
- `pytorch_model.bin`
- `vocab.json`
- `merges.txt`

## Quick start

```fsharp
open FastCoref.Api

let modelDir = "/path/to/biu-nlp/f-coref"

use coref = new CorefModel(modelDir, CorefKind.FCoref)

let result = coref.Predict "Alice goes down the rabbit hole. There she finds a new world."

for cluster in result.GetClustersAsStrings() do
    printfn "Cluster: %s" (String.concat ", " cluster)

// e.g.
// Cluster: Alice, she
// Cluster: the rabbit hole, a new world
```

`Clusters` returned by `Predict` are typed `Cluster<Mention>` values with a non-empty head and a tail of additional mentions. Each `Mention` carries a `CharSpan` (`Start` / `End` of `CharIdx` — a UMX-tagged `int<chr>` for UTF-16 code-unit offsets into the input) and the materialised substring as `Text`. `GetClustersAsStrings()` is a convenience that returns each cluster as a `string list`.

## Using LingMess (higher quality, larger model)

The wrapper is the same `CorefModel`; only the `CorefKind` changes:

```fsharp
open FastCoref.Api
open FastCoref.Clustering

let modelDir = "/path/to/biu-nlp/lingmess-coref"

use coref = new CorefModel(modelDir, CorefKind.LingMess)

let result = coref.Predict "John saw Mary at the park. He waved at her."

for cluster in result.Clusters do
    for mention in Cluster.toList cluster do
        let s = CharIdx.value mention.Span.Start
        let e = CharIdx.value mention.Span.End
        printfn "  [%d..%d] = %s" s e mention.Text
```

The constructor cross-checks `CorefKind` against the checkpoint's `model_type` and throws `ArgumentException` on mismatch (e.g. asking for `CorefKind.FCoref` against a LingMess directory).

LingMess is ~6× larger and noticeably slower than FCoref; reach for it when accuracy on hard cases (linguistically distinct antecedent classes) matters more than throughput.

## Batched inference

`PredictBatch` runs sequentially and preserves input order, returning a `CorefResult[]`:

```fsharp
let texts =
    [ "Alice goes down the rabbit hole. There she finds a new world."
      "The car broke down on the highway. It was very old."
      "John saw Mary at the park. He waved at her." ]

let results = coref.PredictBatch texts
for r in results do
    printfn "%s" r.Text
    for c in r.GetClustersAsStrings() do
        printfn "  %A" c
```

## Pre-tokenized input

If you already have a word array (e.g. from your own tokenizer or treebank), use the `Predict(words)` overload — it returns a `WordCorefResult` whose `Clusters` carry `WordSpan`s (word-index `[Start..End)` half-open spans) instead of character spans:

```fsharp
open FastCoref.Api

use coref = new CorefModel(modelDir, CorefKind.FCoref)

let words = [| "Alice"; "walked"; "into"; "the"; "room"; "."; "She"; "sat"; "down"; "." |]
let result = coref.Predict words

for cluster in result.Clusters do
    for m in Cluster.toList cluster do
        let s = WordIdx.value m.Span.Start
        let e = WordIdx.value m.Span.End
        printfn "  [%d..%d) = %s" s e m.Text

// Any mentions whose char span didn't land cleanly on word boundaries
// (e.g. contractions split by the BPE tokenizer) surface here:
for unaligned in result.UnalignedMentions do
    printfn "unaligned: %s" unaligned.Text
```

Internally the words are joined with single ASCII spaces and the regular `Predict(string)` path runs; mentions are then projected back to word indices. **This is not strict HuggingFace `is_split_into_words` parity** — punctuation and contractions may be tokenized differently than in a true pre-tokenized BPE path. For strict semantics, call `Predict(text: string)` directly.

Validation is strict: empty / null words, or words containing any whitespace, throw `ArgumentException`.

`PredictBatch(documents: IReadOnlyList<IReadOnlyList<string>>)` is the batched equivalent.

## Antecedent logit lookups

Each `CorefResult` carries a `Logits: CorefLogits` field with the antecedent-score matrix produced during inference. Look up the logit for a `(mention, antecedent)` pair via either span flavour:

```fsharp
open FastCoref.Api
open FastCoref.Clustering

let result = coref.Predict "Alice walked home. She sat down."

match result.Clusters with
| cluster :: _ when not cluster.Rest.IsEmpty ->
    let head = cluster.Head
    let antecedent = cluster.Rest.Head
    match result.Logits.TryGet(head.Span, antecedent.Span) with
    | Some score -> printfn "logit %s <- %s = %f" head.Text antecedent.Text score
    | None       -> printfn "no logit (self-link, span unknown, or released)"
| _ -> ()
```

`TryGet` returns `None` for self-links (`i == j`), spans that don't match a pruned mention, or after the matrix has been released. Word-flavoured callers (results from `Predict(words)`) can call `result.Logits.TryGet(wordSpan1, wordSpan2)` against the same `Logits` value — both overloads share the same underlying matrix.

Call `result.Logits.Release()` to drop the matrix when you're done — useful for batch inference where matrices add up.

## Text resolution

`CorefResult.ResolveText()` substitutes every non-head mention with its cluster's head text:

```fsharp
let result = coref.Predict "Alice walked into the room. She sat down."
printfn "%s" (result.ResolveText())
// "Alice walked into the room. Alice sat down."
```

The substitution is verbatim: no POS tagging, no pronoun-case agreement, no gender matching — the head text is just spliced in. Overlapping mentions are resolved leftmost-longest-wins; trailing mentions inside an already-substituted span are skipped. For pronoun-aware resolution, post-process this output with your own NLP tooling.

## JSON / JSONL export

`CorefResult.ToJson()` serialises one result to a single-line JSON string compatible with the Python `fastcoref` `text_idx` schema (char-level, not word-level):

```fsharp
let result = coref.Predict "Alice walked home. She sat down."
printfn "%s" (result.ToJson())
// {"text":"Alice walked home. She sat down.","text_idx":0,
//  "clusters":[[[0,5],[19,22]]],
//  "clusters_strings":[["Alice","She"]]}
```

`CorefModel.PredictToJsonl(texts, outputPath)` batches over an `IReadOnlyList<string>` and writes one JSON object per result, one per line, with each result's `text_idx` matching its position in the input list:

```fsharp
coref.PredictToJsonl([| "Alice walked home. She sat down."
                        "John saw Mary at the park. He waved at her." |],
                     "results.jsonl")
```

## GPU (experimental)

`CorefModel` accepts an optional `device` argument; the default is `torch.CPU`.

```fsharp
open FastCoref.Api
open TorchSharp

if torch.cuda.is_available () then
    use coref = new CorefModel(modelDir, CorefKind.FCoref, torch.CUDA)
    let result = coref.Predict "..."
    printfn "ran on %A" coref.Device
```

The library project itself references the CPU-only `TorchSharp-cpu` runtime so it stays portable. To actually execute on CUDA, **your application's** `.fsproj` (not this library's) must pull in the matching native package — `TorchSharp-cuda-windows` or `TorchSharp-cuda-linux` — so the right `libtorch` flavour is on the load path.

The library does **not** set a default dtype on your behalf. Model construction respects the current `torch` default dtype; to mirror the upstream Python configuration (`torch_dtype=torch.bfloat16`) call `torch.set_default_dtype(torch.bfloat16)` yourself **before** constructing `CorefModel`. GPU support is currently best-effort and not part of the routinely-exercised test matrix.

## API reference

The public surface lives in `module FastCoref.Api`; the typed span / cluster vocabulary lives in `module FastCoref.Clustering`.

### `module FastCoref.Clustering`

```fsharp
type TokenIdx = int<tok>      // UMX-tagged token *position* in the encoder sequence

[<RequireQualifiedAccess>]
module TokenIdx =
    val ofInt : int -> TokenIdx
    val value : TokenIdx -> int

[<Struct>] type TokenSpan = { Start: TokenIdx; End: TokenIdx }   // inclusive

/// Non-empty coreference cluster: `Head` mention + `Rest` of mentions.
type Cluster<'span> = { Head: 'span; Rest: 'span list }

module Cluster =
    val toList : Cluster<'a> -> 'a list
    val length : Cluster<'a> -> int
    val map : ('a -> 'b) -> Cluster<'a> -> Cluster<'b>
```

### `module FastCoref.Api` types

```fsharp
type CharIdx = int<chr>       // UMX-tagged UTF-16 code-unit offset into the input
type WordIdx = int<wrd>       // UMX-tagged word-index into a pre-tokenized input

[<RequireQualifiedAccess>]
module CharIdx =
    val ofInt : int -> CharIdx
    val value : CharIdx -> int

[<RequireQualifiedAccess>]
module WordIdx =
    val ofInt : int -> WordIdx
    val value : WordIdx -> int

type CharSpan = { Start: CharIdx; End: CharIdx }   // half-open [Start..End)
type WordSpan = { Start: WordIdx; End: WordIdx }   // half-open

type Mention     = { Span: CharSpan; Text: string }
type WordMention = { Span: WordSpan; Text: string }

type CorefResult = {
    Text: string
    Clusters: Cluster<Mention> list
    Logits: CorefLogits
} with
    member GetClustersAsStrings : unit -> string list list
    member ResolveText          : unit -> string
    member ToJson               : unit -> string

type WordCorefResult = {
    Words: string[]
    Clusters: Cluster<WordMention> list
    UnalignedMentions: Mention list      // char mentions that didn't align on word boundaries
    Logits: CorefLogits                   // same matrix as the underlying char-level CorefResult
}

[<RequireQualifiedAccess>]
type CorefKind = FCoref | LingMess

// Logit lookups — typed convenience overloads in module FastCoref.Api,
// underlying opaque type in FastCoref.Clustering
type Clustering.CorefLogits with
    member TryGet : CharSpan * CharSpan -> float32 option
    member TryGet : WordSpan * WordSpan -> float32 option

// In FastCoref.Clustering:
type CorefLogits =
    member HasMatrix : bool
    member Release   : unit -> unit
module CorefLogits =
    val empty : CorefLogits
```

- `Text` — the original input string passed to `Predict`.
- `Clusters` — one `Cluster<Mention>` per coreference cluster; each `Mention` carries `Span: CharSpan` (UTF-16 offsets, end exclusive) and the materialised `Text`.
- `GetClustersAsStrings()` — extracts each mention as a substring; equivalent to Python `CorefResult.get_clusters(as_strings=True)`.
- `ResolveText()` — substitutes every non-head mention with the cluster head's text; leftmost-longest-wins on overlap; no POS / pronoun-case agreement.
- `ToJson()` — single-line JSON in the Python `fastcoref` `text_idx` schema (`text`, `text_idx`, `clusters`, `clusters_strings`).
- `Logits.TryGet(spanI, spanJ)` — antecedent logit lookup; `None` for self-link, unknown span, or released matrix.

### `type CorefModel(modelDir, kind [, device]) : IDisposable`

```fsharp
member Predict        : string                                          -> CorefResult
member Predict        : IReadOnlyList<string>                           -> WordCorefResult
member PredictBatch   : IReadOnlyList<string>                           -> CorefResult[]
member PredictBatch   : IReadOnlyList<IReadOnlyList<string>>            -> WordCorefResult[]
member PredictToJsonl : IReadOnlyList<string> * outputPath: string      -> unit
member LoadReport     : Config.LoadReport
member Device         : torch.Device
member Kind           : CorefKind
interface IDisposable
```

- `Predict(string)` / `PredictBatch(IReadOnlyList<string>)` — run coreference resolution; clusters carry character-domain spans.
- `Predict(words)` / `PredictBatch(documents)` — pre-tokenized convenience that joins with single ASCII spaces and projects spans back to word indices. Throws `ArgumentException` on null / empty words or words containing whitespace. Mentions that don't align on word boundaries surface in `WordCorefResult.UnalignedMentions` rather than being silently dropped.
- `PredictToJsonl(texts, path)` — `PredictBatch(texts)` plus one-line-per-result JSON output to `path`; each line carries its position as `text_idx`.
- `LoadReport` — `{ Loaded; Missing; Total }` counts of how many parameters were overwritten from the checkpoint vs kept their random initialisation; useful for catching `state_dict` key mismatches after a checkpoint update.
- `Device` — the TorchSharp device the underlying module is on.
- `Kind` — echoes the `CorefKind` the model was constructed with.
- `Dispose()` — releases the underlying TorchSharp model (and any GPU memory). Safe to call multiple times. Always wrap construction in `use` or call `Dispose()` explicitly.

The constructor throws `ArgumentException` if `kind` disagrees with the checkpoint's `model_type` (e.g. `CorefKind.FCoref` against a LingMess directory).

### Current limitations

- No sentence segmentation — long documents are passed to the encoder as a single sequence and rely on its built-in long-range attention (Longformer for LingMess) or per-window inference (FCoref).
- `ResolveText()` does verbatim head substitution: no POS tagging, no pronoun-case agreement, no gender matching. Callers that need pronoun-aware substitution should post-process the output with their own NLP tooling.
- `Predict(words)` is not strict HuggingFace `is_split_into_words` parity (see "Pre-tokenized input" above) — it's a convenience over the string path.

## Running the tests

```bash
./build.fsx -t Tests
```

`Tests` auto-defaults `FASTCOREF_MODELS_DIR` to `~/.cache/fastcoref` when the env var is unset and that directory exists, so the gated tests run end-to-end as soon as you've executed `./build.fsx -t Setup` (or `DownloadModels`). To point the suite at a different cache, set `FASTCOREF_MODELS_DIR` before invoking the target — it must contain `f-coref/` and `lingmess-coref/` subfolders matching the layout produced by `DownloadModels`.

When `FASTCOREF_MODELS_DIR` is **not** set and the default cache is absent, the model-loading tests short-circuit to a silent pass — pure-function tests (tokenizer, clustering, ResolveText, JSON export, CorefLogits) still run. With it set, the suite loads the real checkpoints and exercises the full forward pass end-to-end.

## Architecture (brief)

The library is intentionally a thin set of small modules; everything is in `src/`:

| File | Responsibility |
|---|---|
| `Utils.fs` | Shared tensor / collection helpers. |
| `Config.fs` | `config.json` loader for both checkpoints. |
| `Tokenizer.fs` | RoBERTa BPE tokenizer (`Microsoft.ML.Tokenizers`). |
| `Roberta.fs` | RoBERTa encoder (used by FCoref). |
| `Longformer.fs` | Longformer-large encoder (used by LingMess). |
| `CorefHead.fs` | Span scorer / antecedent head shared by both models. |
| `Clustering.fs` | Greedy clustering of antecedent predictions. |
| `FCorefModel.fs` | FCoref module: encoder + head + `load`. |
| `LingMessModel.fs` | LingMess module: encoder + 7-expert head + `load`. |
| `FCorefInference.fs` | End-to-end `predict` / `predictBatch` for FCoref. |
| `LingMessInference.fs` | End-to-end `predict` / `predictBatch` for LingMess. |
| `Api.fs` | Public `FastCoref.Api` surface (`CorefModel`, `CorefKind`, `CorefResult`, `WordCorefResult`, `CharIdx`/`WordIdx`, `Mention`, JSON export, ResolveText, typed `CorefLogits.TryGet` overloads). |

Only `Api.fs` is part of the stable public API; the rest is subject to change.

## Acknowledgements

- The original Python implementation: [shon-otmazgin/fastcoref](https://github.com/shon-otmazgin/fastcoref).
- Otmazgin, Cattan, Goldberg. **"LingMess: Linguistically Informed Multi-Expert Scorers for Coreference Resolution."** AACL 2022.
- [TorchSharp](https://github.com/dotnet/TorchSharp) and [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge) — without the PyBridge `load_py` path this port would have needed a full weights-conversion pipeline.

## License

[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

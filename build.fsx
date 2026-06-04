#!/usr/bin/env -S dotnet fsi
#r "nuget: Fake.Core.Target, 5.23.1"
#r "nuget: Fake.IO.FileSystem, 5.23.1"
#r "nuget: Fake.DotNet.Cli, 5.23.1"
#r "nuget: Fake.Core.ReleaseNotes, 5.23.1"
#r "nuget: Fake.Tools.Git, 5.23.1"
#r "nuget: MSBuild.StructuredLogger, 2.2.441"

open System
open System.IO

open Fake.Core
open Fake.Core.TargetOperators
open Fake.DotNet
open Fake.IO
open Fake.IO.FileSystemOperators
open Fake.IO.Globbing.Operators
open Fake.Tools.Git

System.Environment.GetCommandLineArgs()
|> Array.skip 2 // fsi.exe; build.fsx
|> Array.toList
|> Context.FakeExecutionContext.Create false __SOURCE_FILE__
|> Context.RuntimeContext.Fake
|> Context.setExecutionContext

let gitOwner = "et1975"
let gitHome = "https://github.com/" + gitOwner

let gitName = "FastCoref.fs"

let gitRaw = Environment.environVarOrDefault "gitRaw" ("https://raw.github.com/" + gitOwner + "/" + gitName)

let solution = "FastCoref.fs.slnx"
let srcProject = "src/FastCoref.fs.fsproj"
let testProject = "tests/FastCoref.fs.Tests.fsproj"
let packageId = "FastCoref.fs"

let release = ReleaseNotes.load "RELEASE_NOTES.md"

let defaultModelsDir =
    Path.Combine(Environment.environVar "HOME", ".cache", "fastcoref")

let modelsDir =
    Environment.environVarOrDefault "FASTCOREF_MODELS_DIR" defaultModelsDir

let modelSubdirs = [
    "f-coref",        "biu-nlp/f-coref"
    "lingmess-coref", "biu-nlp/lingmess-coref"
]

let requiredModelFiles = [
    "config.json"
    "pytorch_model.bin"
    "vocab.json"
    "merges.txt"
]

module ProcessResult =
    let assertSuccess (r: ProcessResult) =
        if (not r.OK) then failwithf "Error while executing process: %A" r

let private missingModelFiles (subdir: string) =
    let dir = modelsDir @@ subdir
    if not (Directory.Exists dir) then List.map (fun f -> dir @@ f) requiredModelFiles
    else
        requiredModelFiles
        |> List.map (fun f -> dir @@ f)
        |> List.filter (File.exists >> not)

let private downloadModel (subdir: string) (repo: string) =
    let cli =
        match ProcessUtils.tryFindFileOnPath "huggingface-cli" with
        | Some p -> p
        | None ->
            failwith "huggingface-cli not found on PATH. Install with: pip install -U 'huggingface_hub[cli]'"
    let targetDir = modelsDir @@ subdir
    Directory.ensure targetDir
    Trace.tracefn "Downloading %s -> %s" repo targetDir
    let exitCode =
        Shell.Exec(
            cli,
            sprintf "download %s --local-dir %s" repo targetDir)
    if exitCode <> 0 then
        failwithf "huggingface-cli exited with %d while downloading %s" exitCode repo
    match missingModelFiles subdir with
    | [] -> Trace.tracefn "OK: %s has all required files" targetDir
    | missing ->
        failwithf
            "Download completed but required files are missing in %s:\n  %s"
            targetDir
            (String.Join("\n  ", missing))

Target.create "Clean" (fun _ ->
    !! "./**/bin"
    ++ "./**/obj"
    ++ "./docs/output"
    ++ "./.fsdocs"
    ++ "./output"
    |> Seq.iter Shell.cleanDir
)

Target.create "Meta" (fun _ ->
    [ "<Project xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">"
      "<ItemGroup>"
      """<None Include="$(MSBuildThisFileDirectory)/LICENSE.md" Pack="true" PackagePath="\" />"""
      """<None Include="$(MSBuildThisFileDirectory)/README.md" Pack="true" PackagePath="\"/>"""
      """<PackageReference Include="Microsoft.SourceLink.GitHub" Version="1.0.0" PrivateAssets="All"/>"""
      "</ItemGroup>"
      "<PropertyGroup>"
      sprintf "<PackageProjectUrl>https://github.com/%s/%s</PackageProjectUrl>" gitOwner gitName
      "<PackageLicenseFile>LICENSE.md</PackageLicenseFile>"
      "<PackageReadmeFile>README.md</PackageReadmeFile>"
      sprintf "<RepositoryUrl>https://github.com/%s/%s.git</RepositoryUrl>" gitOwner gitName
      "<PackageTags>coref;coreference;nlp;fsharp;torchsharp;dotnet</PackageTags>"
      "<PackageDescription>Coreference resolution for English in pure .NET — F# port of the Python fastcoref library running on TorchSharp.</PackageDescription>"
      "<Authors>Eugene Tolmachev</Authors>"
      sprintf "<PackageReleaseNotes>%s</PackageReleaseNotes>" (List.head release.Notes |> System.Web.HttpUtility.HtmlEncode)
      sprintf "<Version>%s</Version>" (string release.SemVer)
      sprintf "<FsDocsLicenseLink>https://github.com/%s/%s/blob/main/LICENSE.md</FsDocsLicenseLink>" gitOwner gitName
      sprintf "<FsDocsReleaseNotesLink>https://github.com/%s/%s/blob/main/RELEASE_NOTES.md</FsDocsReleaseNotesLink>" gitOwner gitName
      "</PropertyGroup>"
      "</Project>"]
    |> File.write false "Directory.Build.props"
)

Target.create "Restore" (fun _ ->
    DotNet.restore id solution
)

Target.create "Build" (fun _ ->
    DotNet.build (fun opt -> { opt with Configuration = DotNet.BuildConfiguration.Release }) solution
)

Target.create "Tests" (fun _ ->
    if not (Environment.hasEnvironVar "FASTCOREF_MODELS_DIR") && Directory.Exists defaultModelsDir then
        Trace.tracefn "FASTCOREF_MODELS_DIR not set; defaulting to %s" defaultModelsDir
        System.Environment.SetEnvironmentVariable("FASTCOREF_MODELS_DIR", defaultModelsDir)
    let args = "--no-restore"
    DotNet.test (fun a -> a.WithCommon (fun c -> { c with CustomParams = Some args })) testProject
)

Target.create "DownloadFCoref" (fun _ ->
    let subdir, repo = modelSubdirs |> List.find (fst >> (=) "f-coref")
    downloadModel subdir repo
)

Target.create "DownloadLingMess" (fun _ ->
    let subdir, repo = modelSubdirs |> List.find (fst >> (=) "lingmess-coref")
    downloadModel subdir repo
)

Target.create "DownloadModels" ignore

Target.create "CheckModels" (fun _ ->
    let report =
        modelSubdirs
        |> List.map (fun (subdir, repo) -> subdir, repo, missingModelFiles subdir)
    for (subdir, repo, missing) in report do
        match missing with
        | [] -> Trace.tracefn "OK: %s (%s) — all required files present" subdir repo
        | xs -> Trace.traceErrorfn "MISSING in %s (%s):\n  %s" subdir repo (String.Join("\n  ", xs))
    let anyMissing = report |> List.exists (fun (_, _, m) -> not (List.isEmpty m))
    if anyMissing then
        failwithf
            "One or more models incomplete under %s. Run: ./build.fsx -t DownloadModels"
            modelsDir
)

Target.create "Setup" ignore

Target.create "Package" (fun _ ->
    let args = sprintf "/p:Version=%s --no-restore" (string release.SemVer)
    DotNet.pack (fun a -> a.WithCommon (fun c -> { c with CustomParams = Some args })) srcProject
)

Target.create "PublishNuget" (fun _ ->
    let exec dir = DotNet.exec (fun a -> a.WithCommon (fun c -> { c with WorkingDirectory = dir }))
    [ exec "src" "nuget" (sprintf "push bin/Release/%s.%s.nupkg -s nuget.org -k %s" packageId release.NugetVersion (Environment.environVar "nugetkey")) ]
    |> List.iter ProcessResult.assertSuccess
)

// --------------------------------------------------------------------------------------
// Generate the documentation

let fsdocProperties = [
    "Configuration=Release"
    "TargetFramework=net10.0"
]

Target.create "GenerateDocs" (fun _ ->
    Shell.cleanDir ".fsdocs"
    DotNet.exec id "fsdocs" ("build --strict --eval --clean"
        + " --projects " + srcProject
        + " --properties " + String.Join(" ", fsdocProperties))
    |> ProcessResult.assertSuccess
    File.writeString false ("output" @@ "index.html")
        """<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=content/index.html"></head><body></body></html>"""
)

Target.create "WatchDocs" (fun _ ->
    Shell.cleanDir ".fsdocs"
    DotNet.exec id "fsdocs" ("watch --eval"
        + " --projects " + srcProject
        + " --properties " + String.Join(" ", fsdocProperties)) |> ignore
)

Target.create "ReleaseDocs" (fun _ ->
    let tempDocsDir = "temp/gh-pages"
    Shell.cleanDir tempDocsDir
    Repository.cloneSingleBranch "" (sprintf "git@github.com:%s/%s.git" gitOwner gitName) "gh-pages" tempDocsDir

    Repository.fullclean tempDocsDir
    Shell.copyRecursive "output" tempDocsDir true |> Trace.tracefn "%A"
    Staging.stageAll tempDocsDir
    Commit.exec tempDocsDir (sprintf "Update generated documentation for version %s" (string release.SemVer))
    Branches.push tempDocsDir
)

Target.create "All" ignore
Target.create "Release" ignore

"DownloadFCoref"   ==> "DownloadModels"
"DownloadLingMess" ==> "DownloadModels"

"Restore"        ==> "Setup"
"DownloadModels" ==> "Setup"

"All"
  <== ["Clean"; "Restore"; "Meta"; "Build"; "Tests"; "Package"; "GenerateDocs"]

"Build"
  ==> "Tests"

"Build"
  ==> "GenerateDocs"

"Meta"
  ==> "Build"
  ==> "Package"
  ==> "PublishNuget"

"Release"
  <== ["All"; "PublishNuget"; "ReleaseDocs"]

Target.runOrDefault "All"

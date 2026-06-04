module FastCoref.Tests.SmokeTests

open Xunit
open Swensen.Unquote
open TorchSharp
open FastCoref

[<Fact>]
let ``padToLength pads with padId when shorter`` () =
    let result = Utils.padToLength [| 1; 2; 3 |] 6 0
    test <@ result = [| 1; 2; 3; 0; 0; 0 |] @>

[<Fact>]
let ``padToLength truncates from the end when longer`` () =
    let result = Utils.padToLength [| 1; 2; 3; 4; 5 |] 3 0
    test <@ result = [| 1; 2; 3 |] @>

[<Fact>]
let ``padToLength returns a copy when length matches`` () =
    let input = [| 7; 8; 9 |]
    let result = Utils.padToLength input 3 0
    test <@ result = input @>
    test <@ not (obj.ReferenceEquals(result, input)) @>

[<Fact>]
let ``padToLength pads an empty array`` () =
    let result = Utils.padToLength [||] 4 -1
    test <@ result = [| -1; -1; -1; -1 |] @>

[<Fact(Skip = "TorchSharp 0.101.5 testhost hangs on this environment; run manually via a console host")>]
let ``spanLengthMask has correct shape and respects window`` () =
    let mask = Utils.spanLengthMask 5L 3 (torch.CPU)
    let shape = mask.shape
    test <@ shape = [| 5L; 5L |] @>
    test <@ mask.dtype = torch.ScalarType.Bool @>

    let bools = mask.data<bool> () |> Array.ofSeq
    let allowed s e = e >= s && e - s < 3

    let expected =
        [| for s in 0..4 do
               for e in 0..4 do
                   yield allowed s e |]

    test <@ bools = expected @>

[<Fact(Skip = "TorchSharp 0.101.5 testhost hangs on this environment; run manually via a console host")>]
let ``spanLengthMask diagonal entries are all true`` () =
    let mask = Utils.spanLengthMask 4L 2 (torch.CPU)

    for i in 0L .. 3L do
        let v = mask.[i, i].item<bool> ()
        test <@ v = true @>

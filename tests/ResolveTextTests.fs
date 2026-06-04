module FastCoref.Tests.ResolveTextTests

open Xunit
open Swensen.Unquote
open FastCoref
open FastCoref.Api
open FastCoref.Clustering

// ---- Helpers --------------------------------------------------------------

let private mkSpan (s: int) (e: int) : CharSpan =
    { Start = CharIdx.ofInt s
      End = CharIdx.ofInt e }

let private mkMention (s: int) (e: int) (text: string) : Mention =
    { Span = mkSpan s e; Text = text }

let private mkCluster (head: Mention) (rest: Mention list) : Cluster<Mention> =
    { Head = head; Rest = rest }

let private mkResult (text: string) (clusters: Cluster<Mention> list) : CorefResult =
    { Text = text
      Clusters = clusters
      Logits = CorefLogits.empty }

// ---- Tests ----------------------------------------------------------------

[<Fact>]
let ``ResolveText: single cluster, single substitution`` () =
    // "Alice walked. She smiled."
    //  0    5         14 17
    let result =
        mkResult
            "Alice walked. She smiled."
            [ mkCluster (mkMention 0 5 "Alice") [ mkMention 14 17 "She" ] ]

    test <@ result.ResolveText() = "Alice walked. Alice smiled." @>

[<Fact>]
let ``ResolveText: two non-overlapping clusters preserved in order`` () =
    // "John saw Mary. He waved at her."
    //  0    5   9    14 15  18    27  30
    let result =
        mkResult
            "John saw Mary. He waved at her."
            [ mkCluster (mkMention 0 4 "John") [ mkMention 15 17 "He" ]
              mkCluster (mkMention 9 13 "Mary") [ mkMention 27 30 "her" ] ]

    test <@ result.ResolveText() = "John saw Mary. John waved at Mary." @>

[<Fact>]
let ``ResolveText: overlapping mentions, leftmost-longest wins`` () =
    // Both Rest spans share Start=0; lengths 7 vs 3.
    // Sort key (Start asc, Length desc) puts "her car" (len 7) first;
    // "her" (len 3, Start=0 < cursor=7) is then skipped.
    // Head spans are unused for substitution; (0,0) is a fine dummy.
    let result =
        mkResult
            "her car"
            [ mkCluster (mkMention 0 0 "Alice") [ mkMention 0 7 "her car" ]
              mkCluster (mkMention 0 0 "She") [ mkMention 0 3 "her" ] ]

    test <@ result.ResolveText() = "Alice" @>

[<Fact>]
let ``ResolveText: no clusters returns input verbatim`` () =
    let result = mkResult "unchanged input." []
    test <@ result.ResolveText() = "unchanged input." @>

[<Fact>]
let ``ResolveText: replacement at start and end of string`` () =
    // "She talks to her."
    //  0   3        13  16
    let result =
        mkResult
            "She talks to her."
            [ mkCluster
                  (mkMention 0 0 "Alice")
                  [ mkMention 0 3 "She"
                    mkMention 13 16 "her" ] ]

    test <@ result.ResolveText() = "Alice talks to Alice." @>

[<Fact>]
let ``ResolveText: preserves surrounding punctuation and spaces`` () =
    // "X, Y, and Z."
    //  0  3     10 11
    let result =
        mkResult
            "X, Y, and Z."
            [ mkCluster
                  (mkMention 0 0 "W")
                  [ mkMention 0 1 "X"
                    mkMention 3 4 "Y"
                    mkMention 10 11 "Z" ] ]

    test <@ result.ResolveText() = "W, W, and W." @>

// swift-tools-version:6.0
import PackageDescription

// Embed Info.plist so macOS shows a proper reason when asking for microphone access.
let plist = Context.packageDirectory + "/Sources/minutes-rec/Info.plist"

let package = Package(
    name: "minutes-rec",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "minutes-rec",
            path: "Sources/minutes-rec",
            exclude: ["Info.plist"],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", plist])
            ]
        )
    ],
    swiftLanguageModes: [.v5]
)

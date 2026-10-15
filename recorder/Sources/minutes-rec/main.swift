// minutes-rec: records the microphone and system (call) audio as two aligned
// 16 kHz mono WAV files, using ScreenCaptureKit. Nothing leaves the Mac.
//
//   minutes-rec record <dir> [--no-mic] [--no-system]   Ctrl+C to stop
//   minutes-rec convert <in-audio> <out.wav>             any audio -> 16 kHz mono WAV
//   minutes-rec check                                    permission status

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

let targetRate = 16_000.0

func log(_ msg: String) {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
}

func fail(_ msg: String, code: Int32 = 1) -> Never {
    log("minutes-rec: \(msg)")
    exit(code)
}

// MARK: - Track writer

/// Writes one track as a 16 kHz mono 16-bit WAV, keeping it aligned to a shared clock
/// by padding gaps with silence and trimming when the device clock runs ahead.
final class TrackWriter {
    private var file: AVAudioFile?
    private let outFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var inFormat: AVAudioFormat?
    private(set) var framesWritten: AVAudioFramePosition = 0
    private var sumSquares = 0.0
    private var sampleCount = 0

    init(url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: targetRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let f = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        file = f
        outFormat = f.processingFormat
    }

    var seconds: Double { Double(framesWritten) / targetRate }

    /// RMS level in dBFS since the previous call.
    func takeLevel() -> Double {
        defer { sumSquares = 0; sampleCount = 0 }
        guard sampleCount > 0 else { return -120 }
        return 10 * log10(max(sumSquares / Double(sampleCount), 1e-12))
    }

    /// `position` is where the buffer starts on the shared timeline, in output frames.
    /// Pass nil to append sequentially (file conversion).
    func append(_ buf: AVAudioPCMBuffer, at position: AVAudioFramePosition? = nil) throws {
        guard let file else { return }
        if inFormat == nil || inFormat! != buf.format {
            guard let c = AVAudioConverter(from: buf.format, to: outFormat) else {
                throw NSError(domain: "minutes-rec", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "cannot convert from \(buf.format)"])
            }
            c.downmix = true
            converter = c
            inFormat = buf.format
        }
        let ratio = outFormat.sampleRate / buf.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buf.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        converter!.convert(to: out, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buf
        }
        if let error { throw error }

        if let position {
            // Line up the first buffer exactly; after that, only correct drift beyond 20 ms.
            let tolerance = framesWritten == 0 ? 0 : AVAudioFramePosition(targetRate * 0.02)
            let drift = position - framesWritten
            if drift > tolerance {
                try writeSilence(drift)
            } else if drift < -tolerance {
                let skip = Int(min(-drift, AVAudioFramePosition(out.frameLength)))
                let p = out.floatChannelData![0]
                let keep = Int(out.frameLength) - skip
                if keep > 0 { memmove(p, p + skip, keep * MemoryLayout<Float>.size) }
                out.frameLength = AVAudioFrameCount(keep)
            }
        }
        guard out.frameLength > 0 else { return }
        let p = out.floatChannelData![0]
        for i in 0..<Int(out.frameLength) { sumSquares += Double(p[i] * p[i]) }
        sampleCount += Int(out.frameLength)
        try file.write(from: out)
        framesWritten += AVAudioFramePosition(out.frameLength)
    }

    private func writeSilence(_ frames: AVAudioFramePosition) throws {
        guard let file else { return }
        var left = frames
        while left > 0 {
            let n = AVAudioFrameCount(min(left, AVAudioFramePosition(targetRate)))
            guard let buf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: n) else { return }
            buf.frameLength = n
            memset(buf.floatChannelData![0], 0, Int(n) * MemoryLayout<Float>.size)
            try file.write(from: buf)
            framesWritten += AVAudioFramePosition(n)
            left -= AVAudioFramePosition(n)
        }
    }

    func finish() { file = nil }  // AVAudioFile finalises the WAV header on release
}

func pcmBuffer(from sb: CMSampleBuffer) -> AVAudioPCMBuffer? {
    guard let fd = CMSampleBufferGetFormatDescription(sb) else { return nil }
    let format = AVAudioFormat(cmAudioFormatDescription: fd)
    let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sb))
    guard frames > 0, let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
    buf.frameLength = frames
    let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
        sb, at: 0, frameCount: Int32(frames), into: buf.mutableAudioBufferList)
    return status == noErr ? buf : nil
}

// MARK: - Recorder

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let dir: URL
    let wantMic: Bool
    let wantSystem: Bool
    let queue = DispatchQueue(label: "minutes.audio")
    var mic: TrackWriter?
    var system: TrackWriter?
    var t0: CMTime?
    var stream: SCStream?
    var startedAt = Date()
    var timer: DispatchSourceTimer?
    var stopping = false
    var micName = ""
    let tty = isatty(2) != 0

    init(dir: URL, wantMic: Bool, wantSystem: Bool) {
        self.dir = dir
        self.wantMic = wantMic
        self.wantSystem = wantSystem
    }

    func run() async {
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if wantMic {
                if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
                    _ = await AVCaptureDevice.requestAccess(for: .audio)
                }
                guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
                    fail("microphone access is off. Allow your terminal app in System Settings > Privacy & Security > Microphone.", code: 2)
                }
                micName = AVCaptureDevice.default(for: .audio)?.localizedName ?? "default"
                mic = try TrackWriter(url: dir.appendingPathComponent("mic.wav"))
            }
            // ScreenCaptureKit records the microphone too, so it needs this permission in every mode.
            if !CGPreflightScreenCaptureAccess() {
                CGRequestScreenCaptureAccess()
                fail("recording needs Screen & System Audio Recording. Allow your terminal app in System Settings > Privacy & Security > Screen & System Audio Recording, then quit and reopen the terminal.", code: 2)
            }
            if wantSystem {
                system = try TrackWriter(url: dir.appendingPathComponent("system.wav"))
            }

            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let display = content.displays.first else { fail("no display found") }
            let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
            let config = SCStreamConfiguration()
            config.width = 2
            config.height = 2
            config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
            config.capturesAudio = wantSystem
            config.sampleRate = 48_000
            config.channelCount = 2
            config.excludesCurrentProcessAudio = true
            config.captureMicrophone = wantMic

            let s = SCStream(filter: filter, configuration: config, delegate: self)
            // A screen output is required to stop ScreenCaptureKit logging dropped frames; frames are ignored.
            try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
            if wantSystem { try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue) }
            if wantMic { try s.addStreamOutput(self, type: .microphone, sampleHandlerQueue: queue) }
            try await s.startCapture()
            stream = s
            startedAt = Date()
            // Written now as well as at the end, so a crash still leaves a usable session.
            writeMeta(duration: 0)
        } catch {
            fail("could not start recording: \(error.localizedDescription)", code: 2)
        }

        log("Recording to \(dir.path)")
        if wantMic { log("  you:  \(micName)") }
        if wantSystem { log("  call: system audio") }
        log("Press Ctrl+C to stop.")
        startStatus()
    }

    func startStatus() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.5, repeating: tty ? 0.5 : 60)
        t.setEventHandler { [weak self] in self?.printStatus() }
        t.resume()
        timer = t
    }

    func printStatus() {
        let elapsed = Int(Date().timeIntervalSince(startedAt))
        let clock = String(format: "%02d:%02d:%02d", elapsed / 3600, elapsed / 60 % 60, elapsed % 60)
        func meter(_ w: TrackWriter?) -> String {
            guard let w else { return "off" }
            let db = w.takeLevel()
            let n = max(0, min(10, Int(((db + 60) / 50 * 10).rounded())))
            return String(repeating: "█", count: n) + String(repeating: "·", count: 10 - n)
        }
        let line = "● REC \(clock)   you \(meter(mic))   call \(meter(system))"
        if tty {
            FileHandle.standardError.write(Data("\r\u{1B}[2K\(line)".utf8))
        } else {
            log(line)
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard !stopping, sb.isValid else { return }
        let writer: TrackWriter?
        switch type {
        case .audio: writer = system
        case .microphone: writer = mic
        default: return
        }
        guard let writer, let buf = pcmBuffer(from: sb) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        if t0 == nil { t0 = pts }
        let offset = CMTimeGetSeconds(CMTimeSubtract(pts, t0!))
        do {
            try writer.append(buf, at: AVAudioFramePosition((offset * targetRate).rounded()))
        } catch {
            log("\nminutes-rec: write error: \(error.localizedDescription)")
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("\nminutes-rec: capture stopped: \(error.localizedDescription)")
        Task { await self.stop(code: 1) }
    }

    func stop(code: Int32 = 0) async {
        var already = false
        queue.sync {
            already = stopping
            stopping = true
            timer?.cancel()
        }
        if already { return }
        try? await stream?.stopCapture()
        var duration = 0.0
        queue.sync {
            duration = max(mic?.seconds ?? 0, system?.seconds ?? 0)
            mic?.finish()
            system?.finish()
        }
        if tty { log("") }
        guard stream != nil else {
            log("Stopped before recording began.")
            exit(code)
        }
        writeMeta(duration: duration)
        log(String(format: "Stopped. %.0f s recorded.", duration))
        exit(code)
    }

    func writeMeta(duration: Double) {
        var tracks: [String: String] = [:]
        if mic != nil { tracks["mic"] = "mic.wav" }
        if system != nil { tracks["system"] = "system.wav" }
        let meta: [String: Any] = [
            "started_at": ISO8601DateFormatter().string(from: startedAt),
            "duration_seconds": NSDecimalNumber(string: String(format: "%.1f", duration)),
            "tracks": tracks,
            "mic_device": micName,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: dir.appendingPathComponent("recording.json"))
        }
    }
}

// MARK: - Commands

func convert(_ input: String, _ output: String) {
    do {
        let inFile = try AVAudioFile(forReading: URL(fileURLWithPath: input))
        let writer = try TrackWriter(url: URL(fileURLWithPath: output))
        let chunk: AVAudioFrameCount = 1 << 16
        guard let buf = AVAudioPCMBuffer(pcmFormat: inFile.processingFormat, frameCapacity: chunk) else {
            fail("cannot allocate buffer")
        }
        while inFile.framePosition < inFile.length {
            try inFile.read(into: buf, frameCount: chunk)
            if buf.frameLength == 0 { break }
            try writer.append(buf)
        }
        writer.finish()
        print(String(format: "%.1f", writer.seconds))
    } catch {
        fail("convert failed: \(error.localizedDescription)")
    }
}

/// Reports permission status, asking for any permission macOS has not asked about yet,
/// so the terminal app shows up in System Settings.
func check() {
    if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
        let done = DispatchSemaphore(value: 0)
        AVCaptureDevice.requestAccess(for: .audio) { _ in done.signal() }
        done.wait()
    }
    if !CGPreflightScreenCaptureAccess() {
        CGRequestScreenCaptureAccess()
    }
    let mic: String
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: mic = "granted"
    case .denied, .restricted: mic = "denied"
    default: mic = "not asked yet"
    }
    print("microphone: \(mic)")
    print("screen & system audio recording: \(CGPreflightScreenCaptureAccess() ? "granted" : "not granted")")
    print("input device: \(AVCaptureDevice.default(for: .audio)?.localizedName ?? "none")")
}

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "record":
    guard args.count >= 2 else { fail("usage: minutes-rec record <dir> [--no-mic] [--no-system]") }
    let recorder = Recorder(dir: URL(fileURLWithPath: args[1]),
                            wantMic: !args.contains("--no-mic"),
                            wantSystem: !args.contains("--no-system"))
    var sources: [DispatchSourceSignal] = []
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        signal(sig, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: sig, queue: .global())
        src.setEventHandler { Task { await recorder.stop() } }
        src.resume()
        sources.append(src)
    }
    Task { await recorder.run() }
    dispatchMain()
case "convert":
    guard args.count == 3 else { fail("usage: minutes-rec convert <in-audio> <out.wav>") }
    convert(args[1], args[2])
case "check":
    check()
default:
    fail("usage: minutes-rec record <dir> | convert <in> <out.wav> | check")
}

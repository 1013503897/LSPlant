module;

#include "logging.hpp"

export module lsplant:jit;

import :art_method;
import :common;
import :thread;
import hook_helper;

namespace lsplant::art::jit {
enum class CompilationKind {
    kOsr [[maybe_unused]],
    kBaseline [[maybe_unused]],
    kOptimized,
};

export class Jit {
    // Captured live Jit instance: ART passes it as `thiz` whenever it JITs a hot method. The
    // traceless conversion (M-C) needs it to force-compile -- Runtime::GetJit is inlined (no
    // symbol), so we snarf the instance from these hooks instead.
    inline static Jit *captured_ = nullptr;

    inline static auto EnqueueOptimizedCompilation_ =
        "_ZN3art3jit3Jit27EnqueueOptimizedCompilationEPNS_9ArtMethodEPNS_6ThreadE"_sym.hook->*[]
        <MemBackup auto backup>
        (Jit *thiz, ArtMethod *method, Thread *self) static -> void {
            captured_ = thiz;
            if (auto target = IsBackup(method); target) [[unlikely]] {
                LOGD("Propagate enqueue compilation: %p -> %p", method, target);
                method = target;
            }
            return backup(thiz, method, self);
        };

    // Separate resolved callable for the SAME symbol, so the conversion can invoke it directly to
    // force-compile a method (goes through the hook above transparently for non-backup methods).
    inline static auto EnqueueOptimizedCompilationCall_ =
        "_ZN3art3jit3Jit27EnqueueOptimizedCompilationEPNS_9ArtMethodEPNS_6ThreadE"_sym
            .as<void(Jit *, ArtMethod *, Thread *)>;

    inline static auto AddCompileTask_ =
        "_ZN3art3jit3Jit14AddCompileTaskEPNS_6ThreadEPNS_9ArtMethodENS_15CompilationKindEb"_sym.hook->*[]
        <MemBackup auto backup>
        (Jit *thiz, Thread *self, ArtMethod *method, CompilationKind compilation_kind, bool precompile) static -> void {
            captured_ = thiz;
            if (compilation_kind == CompilationKind::kOptimized && !precompile) {
                if (auto b = IsHooked(method); b) [[unlikely]] {
                    LOGD("Propagate compile task: %p -> %p", method, b);
                    method = b;
                }
            }
            return backup(thiz, self, method, compilation_kind, precompile);
        };

    // Universal compile sink: EVERY JIT compilation funnels through Jit::CompileMethod regardless of
    // the enqueue path (nterp-hotness, OSR, optimized). The nterp -> JIT enqueue helper has no stable
    // symbol across versions, so this is the reliable place to snarf the live Jit instance for M-C.
    inline static auto CompileMethod_ =
        "_ZN3art3jit3Jit13CompileMethodEPNS_9ArtMethodEPNS_6ThreadENS_15CompilationKindEb"_sym.hook->*[]
        <MemBackup auto backup>
        (Jit *thiz, ArtMethod *method, Thread *self, CompilationKind compilation_kind, bool osr) static -> bool {
            captured_ = thiz;
            return backup(thiz, method, self, compilation_kind, osr);
        };

public:
    // Force `method` onto the JIT optimized-compile queue (best effort). Needs the captured Jit
    // instance (some method must have been JIT'd naturally first). Returns false if not yet captured.
    static bool ForceOptimizedCompile(ArtMethod *method, Thread *self) {
        if (!captured_ || !EnqueueOptimizedCompilationCall_) return false;
        EnqueueOptimizedCompilationCall_(captured_, method, self);
        return true;
    }
    static bool HasCapturedJit() { return captured_ != nullptr; }
    static bool Init(const HookHandler &handler) {
        // NOTE: these were historically gated to <= Android 14 (U). The mangled symbols
        // (EnqueueOptimizedCompilation / AddCompileTask / CompileMethod) are unchanged on Android
        // 15/16 -- verified present in this device's libart -- so install unconditionally. handler()
        // is best-effort: a symbol that doesn't resolve is skipped, never a crash. Without these the
        // M-C traceless conversion can never capture the live Jit (every convert bails to in-place).
        handler(EnqueueOptimizedCompilation_);
        handler(AddCompileTask_);
        handler(CompileMethod_);  // universal capture point for the live Jit instance
        handler(EnqueueOptimizedCompilationCall_);  // resolve the direct-call alias
        LOGI("jit capture hooks registered (sdk=%d)", GetAndroidApiLevel());
        return true;
    }
};
}  // namespace lsplant::art::jit

module;

#include <jni.h>
#include <parallel_hashmap/phmap.h>
#include <sys/system_properties.h>

#include <functional>
#include <list>
#include <shared_mutex>
#include <string_view>

#include "logging.hpp"

export module lsplant:common;
export import jni_helper;
export import hook_helper;

export namespace lsplant {

namespace art {
class ArtMethod;
namespace mirror {
class Class;
}
namespace dex {
class ClassDef {};
}  // namespace dex

}  // namespace art

enum class Arch {
    kArm,
    kArm64,
    kX86,
    kX86_64,
    kRiscv64,
};

consteval inline Arch GetArch() {
#if defined(__i386__)
    return Arch::kX86;
#elif defined(__x86_64__)
    return Arch::kX86_64;
#elif defined(__arm__)
    return Arch::kArm;
#elif defined(__aarch64__)
    return Arch::kArm64;
#elif defined(__riscv)
    return Arch::kRiscv64;
#else
#error "unsupported architecture"
#endif
}

template <class K, class V, class Hash = phmap::priv::hash_default_hash<K>,
          class Eq = phmap::priv::hash_default_eq<K>,
          class Alloc = phmap::priv::Allocator<phmap::priv::Pair<const K, V>>, size_t N = 4>
using SharedHashMap = phmap::parallel_flat_hash_map<K, V, Hash, Eq, Alloc, N, std::shared_mutex>;

template <class T, class Hash = phmap::priv::hash_default_hash<T>,
          class Eq = phmap::priv::hash_default_eq<T>, class Alloc = phmap::priv::Allocator<T>,
          size_t N = 4>
using SharedHashSet = phmap::parallel_flat_hash_set<T, Hash, Eq, Alloc, N, std::shared_mutex>;

constexpr auto kArch = GetArch();

template <typename T>
constexpr inline auto RoundUpTo(T v, size_t size) {
    return v + size - 1 - ((v + size - 1) & (size - 1));
}

[[gnu::const]] inline auto GetAndroidApiLevel() {
    static auto kApiLevel = []() {
        std::array<char, PROP_VALUE_MAX> prop_value;
        __system_property_get("ro.build.version.sdk", prop_value.data());
        int base = atoi(prop_value.data());
        __system_property_get("ro.build.version.preview_sdk", prop_value.data());
        return base + atoi(prop_value.data());
    }();
    return kApiLevel;
}

inline auto IsJavaDebuggable(JNIEnv * env) {
    static auto kDebuggable = [&env]() {
        auto sdk_int = GetAndroidApiLevel();
        if (sdk_int < __ANDROID_API_P__) {
            return false;
        }
        auto runtime_class = JNI_FindClass(env, "dalvik/system/VMRuntime");
        if (!runtime_class) {
            LOGE("Failed to find VMRuntime");
            return false;
        }
        auto get_runtime_method = JNI_GetStaticMethodID(env, runtime_class, "getRuntime",
                                                        "()Ldalvik/system/VMRuntime;");
        if (!get_runtime_method) {
            LOGE("Failed to find VMRuntime.getRuntime()");
            return false;
        }
        auto is_debuggable_method =
            JNI_GetMethodID(env, runtime_class, "isJavaDebuggable", "()Z");
        if (!is_debuggable_method) {
            LOGE("Failed to find VMRuntime.isJavaDebuggable()");
            return false;
        }
        auto runtime = JNI_CallStaticObjectMethod(env, runtime_class, get_runtime_method);
        if (!runtime) {
            LOGE("Failed to get VMRuntime");
            return false;
        }
        bool is_debuggable = JNI_CallBooleanMethod(env, runtime, is_debuggable_method);
        LOGD("java runtime debuggable %s", is_debuggable ? "true" : "false");
        return is_debuggable;
    }();
    return kDebuggable;
}

constexpr auto kPointerSize = sizeof(void *);

SharedHashMap<art::ArtMethod *, std::pair<jobject, art::ArtMethod *>> hooked_methods_;

SharedHashMap<const art::dex::ClassDef *, phmap::flat_hash_set<art::ArtMethod *>>
    hooked_classes_;

SharedHashSet<art::ArtMethod *> deoptimized_methods_set_;

SharedHashMap<const art::dex::ClassDef *, phmap::flat_hash_set<art::ArtMethod *>>
    deoptimized_classes_;

std::list<std::pair<art::ArtMethod *, art::ArtMethod *>> jit_movements_;
std::shared_mutex jit_movements_lock_;

inline art::ArtMethod *IsHooked(art::ArtMethod * art_method, bool including_backup = false) {
    art::ArtMethod *backup = nullptr;
    hooked_methods_.if_contains(art_method, [&backup, &including_backup](const auto &it) {
        if (including_backup || it.second.first) backup = it.second.second;
    });
    return backup;
}

inline art::ArtMethod *IsBackup(art::ArtMethod * art_method) {
    art::ArtMethod *backup = nullptr;
    hooked_methods_.if_contains(art_method, [&backup](const auto &it) {
        if (!it.second.first) backup = it.second.second;
    });
    return backup;
}

inline bool IsDeoptimized(art::ArtMethod * art_method) {
    return deoptimized_methods_set_.contains(art_method);
}

inline std::list<std::pair<art::ArtMethod *, art::ArtMethod *>> GetJitMovements() {
    std::unique_lock lk(jit_movements_lock_);
    return std::move(jit_movements_);
}

inline void RecordHooked(art::ArtMethod * target, const art::dex::ClassDef *class_def,
                         jobject reflected_backup, art::ArtMethod *backup) {
    hooked_classes_.lazy_emplace_l(
        class_def, [&target](auto &it) { it.second.emplace(target); },
        [&class_def, &target](const auto &ctor) {
            ctor(class_def, phmap::flat_hash_set<art::ArtMethod *>{target});
        });
    hooked_methods_.insert({std::make_pair(target, std::make_pair(reflected_backup, backup)),
                            std::make_pair(backup, std::make_pair(nullptr, target))});
}

// Stop tracking `target` as hooked: ART will then JIT-compile it and LSPlant's instrumentation
// hooks will no longer re-pin entry==trampoline (the traceless conversion relies on this so the
// method's entry can become real JIT code). Erases both target->backup and backup->target.
inline void UnrecordHooked(art::ArtMethod * target) {
    art::ArtMethod *backup = nullptr;
    hooked_methods_.if_contains(target, [&backup](const auto &it) { backup = it.second.second; });
    hooked_methods_.erase(target);
    if (backup) hooked_methods_.erase(backup);
    deoptimized_methods_set_.erase(target);
}

inline void RecordDeoptimized(const art::dex::ClassDef *class_def, art::ArtMethod *art_method) {
    { deoptimized_classes_[class_def].emplace(art_method); }
    deoptimized_methods_set_.insert(art_method);
}

inline void RecordJitMovement(art::ArtMethod * target, art::ArtMethod * backup) {
    std::unique_lock lk(jit_movements_lock_);
    jit_movements_.emplace_back(target, backup);
}

// --- Traceless (SSOL) trap registry: JIT-move-follow (M-B) -----------------------------------
// A traceless hook traps the target method's compiled quick-entry CODE page. ART's JIT cache GC
// can later FREE or RECOMPILE that body, moving the target's entry to a different page. The old
// page is then recycled by ART for unrelated code while our KPM trap + static instruction
// snapshot still point at it -> the snapshot mis-simulates the recycled code -> SIGILL. To follow
// the move we remember every live traceless trap and, right after each JIT GC, re-read each
// target's current entry; if it moved we disarm the stale trap and (if the new entry is still a
// trappable body) re-arm at the new page. `target`/`backup` are ArtMethod* (incomplete here --
// the ArtMethod work is done in lsplant.cc, which has the complete type). `qc` is the currently
// trapped code address; `trampoline` is the hook entry to re-route to on re-arm.
struct TracelessTrap {
    art::ArtMethod *target;
    void *qc;
    void *trampoline;
    art::ArtMethod *backup;
};
std::list<TracelessTrap> traceless_traps_;
std::shared_mutex traceless_traps_lock_;

// Record (or update, on re-hook of the same target) a live traceless trap.
inline void RecordTracelessTrap(art::ArtMethod * target, void *qc, void *trampoline,
                                art::ArtMethod *backup) {
    std::unique_lock lk(traceless_traps_lock_);
    for (auto &t : traceless_traps_)
        if (t.target == target) {
            t.qc = qc;
            t.trampoline = trampoline;
            t.backup = backup;
            return;
        }
    traceless_traps_.emplace_back(TracelessTrap{target, qc, trampoline, backup});
}

// Drop a traceless trap from the registry (on real unhook) so revalidation never touches a freed
// backup. Returns the trapped qc (0 if not tracked) so the caller can disarm the KPM trap.
inline void *ForgetTracelessTrap(art::ArtMethod * target) {
    std::unique_lock lk(traceless_traps_lock_);
    for (auto it = traceless_traps_.begin(); it != traceless_traps_.end(); ++it)
        if (it->target == target) {
            void *qc = it->qc;
            traceless_traps_.erase(it);
            return qc;
        }
    return nullptr;
}

// Set by Init (lsplant.cc) when the traceless backend is configured; invoked by the JIT-GC hook
// (jit_code_cache.cxx) AFTER the collection runs. Empty on normal (non-traceless) builds, where
// it is never called -> zero behavior change.
std::function<void()> on_jit_gc_revalidate_;
}  // namespace lsplant

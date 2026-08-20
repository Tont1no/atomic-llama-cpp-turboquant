#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

int main() {
    // This is the exact guarded-wrapper command, with the parser's CPU-safe
    // non-offload sentinel substituted for CUDA0. It proves that every option,
    // especially server-scoped --kv-unified, belongs to the selected example.
    std::vector<std::string> arguments = {
        "test-qwen38-recurrent-parity", "--model", "target.gguf",
        "--ctx-size", "2048", "--parallel", "1",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "on", "--batch-size", "2048", "--ubatch-size", "128",
        "--n-gpu-layers", "all", "--device", "none", "--split-mode", "none",
        "--fit", "off", "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--no-cache-idle-slots", "--no-cache-prompt", "--no-webui", "--metrics",
        "--reasoning", "off", "-lv", "4", "--kv-unified",
    };
    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }

    const auto temp_root = std::filesystem::temp_directory_path() /
            ("qwen38-parity-cli-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto config_dir = temp_root / "llama.cpp";
    std::filesystem::create_directories(config_dir);
    {
        std::ofstream config(config_dir / "config.ini");
        config << "spec-type = draft-mtp\n";
        require(config.good());
    }

#ifdef _WIN32
    const char * config_env_name = "APPDATA";
#else
    const char * config_env_name = "XDG_CONFIG_HOME";
#endif
    const char * previous_ptr = std::getenv(config_env_name);
    const std::string previous = previous_ptr != nullptr ? previous_ptr : "";
#ifdef _WIN32
    require(_putenv_s(config_env_name, temp_root.string().c_str()) == 0);
#else
    require(setenv(config_env_name, temp_root.string().c_str(), 1) == 0);
#endif

    common_params contaminated;
    require(common_params_parse((int) argv.size(), argv.data(), contaminated, LLAMA_EXAMPLE_SERVER));
    require(contaminated.speculative.types ==
            std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_DRAFT_MTP });

    common_params params;
    require(common_params_parse_no_system_config(
            (int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
    require(params.model.path == "target.gguf");
    require(params.n_ctx == 2048);
    require(params.n_parallel == 1);
    require(params.cache_type_k == GGML_TYPE_Q8_0);
    require(params.cache_type_v == GGML_TYPE_Q8_0);
    require(params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
    require(params.n_batch == 2048);
    require(params.n_ubatch == 128);
    require(params.n_gpu_layers == -2);
    require(params.devices.size() == 1 && params.devices[0] == nullptr);
    require(params.split_mode == LLAMA_SPLIT_MODE_NONE);
    require(params.kv_unified);
    require(!params.fit_params);
    require(params.cache_ram_mib == 0);
    require(params.n_ctx_checkpoints == 0);
    require(!params.cache_idle_slots);
    require(!params.cache_prompt);
    require(!params.ui);
    require(params.endpoint_metrics);
    require(params.enable_reasoning == 0);
    require(params.verbosity == 4);
    require(params.speculative.types ==
            std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE });

#ifdef _WIN32
    require(_putenv_s(config_env_name, previous.c_str()) == 0);
#else
    if (previous_ptr != nullptr) {
        require(setenv(config_env_name, previous.c_str(), 1) == 0);
    } else {
        require(unsetenv(config_env_name) == 0);
    }
#endif
    std::filesystem::remove_all(temp_root);
    return 0;
}

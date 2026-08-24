import pytest
from utils import *
import base64
import requests
import struct
import time

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12
CACHE_SCOPE_A = "v1_" + "a" * 64
CACHE_SCOPE_B = "v1_" + "b" * 64
CACHE_SCOPE_PURGE = "v1_" + "c" * 64
CACHE_SCOPE_TTL = "v1_" + "d" * 64
DRAFT_MODEL_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join("tmp", "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # Legacy files have no scope metadata. They still restore for compatibility,
    # but fail closed and cannot seed a scoped completion.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 21


def test_slot_restore_rejects_different_scope():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "slot_scope_bound.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_scope_bound.bin",
        "cache_key": CACHE_SCOPE_B,
    })
    assert res.status_code == 400
    assert CACHE_SCOPE_A not in str(res.body)
    assert CACHE_SCOPE_B not in str(res.body)


def test_slot_files_fail_closed_with_draft_context():
    global server
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = "./tmp"
    server.model_draft = download_file(DRAFT_MODEL_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"
    server.start()

    completion = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert completion.status_code == 200

    save = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "must_not_save_target_only.bin",
    })
    assert save.status_code == 501

    restore = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "must_not_restore_target_only.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert restore.status_code == 501



def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


def test_prompt_cache_scope_isolation():
    global server
    server.start()

    prompt = "What is the capital of France?"
    cache_key = CACHE_SCOPE_A

    first = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert first.status_code == 200
    assert first.body["timings"]["cache_n"] == 0
    assert cache_key not in str(first.body)

    same_scope = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert same_scope.status_code == 200
    assert same_scope.body["timings"]["cache_n"] > 0

    different_scope = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_B,
    })
    assert different_scope.status_code == 200
    assert different_scope.body["timings"]["cache_n"] == 0

    restored_scope = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert restored_scope.status_code == 200
    assert restored_scope.body["timings"]["cache_n"] > 0


def test_unscoped_prompt_is_not_retained():
    global server
    server.start()

    prompt = "What is the capital of France?"
    for _ in range(2):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] == 0


def test_prompt_cache_scope_purge():
    global server
    server.start()

    prompt = "What is the capital of France?"
    cache_key = CACHE_SCOPE_PURGE

    first = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert first.status_code == 200

    purge = server.make_request("POST", "/v1/chat/completions/control", data={
        "action": "cache_purge",
        "cache_key": cache_key,
    })
    assert purge.status_code == 200
    assert purge.body["success"] is True

    after_purge = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert after_purge.status_code == 200
    assert after_purge.body["timings"]["cache_n"] == 0


def test_prompt_cache_scope_active_purge_is_reported_pending():
    global server
    server.start()

    url = f"http://{server.server_host}:{server.server_port}/completion"
    active = requests.post(url, json={
        "prompt": "Count forever: 1, 2, 3,",
        "n_predict": 4096,
        "stream": True,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_PURGE,
    }, stream=True)
    try:
        assert active.status_code == 200
        purge = server.make_request("POST", "/v1/chat/completions/control", data={
            "action": "cache_purge",
            "cache_key": CACHE_SCOPE_PURGE,
        })
        assert purge.status_code == 200
        assert purge.body["success"] is False
        assert purge.body["pending"] is True
    finally:
        active.close()


@pytest.mark.parametrize("cache_key", [
    "v1_" + "a" * 63,
    "v1_" + "A" * 64,
    "v2_" + "a" * 64,
    "not-a-scope",
])
def test_prompt_cache_scope_rejects_invalid_keys(cache_key):
    global server
    server.start()

    completion = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert completion.status_code == 400

    purge = server.make_request("POST", "/v1/chat/completions/control", data={
        "action": "cache_purge",
        "cache_key": cache_key,
    })
    assert purge.status_code == 400


def test_prompt_cache_scope_ttl(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_PROMPT_CACHE_TTL_MS", "20")
    server.start()

    prompt = "What is the capital of France?"
    cache_key = CACHE_SCOPE_TTL

    first = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert first.status_code == 200

    # The queue's idle maintenance tick is one second. Verify expiry happens
    # before another completion is submitted, not merely on the next lookup.
    time.sleep(1.2)

    saved = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "slot_expired.bin",
    })
    assert saved.status_code == 200
    assert saved.body["n_saved"] == 0

    expired = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": cache_key,
    })
    assert expired.status_code == 200
    assert expired.body["timings"]["cache_n"] == 0


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join("tmp", "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
        "cache_key": CACHE_SCOPE_A,
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "cache_key": CACHE_SCOPE_A,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content

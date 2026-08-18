#include "server-task.h"

#include <cassert>

int main() {
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_COMPLETION, 0) == 1);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_COMPLETION, 2) == 3);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_EMBEDDING, 0) == 1);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_RERANK, 0) == 1);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_INFILL, 0) == 1);

    assert(server_inference_slot_demand(SERVER_TASK_TYPE_METRICS, 3) == 0);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_CONTROL, 3) == 0);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_CANCEL, 3) == 0);
    assert(server_inference_slot_demand(SERVER_TASK_TYPE_SLOT_SAVE, 3) == 0);

    return 0;
}

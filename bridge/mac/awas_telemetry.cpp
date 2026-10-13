// AWAS telemetry plugin for ETS2 on macOS (and Linux).
//
// The Windows bridge reads telemetry through scs-sdk-plugin's shared memory,
// which has no macOS build. This plugin is the Mac stand-in, and smaller: it
// registers the handful of channels AWAS uses and sends them as one JSON line
// over UDP to 127.0.0.1:8098, where awas_bridge.py picks them up and forwards
// them to the Pi with the road view. Nothing leaves the laptop from here.
//
// Build (x86_64: ETS2 for Mac is an Intel binary, run under Rosetta):
//     ./build_plugin.sh
// The game loads every library in the "plugins" folder next to its binary:
//     Euro Truck Simulator 2.app/Contents/MacOS/plugins/awas_telemetry.dylib

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>

#include "scssdk_telemetry.h"
#include "eurotrucks2/scssdk_eut2.h"
#include "eurotrucks2/scssdk_telemetry_eut2.h"
#include "common/scssdk_telemetry_common_channels.h"
#include "common/scssdk_telemetry_truck_common_channels.h"

namespace {

const int PORT = 8098;
const double SEND_EVERY_SEC = 0.05;    // 20 Hz, the bridge's own telemetry rate

struct State {
    float speed = 0.0f;                // m/s, negative when reversing
    float steer = 0.0f;                // effective steering, -1..1
    bool engine = false;
    int gear = 0;                      // <0 reverse, 0 neutral, >0 forward
    bool lblinker = false;             // the stalk, not the flashing lamp
    bool rblinker = false;
    float speed_limit = 0.0f;          // m/s, 0 when none
    bool paused = true;                // the SDK starts paused
};

State state;
int sock = -1;
sockaddr_in dest {};
double last_sent = -1.0;
scs_log_t game_log = nullptr;

double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void log_message(const char *text) {
    if (game_log) game_log(SCS_LOG_TYPE_message, text);
}

void send_raw(const char *line) {
    if (sock >= 0)
        sendto(sock, line, strlen(line), 0, reinterpret_cast<const sockaddr *>(&dest), sizeof dest);
}

void send_state() {
    char line[384];
    snprintf(line, sizeof line,
             "{\"v\":1,\"speed\":%.3f,\"steer\":%.4f,\"engine\":%s,\"paused\":%s,"
             "\"gear\":%d,\"lblinker\":%s,\"rblinker\":%s,\"speed_limit\":%.3f}",
             state.speed, state.steer, state.engine ? "true" : "false",
             state.paused ? "true" : "false", state.gear,
             state.lblinker ? "true" : "false", state.rblinker ? "true" : "false",
             state.speed_limit);
    send_raw(line);
    last_sent = now_sec();
}

// ------------------------------------------------------------ callbacks

SCSAPI_VOID on_frame_end(const scs_event_t, const void *const, const scs_context_t) {
    if (now_sec() - last_sent >= SEND_EVERY_SEC) send_state();
}

SCSAPI_VOID on_pause(const scs_event_t event, const void *const, const scs_context_t) {
    state.paused = (event == SCS_TELEMETRY_EVENT_paused);
    send_state();                      // frames may stop while paused; say so now
}

SCSAPI_VOID store_float(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                        const scs_context_t context) {
    *static_cast<float *>(context) = value ? value->value_float.value : 0.0f;
}

SCSAPI_VOID store_bool(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                       const scs_context_t context) {
    *static_cast<bool *>(context) = value ? (value->value_bool.value != 0) : false;
}

SCSAPI_VOID store_s32(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                      const scs_context_t context) {
    *static_cast<int *>(context) = value ? value->value_s32.value : 0;
}

}  // namespace

// ------------------------------------------------------------ SDK entry points

SCSAPI_RESULT scs_telemetry_init(const scs_u32_t version, const scs_telemetry_init_params_t *const params) {
    if (version != SCS_TELEMETRY_VERSION_1_00 && version != SCS_TELEMETRY_VERSION_1_01)
        return SCS_RESULT_unsupported;
    const auto *p = static_cast<const scs_telemetry_init_params_v100_t *>(params);
    game_log = p->common.log;

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        if (game_log) game_log(SCS_LOG_TYPE_error, "[awas] could not open a UDP socket");
        return SCS_RESULT_generic_error;
    }
    dest.sin_family = AF_INET;
    dest.sin_port = htons(PORT);
    dest.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    state = State();
    last_sent = -1.0;

    bool events_ok =
        p->register_for_event(SCS_TELEMETRY_EVENT_frame_end, on_frame_end, nullptr) == SCS_RESULT_ok &&
        p->register_for_event(SCS_TELEMETRY_EVENT_paused, on_pause, nullptr) == SCS_RESULT_ok &&
        p->register_for_event(SCS_TELEMETRY_EVENT_started, on_pause, nullptr) == SCS_RESULT_ok;
    if (!events_ok) {
        if (game_log) game_log(SCS_LOG_TYPE_error, "[awas] could not register telemetry events");
        close(sock);
        sock = -1;
        return SCS_RESULT_generic_error;
    }

    // A channel the game does not offer only loses that one field.
    const auto reg = [&](const char *name, scs_value_type_t type, scs_telemetry_channel_callback_t cb, void *ctx) {
        p->register_for_channel(name, SCS_U32_NIL, type, SCS_TELEMETRY_CHANNEL_FLAG_none, cb, ctx);
    };
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_speed, SCS_VALUE_TYPE_float, store_float, &state.speed);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_effective_steering, SCS_VALUE_TYPE_float, store_float, &state.steer);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_engine_enabled, SCS_VALUE_TYPE_bool, store_bool, &state.engine);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_engine_gear, SCS_VALUE_TYPE_s32, store_s32, &state.gear);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_lblinker, SCS_VALUE_TYPE_bool, store_bool, &state.lblinker);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_rblinker, SCS_VALUE_TYPE_bool, store_bool, &state.rblinker);
    reg(SCS_TELEMETRY_TRUCK_CHANNEL_navigation_speed_limit, SCS_VALUE_TYPE_float, store_float, &state.speed_limit);

    char msg[160];
    snprintf(msg, sizeof msg, "[awas] telemetry plugin ready (%s), sending to 127.0.0.1:%d",
             p->common.game_id, PORT);
    log_message(msg);
    return SCS_RESULT_ok;
}

SCSAPI_VOID scs_telemetry_shutdown(void) {
    send_raw("{\"v\":1,\"bye\":true}");    // the game is closing: not "paused", gone
    if (sock >= 0) close(sock);
    sock = -1;
    game_log = nullptr;
}

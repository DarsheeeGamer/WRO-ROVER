#include "hub.h"

// Constructor
HubClient::HubClient(Stream& stream) : _stream(stream) {}

// --- Internal raw sender ---
bool HubClient::sendRaw(const char* json) {
    _stream.print("|");
    _stream.print(json);
    _stream.print("|");
    _stream.flush();
    return true;
}

// --- Send JSON message ---
bool HubClient::sendMessage(const char* cmd, const char* src, const char* dst, JsonObject data) {
    StaticJsonDocument<HUB_JSON_CAPACITY> doc;
    doc["cmd"] = cmd;
    doc["src"] = src;
    doc["dst"] = dst;

    if (!data.isNull()) {
        doc["data"] = data;
    }

    char out[HUB_BUFFER_SIZE];
    size_t n = serializeJson(doc, out, sizeof(out));
    if (n == 0) {
        return false;
    }

    return sendRaw(out);
}

// --- Receive a message ---
bool HubClient::receiveMessage(JsonDocument& doc) {
    static String buffer;
    while (_stream.available()) {
        char c = _stream.read();
        if (c == '|') {
            if (buffer.length() > 0) {
                DeserializationError err = deserializeJson(doc, buffer);
                buffer = "";
                return !err;
            }
        } else {
            buffer += c;
            if (buffer.length() > HUB_BUFFER_SIZE - 1) {
                buffer = "";
                return false; // overflow
            }
        }
    }
    return false; // no complete frame yet
}

// --- High-Level Wrappers ---

bool HubClient::ping(const char* src, const char* dst) {
    StaticJsonDocument<64> data;
    return sendMessage("PING", src, dst, data.to<JsonObject>());
}

bool HubClient::status(const char* src, const char* dst) {
    StaticJsonDocument<64> data;
    return sendMessage("STATUS", src, dst, data.to<JsonObject>());
}

bool HubClient::fetch(const char* src, const char* dst, const char* key) {
    StaticJsonDocument<64> data;
    data["key"] = key;
    return sendMessage("FETCH", src, dst, data.as<JsonObject>());
}

bool HubClient::post(const char* src, const char* dst, JsonObject payload) {
    return sendMessage("POST", src, dst, payload);
}

bool HubClient::data(const char* src, const char* dst, JsonObject payload) {
    return sendMessage("DATA", src, dst, payload);
}

bool HubClient::config(const char* src, const char* dst, JsonObject settings) {
    return sendMessage("CONFIG", src, dst, settings);
}

bool HubClient::ack(const char* src, const char* dst, const char* ref) {
    StaticJsonDocument<64> data;
    data["ref"] = ref;
    return sendMessage("ACK", src, dst, data.as<JsonObject>());
}

bool HubClient::err(const char* src, const char* dst, const char* message) {
    StaticJsonDocument<128> data;
    data["msg"] = message;
    return sendMessage("ERR", src, dst, data.as<JsonObject>());
}

bool HubClient::debug(const char* src, const char* dst, const char* message) {
    StaticJsonDocument<128> data;
    data["log"] = message;
    return sendMessage("DEBUG", src, dst, data.as<JsonObject>());
}

bool HubClient::trace(const char* src, const char* dst, JsonObject traceData) {
    return sendMessage("TRACE", src, dst, traceData);
}

bool HubClient::registerDevice(const char* src, const char* name) {
    StaticJsonDocument<128> data;
    data["name"] = name;
    return sendMessage("REGISTER", src, "HUB", data.as<JsonObject>());
}

bool HubClient::pull(const char* src, const char* dst) {
    StaticJsonDocument<64> data;
    return sendMessage("PULL", src, dst, data.to<JsonObject>());
}

bool HubClient::disc(const char* src) {
    StaticJsonDocument<64> data;
    return sendMessage("DISC", src, "HUB", data.to<JsonObject>());
}

bool HubClient::conn(const char* src) {
    StaticJsonDocument<64> data;
    return sendMessage("CONN", src, "HUB", data.to<JsonObject>());
}

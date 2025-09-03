#ifndef HUB_H
#define HUB_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include <Stream.h>

/**
 * HUB Protocol High-Level API
 * 
 * Message framing:
 *   |{ "cmd": "PING", "src": "x1Device", "dst": "HUB", "data": {} }|
 *
 * Each function builds and sends a JSON packet framed with `|...|`
 * and reads responses from the HUB.
 */

#define HUB_BUFFER_SIZE 1024
#define HUB_JSON_CAPACITY 2048

class HubClient {
public:
    explicit HubClient(Stream& stream);

    // === Core communication ===
    bool sendMessage(const char* cmd, const char* src, const char* dst, JsonObject data);
    bool receiveMessage(JsonDocument& doc);

    // === High-Level Commands ===
    bool ping(const char* src, const char* dst = "HUB");
    bool status(const char* src, const char* dst = "HUB");
    bool fetch(const char* src, const char* dst, const char* key);
    bool post(const char* src, const char* dst, JsonObject payload);
    bool data(const char* src, const char* dst, JsonObject payload);
    bool config(const char* src, const char* dst, JsonObject settings);
    bool ack(const char* src, const char* dst, const char* ref);
    bool err(const char* src, const char* dst, const char* message);
    bool debug(const char* src, const char* dst, const char* message);
    bool trace(const char* src, const char* dst, JsonObject traceData);
    bool registerDevice(const char* src, const char* name);
    bool pull(const char* src, const char* dst = "HUB");
    bool disc(const char* src);
    bool conn(const char* src);

private:
    Stream& _stream;
    char _buffer[HUB_BUFFER_SIZE];

    bool sendRaw(const char* json);
};

#endif // HUB_H

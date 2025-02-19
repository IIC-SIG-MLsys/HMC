/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "signal_handle.h"
#include "log.h"
#include <signal.h>
#include <iostream>

static std::atomic<bool> running(true);

void signal_handler(int sig) {
    if (sig == SIGINT) { // 捕获 Ctrl+C
        running = false;
        logError("Caught Ctrl+C, preparing to exit...");
    }
}

void setup_signal_handler() {
    std::signal(SIGINT, signal_handler);
}

bool should_exit() {
    return !running.load();
}
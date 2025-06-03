/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "log.h"
#include "signal_handle.h"
#include <iostream>
#include <signal.h>

static std::atomic<bool> running(true);

void signal_handler(int sig) {
  if (sig == SIGINT) { // capture Ctrl+C
    running = false;
    logError("Caught Ctrl+C, preparing to exit...");
    exit(-1);
  }
}

void setup_signal_handler() { std::signal(SIGINT, signal_handler); }

bool should_exit() { return !running.load(); }
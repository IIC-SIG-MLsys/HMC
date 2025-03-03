/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */

#include "parser.h"

namespace hddt {

ProtoParser::ProtoParser(std::string Proto_file, uint32_t node_rank) {
  /* parse task lists of this node from Proto file */
  
};

ProtoParser::~ProtoParser() {};

uint32_t ProtoParser::getParallelism() {
  return TaskLists.size();
}

std::forward_list<TaskMeta>* ProtoParser::getTaskList(uint32_t index) {
  return &TaskLists[index];
}

}